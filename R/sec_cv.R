# K-fold cross-validation of the SEC penalty rho = c * sqrt(log(p) / n).

# Training and validation correlation matrices of each fold. Folds come from
# a permutation drawn with the seed; the first n %% K folds have one extra
# sample.
.sec_folds <- function(X, n_splits, seed) {
    n <- X$size(1)
    perm <- as.integer(.randperm_seeded(n, seed)) + 1L
    sizes <- rep(n %/% n_splits, n_splits)
    extra <- n %% n_splits
    sizes[seq_len(extra)] <- sizes[seq_len(extra)] + 1L
    ends <- cumsum(sizes)
    lapply(seq_len(n_splits), function(i) {
        val <- perm[(ends[i] - sizes[i] + 1L):ends[i]]
        train <- setdiff(seq_len(n), val)
        tr <- .corr_dense(X$index_select(1L, .as_t(train, torch::torch_long())))
        va <- .corr_dense(X$index_select(1L, .as_t(val, torch::torch_long())))
        if (!.item(torch::torch_isfinite(tr)$all()) ||
            !.item(torch::torch_isfinite(va)$all())) {
            stop("A fold correlation matrix has non-finite entries.",
                 call. = FALSE)
        }
        list(train = tr, val = va)
    })
}

# Mean over folds of the squared Frobenius distance between the SEC estimate
# on the training part and the validation correlation; the fold errors are
# summed in fold order.
.sec_cv_score <- function(folds, rho, sec_args) {
    errs <- vapply(folds, function(f) {
        R_hat <- do.call(.sec_dense, c(list(Rn = f$train, rho = rho),
                                       sec_args))
        if (!.item(torch::torch_isfinite(R_hat)$all())) {
            stop("An SEC estimate has non-finite entries.", call. = FALSE)
        }
        .item(torch::linalg_norm(R_hat - f$val, ord = "fro"))^2
    }, numeric(1L))
    Reduce(`+`, errs) / length(errs)
}

# Grid for the second pass: 'points' values between the neighbors of the
# best first-pass value, excluding values already evaluated.
.refine_grid <- function(c_coarse, best_c, points) {
    i <- match(best_c, c_coarse)
    k <- length(c_coarse)
    bounds <- if (i == 1L) c_coarse[1:2] else if (i == k) {
        c_coarse[(k - 1L):k]
    } else {
        c_coarse[c(i - 1L, i + 1L)]
    }
    if (bounds[2L] <= bounds[1L]) {
        return(numeric(0L))
    }
    step <- (bounds[2L] - bounds[1L]) / (points - 1L)
    refined <- bounds[1L] + (seq_len(points) - 1L) * step
    refined[!(round(refined, 12L) %in% round(c_coarse, 12L))]
}
