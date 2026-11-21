# Selection of the SEC penalty by cross-validation and the final fit.

# Scores of the grid values c; returns the score table and rho values.
.sec_score_grid <- function(c_values, base, folds, sec_args) {
    rho <- c_values * base
    score <- vapply(rho, function(r) .sec_cv_score(folds, r, sec_args),
                    numeric(1L))
    data.frame(c = c_values, rho = rho, score = score)
}

# Value with the smallest score; ties are broken by the smaller key.
.argmin_score <- function(key, score) {
    key[order(score, key)[1L]]
}

# Cross-validated penalty, score table (ordered by rho), and the SEC
# estimate of Rn at that penalty.
.sec_cv <- function(X, Rn, c_grid, n_splits, seed, refine, refine_points,
                    sec_args) {
    n <- X$size(1)
    p <- X$size(2)
    if (n_splits < 2L || n_splits > n) {
        stop("'n_splits' must be in [2, n]; got ", n_splits, " for n = ", n,
             ".", call. = FALSE)
    }
    base <- sqrt(max(log(p) / n, 0))
    folds <- .sec_folds(X, n_splits, seed)
    c_coarse <- sort(unique(as.numeric(c_grid)))
    scores <- .sec_score_grid(c_coarse, base, folds, sec_args)
    if (refine && refine_points >= 2L && length(c_coarse) >= 2L) {
        best_c <- .argmin_score(scores$c, scores$score)
        c_new <- .refine_grid(c_coarse, best_c, refine_points)
        if (length(c_new) > 0L) {
            scores <- rbind(scores, .sec_score_grid(c_new, base, folds,
                                                    sec_args))
        }
    }
    best_rho <- .argmin_score(scores$rho, scores$score)
    .warn_grid_edge(best_rho, base, c_coarse)
    scores <- scores[order(scores$rho, method = "radix"), , drop = FALSE]
    rownames(scores) <- NULL
    R_hat <- do.call(.sec_dense, c(list(Rn = Rn, rho = best_rho), sec_args))
    list(best_rho = best_rho, scores = scores, estimate = R_hat)
}

# Warning when the selected c lies at an end of the grid.
.warn_grid_edge <- function(best_rho, base, c_coarse) {
    if (base <= 0) {
        return(invisible(NULL))
    }
    best_c <- best_rho / base
    c_min <- c_coarse[1L]
    c_max <- c_coarse[length(c_coarse)]
    range_text <- paste0("[", c_min, ", ", c_max, "]")
    if (abs(best_c - c_min) <= 1e-12) {
        warning("Best c = ", signif(best_c, 3L), " occurs at the LOWER edge ",
                "of c_grid ", range_text, ". Consider expanding c_grid to ",
                "include smaller c values.", call. = FALSE)
    } else if (abs(best_c - c_max) <= 1e-12) {
        warning("Best c = ", signif(best_c, 3L), " occurs at the UPPER edge ",
                "of c_grid ", range_text, ". Consider expanding c_grid to ",
                "include larger c values.", call. = FALSE)
    }
    invisible(NULL)
}
