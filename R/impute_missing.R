# Left-censored normal model on the centered log-ratio scale: least squares
# start, expectation maximization, and maximum likelihood by LBFGS.

# Design matrix with an intercept column; 'design' is an n x q tensor or NULL.
.build_X <- function(design, n) {
    ones <- torch::torch_ones(n, 1L, dtype = torch::torch_float64())
    if (is.null(design)) {
        return(ones)
    }
    torch::torch_cat(list(ones, design), dim = 2L)
}

# Normal-equation matrices X' diag(w_j) X for each column j, accumulated over
# chunks of columns; result is d x p' x p'.
.weighted_gram <- function(X, W, chunk_bytes = 256 * 1024^2) {
    n <- X$size(1)
    p_prime <- X$size(2)
    d <- W$size(2)
    chunk <- max(1L, as.integer(chunk_bytes %/% max(1, n * p_prime * 8)))
    X_t <- X$t()
    XtWX <- torch::torch_empty(d, p_prime, p_prime,
                               dtype = torch::torch_float64())
    for (j0 in seq(1L, d, by = chunk)) {
        j1 <- min(j0 + chunk - 1L, d)
        Xw <- (X$unsqueeze(3) * W[, j0:j1, drop = FALSE]$unsqueeze(2))$
            permute(c(3L, 1L, 2L))
        XtWX[j0:j1, , ] <- torch::torch_matmul(X_t$unsqueeze(1), Xw)
    }
    XtWX
}

# Weighted least squares fit of each column of Y (NaN marks a missing entry)
# on the design. Returns (p' + 1) x d: coefficients, then log(sigma). Columns
# that cannot be estimated are zero.
.ols_estimate <- function(Y, design = NULL, ridge_eps = 1e-8,
                          cond_warn = 1e12) {
    eps64 <- .Machine$double.eps
    X <- .build_X(design, Y$size(1))
    p_prime <- X$size(2)
    W <- torch::torch_isnan(Y)$logical_not()$to(dtype = torch::torch_float64())
    Y_filled <- Y$nan_to_num(nan = 0)
    A <- .weighted_gram(X, W)
    if (ridge_eps > 0) {
        I <- torch::torch_eye(p_prime, dtype = torch::torch_float64())
        A <- A + (I * ridge_eps)$unsqueeze(1)
    }
    B <- X$t()$matmul(W * Y_filled)$t()$unsqueeze(3)
    svals <- torch::linalg_svdvals(A)
    cond <- svals[, 1] / svals[, p_prime]$clamp(min = eps64)
    rank <- (svals > svals[, 1L, drop = FALSE] * (p_prime * eps64))$
        sum(dim = 2L)
    cond_r <- as.numeric(cond)
    well <- which(is.finite(cond_r) & cond_r < 1 / eps64)
    ill <- setdiff(seq_along(cond_r), well)
    beta_d <- torch::torch_zeros_like(B)
    if (length(well) > 0L) {
        beta_d[well, , ] <- torch::linalg_solve(A[well, , , drop = FALSE],
                                                B[well, , , drop = FALSE])
    }
    if (length(ill) > 0L) {
        beta_d[ill, , ] <- torch::linalg_pinv(A[ill, , , drop = FALSE])$
            matmul(B[ill, , , drop = FALSE])
    }
    .ols_finish(beta_d$squeeze(3)$t(), X, W, Y_filled, rank, cond,
                cond_warn)
}

# Residual scale and the zeroing of columns that cannot be estimated.
.ols_finish <- function(beta, X, W, Y_filled, rank, cond, cond_warn) {
    p_prime <- X$size(2)
    resid <- torch::torch_where(W$to(dtype = torch::torch_bool()),
                                Y_filled - X$matmul(beta),
                                torch::torch_zeros_like(Y_filled))
    sse <- resid$pow(2)$sum(dim = 1L)
    n_obs <- W$sum(dim = 1L)
    df_resid <- (n_obs - rank)$to(dtype = torch::torch_float64())$
        clamp(min = 1)
    log_sigma <- (sse / df_resid)$log()$mul(0.5)$unsqueeze(1)
    est <- torch::torch_cat(list(beta, log_sigma), dim = 1L)
    bad <- as.logical(torch::torch_isnan(est)$any(dim = 1L)) |
        as.logical(torch::torch_isinf(est)$any(dim = 1L)) |
        as.numeric(cond) > cond_warn | as.numeric(rank) < p_prime |
        as.numeric(n_obs) < p_prime
    if (any(bad)) {
        est[, bad] <- 0
    }
    est
}
