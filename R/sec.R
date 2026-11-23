# Sparse estimation of a correlation matrix (SEC): accelerated proximal
# gradient iterations with projection onto the positive semidefinite cone,
# rescaling to unit diagonal, and a hard threshold. Adapted from the MATLAB
# code sec.m of Cui, Leng and Sun (2016) and distributed with the permission
# of Chenlei Leng (email, November 2026).

# Projection onto the positive semidefinite cone, retried with a larger
# jitter when the eigendecomposition fails.
.projection_psd <- function(A, identity, jitter0 = 1e-12, max_retries = 5L) {
    A <- (A + A$t())$mul(0.5)
    A$nan_to_num_()
    scale <- .item(A$abs()$mean())
    base <- if (is.finite(scale) && scale > 0) scale else 1
    jitter <- jitter0 * base
    for (attempt in seq_len(max_retries + 2L)) {
        res <- tryCatch(torch::linalg_eigh(A + identity * jitter),
                        error = function(e) e)
        if (!inherits(res, "error")) {
            evals <- res[[1L]]$clamp(min = 0)
            out <- (res[[2L]] * evals$unsqueeze(1L))$matmul(res[[2L]]$t())
            return((out + out$t())$mul(0.5))
        }
        jitter <- jitter * 10
    }
    stop("PSD projection failed after ", max_retries + 2L,
         " eigendecomposition attempts.", call. = FALSE)
}

# Proximal gradient iterations. Returns the last iterate R.
.sec_iterate <- function(Rn_work, rhoW, tiny_mask, I, epsilon, tol,
                         max_iter, restart, line_search_apg) {
    Z <- torch::torch_zeros_like(Rn_work)
    Y <- Z
    t <- 1
    L <- 1
    tau <- 0.75
    eta <- 0.9
    epsI <- I * epsilon
    res_old <- NULL
    for (k in seq_len(max_iter)) {
        Yold <- Y
        told <- t
        X <- Z + Rn_work
        R <- X$sign() * (X$abs() - rhoW)$clamp(min = 0)
        R$masked_fill_(tiny_mask, 0)
        R$fill_diagonal_(1)
        R <- (R + R$t())$mul(0.5)
        step <- if (line_search_apg) tau else L
        Y <- .projection_psd(Z - (R - epsI) / step, I)
        num <- .item(torch::linalg_norm(Z - Y, ord = "fro"))
        den <- 1 + .item(torch::linalg_norm(Z, ord = "fro"))
        res_val <- if (line_search_apg) tau * num / den else num / den
        if (line_search_apg && k %% 5L == 0L && tau < L &&
            !is.null(res_old) && res_val > res_old) {
            tau <- min(L, tau / eta)
        }
        if (k > 1L) {
            res_old <- res_val
        }
        if (res_val <= tol) {
            break
        }
        if (!is.null(restart) && restart > 0L && k %% restart == 0L) {
            t <- 1
            told <- t
        }
        t <- (1 + sqrt(1 + 4 * told * told)) / 2
        Z <- Y + (Y - Yold) * ((told - 1) / t)
    }
    R
}
