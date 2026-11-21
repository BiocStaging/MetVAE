# SEC estimate for one penalty value.

# Sparse correlation estimate (dense d x d tensor) from the correlation
# matrix Rn for the penalty rho.
#
# The columns of Rn are centered and scaled before the iterations, and the
# result is symmetrized. In Cui, Leng and Sun (2016) the standardization is
# applied to the data variables; this column standardization of the
# correlation matrix follows the implementation of the Python package metvae.
.sec_dense <- function(Rn, rho, epsilon = 1e-5, tol = 1e-3, max_iter = 1000L,
                       restart = 50L, line_search_apg = TRUE, delta = NULL,
                       n_samples = NULL, c_delta = 0.1, threshold = 0.1) {
    if (max_iter < 1L) {
        stop("'max_iter' must be at least 1.", call. = FALSE)
    }
    p <- Rn$size(1)
    f64 <- torch::torch_float64()
    zero <- torch::torch_tensor(0, dtype = f64)
    if (is.null(delta)) {
        delta <- if (!is.null(n_samples) && n_samples > 0) {
            c_delta * sqrt(max(log(p) / n_samples, 0))
        } else {
            1e-6
        }
    }
    abs_Rn <- Rn$abs()
    eye_mask <- torch::torch_eye(p, dtype = torch::torch_bool())
    tiny_mask <- abs_Rn$lt(delta)$logical_and(eye_mask$logical_not())
    W <- torch::torch_where(abs_Rn$le(delta), zero,
                            .rdiv(1, abs_Rn$clamp(min = 1e-300)))
    W$fill_diagonal_(0)
    col_mean <- Rn$mean(dim = 1L, keepdim = TRUE)
    col_std <- Rn$std(dim = 1L, unbiased = TRUE, keepdim = TRUE)$
        clamp(min = 1e-12)
    Rn_work <- (Rn - col_mean) / col_std
    Rn_work <- (Rn_work + Rn_work$t())$mul(0.5)
    I <- torch::torch_eye(p, dtype = f64)
    R <- .sec_iterate(Rn_work, W * rho, tiny_mask, I, epsilon, tol, max_iter,
                      restart, line_search_apg)
    .sec_rescale(R, I, epsilon, threshold, zero)
}

# Rescaling of the iterate to unit diagonal and the hard threshold.
.sec_rescale <- function(R, I, epsilon, threshold, zero) {
    epsI <- I * epsilon
    R_cal <- R - epsI
    lam_min <- .item(torch::linalg_eigvalsh(R_cal)$min())
    if (lam_min < 0) {
        R_cal <- R_cal + I * (-lam_min)
    }
    d <- R_cal$diag()
    if (.item(d$le(0)$any())) {
        warning("The rescaled SEC matrix has nonpositive diagonal entries; ",
                "the corresponding scaling factors are set to zero.",
                call. = FALSE)
    }
    b <- torch::torch_ones_like(d)$sub(epsilon)
    d <- (b / d$clamp(min = 1e-300))$clamp(min = 0)$sqrt()
    D <- torch::torch_diag(d)
    R_out <- D$matmul(R_cal)$matmul(D)
    R_out <- (R_out + R_out$t())$mul(0.5) + epsI
    if (threshold > 0) {
        diag_vals <- R_out$diagonal()$clone()
        R_out <- torch::torch_where(R_out$abs()$ge(threshold), R_out, zero)
        R_out$diagonal()$copy_(diag_vals)
    }
    R_out
}
