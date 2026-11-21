# Maximum likelihood fit of the left-censored normal model.

# Negative log-likelihood and its gradient at a flat parameter tensor holding
# the (p' + 1) x d matrix [coefficients; log(sigma)] in row-major order.
.tobit_objective <- function(X, Y, th, sigma_floor) {
    obs <- torch::torch_isnan(Y)$logical_not()
    obs_f <- obs$to(dtype = torch::torch_float64())
    cens <- obs$logical_not()
    zero <- torch::torch_zeros_like(Y)
    p_prime <- X$size(2)
    d <- Y$size(2)
    function(x) {
        par <- x$detach()$requires_grad_(TRUE)
        m <- par$view(c(p_prime + 1L, d))
        sigma <- m[(p_prime + 1L):(p_prime + 1L), , drop = FALSE]$exp()$
            clamp(min = sigma_floor)
        inv_sigma <- .rdiv(1, sigma)
        mu <- X$matmul(m[1:p_prime, , drop = FALSE])
        e <- torch::torch_where(obs, Y - mu, zero)
        ll_unc <- (obs_f * (sigma$log()$neg() -
                                (e * inv_sigma)$pow(2)$mul(0.5)))$sum()
        z_c <- torch::torch_masked_select((th - mu) * inv_sigma, cens)
        ll_cens <- .log_ndtr(z_c)$sum()
        loss <- (ll_unc + ll_cens)$neg()
        grad <- torch::autograd_grad(loss, par)[[1L]]
        list(loss = .item(loss), grad = grad$detach())
    }
}

# Expectation maximization start followed by LBFGS. Returns the
# (p' + 1) x d estimate; non-finite entries are replaced by 'init'.
.fit_censored_normal <- function(Y, design, th, init, max_iter = 100L,
                                 tol = 1e-6, sigma_floor = 1e-6,
                                 em_steps = 20L) {
    em <- .tobit_em(Y, design, th, init, steps = em_steps, tol = tol)
    start <- torch::torch_cat(list(em$beta, em$log_sigma), dim = 1L)
    shape <- start$size()
    fn <- .tobit_objective(em$X, Y, em$th, sigma_floor)
    params <- tryCatch(
        .lbfgs(fn, start$reshape(-1L), max_iter = max_iter,
               tolerance_grad = tol, tolerance_change = tol)$view(shape),
        error = function(e) start$clone())
    bad <- torch::torch_isfinite(params)$logical_not()
    if (.item(bad$any())) {
        params <- torch::torch_where(bad, init, params)
    }
    params
}
