# Expectation maximization and maximum likelihood for the left-censored
# normal model.

# Standard normal distribution function and density, written as in
# torch.distributions.Normal(0, 1).
.std_normal_cdf <- function(a) {
    torch::torch_erf(a$div(sqrt(2)))$add(1)$mul(0.5)
}
.std_normal_pdf <- function(a) {
    a$pow(2)$neg()$div(2)$sub(log(sqrt(2 * pi)))$exp()
}

# Expectation maximization from 'init' ((p' + 1) x d). Returns the design,
# the thresholds broadcast to n x d, the coefficients, and log(sigma) (1 x d).
.tobit_em <- function(Y, design, th, init, steps = 20L, tol = 1e-6) {
    n <- Y$size(1)
    X <- .build_X(design, n)
    p_prime <- X$size(2)
    th <- th$expand_as(Y)
    unc <- torch::torch_isnan(Y)$logical_not()
    cens <- unc$logical_not()
    beta <- init[seq_len(p_prime), , drop = FALSE]
    sigma <- init[(p_prime + 1L):(p_prime + 1L), , drop = FALSE]$exp()$
        clamp(min = 1e-8)
    qr <- torch::linalg_qr(X, mode = "reduced")
    zero <- torch::torch_zeros_like(Y)
    prev_obj <- Inf
    for (it in seq_len(steps)) {
        mu <- X$matmul(beta)
        a <- (th - mu) / sigma
        Phi <- .std_normal_cdf(a)$clamp(min = .Machine$double.eps)
        lam <- .std_normal_pdf(a) / Phi
        y_bar <- torch::torch_where(unc, Y, mu - sigma * lam)
        var_cens <- sigma$pow(2) * (.rsub(a * lam, 1) - lam * lam)
        beta_new <- torch::linalg_solve_triangular(
            qr[[2]], qr[[1]]$t()$matmul(y_bar), upper = TRUE)
        mu_new <- X$matmul(beta_new)
        sse_unc <- torch::torch_where(unc, Y - mu_new, zero)$pow(2)$
            sum(dim = 1L)
        sse_cens <- (torch::torch_where(cens, (y_bar - mu_new)$pow(2), zero) +
                         torch::torch_where(cens, var_cens, zero))$
            sum(dim = 1L)
        sse_total <- sse_unc + sse_cens
        sigma_new <- (sse_total / n)$clamp(min = 1e-16)$sqrt()$unsqueeze(1)
        obj <- .item(sse_total$sum() + sigma_new$log()$sum())
        rel_change <- abs(obj - prev_obj) / (abs(prev_obj) + 1e-12)
        prev_obj <- obj
        beta <- beta_new
        sigma <- sigma_new
        if (isTRUE(rel_change < tol)) {
            break
        }
    }
    list(X = X, th = th, beta = beta, log_sigma = sigma$log())
}
