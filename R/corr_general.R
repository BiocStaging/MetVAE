# Correlation with missing log values: pairwise variance of log ratios over
# the samples where both features are finite.

# Variance over the finite entries along dimension 'dim' (population form).
.nanvar <- function(x, dim) {
    mask <- torch::torch_isfinite(x)
    count <- mask$sum(dim = dim, keepdim = TRUE)$clamp(min = 1L)
    x_filled <- x$nan_to_num(nan = 0, posinf = 0, neginf = 0)
    mean <- x_filled$sum(dim = dim, keepdim = TRUE) / count
    sq_diff <- (x_filled - mean)$pow(2) * mask
    (sq_diff$sum(dim = dim, keepdim = TRUE) / count$clamp(min = 1L))$
        squeeze(dim)
}

# Pairwise variance of log ratios (d x d) of the feature-centered logs.
.compute_vlr <- function(data) {
    log_x <- data$log()$clone()
    log_x <- torch::torch_where(torch::torch_isfinite(log_x), log_x, .nan_t())
    clr <- log_x - log_x$nanmean(dim = 1L, keepdim = TRUE)
    M <- torch::torch_isfinite(clr)$to(dtype = clr$dtype)
    Y <- clr$nan_to_num(nan = 0, posinf = 0, neginf = 0)
    A <- Y$square()
    K <- M$t()$matmul(M)
    S <- A$t()$matmul(M)
    P <- Y$t()$matmul(Y)
    vlr <- ((S + S$t()) - P$mul(2)) / K$clamp(min = 1)
    vlr <- torch::torch_where(K$gt(0), vlr, .nan_t())
    vlr <- (vlr + vlr$t())$mul(0.5)$clamp(min = 0)
    vlr$fill_diagonal_(0)
}

# Correlation matrix of an abundance tensor whose log may be non-finite.
.corr_general <- function(data, threshold = 0.2, eps = 1e-12) {
    d <- data$size(2)
    log_x <- data$log()
    log_x <- torch::torch_where(torch::torch_isfinite(log_x), log_x, .nan_t())
    clr <- log_x - log_x$nanmean(dim = 2L, keepdim = TRUE)
    vlr <- .compute_vlr(data)
    log_var <- .normalize_var(.nanvar(clr, 1L), d, eps)
    std <- log_var$sqrt()
    std_prod <- (std$unsqueeze(1) * std$unsqueeze(2))$clamp(min = eps)
    numer <- vlr - log_var$expand(c(d, d)) - log_var$view(c(-1L, 1L))$
        expand(c(d, d))
    rho <- numer / std_prod$mul(-2)
    rho <- rho$nan_to_num(nan = 0, posinf = 0, neginf = 0)
    rho <- (rho + rho$t())$mul(0.5)$clamp(min = -1, max = 1)
    rho$fill_diagonal_(1)
    torch::torch_where(rho$abs()$ge(threshold), rho, .zero_t())
}

# float64 scalar tensors.
.nan_t <- function() {
    torch::torch_tensor(NaN, dtype = torch::torch_float64())
}
.zero_t <- function() {
    torch::torch_tensor(0, dtype = torch::torch_float64())
}
