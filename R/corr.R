# Correlation of log abundances derived from the variance of log ratios,
# with hard thresholding.

# Correlation matrix (d x d) of an n x d abundance tensor. When every log
# value is finite the matrix is formed from one Gram matrix in the buffers
# 'out' and 'scratch'; otherwise non-finite log values are treated as
# missing.
.corr_dense <- function(data, threshold = 0.2, eps = 1e-12, out = NULL,
                        scratch = NULL) {
    n <- data$size(1)
    d <- data$size(2)
    if (d < 3L) {
        stop("Need at least d >= 3 features to compute stable correlations.",
             call. = FALSE)
    }
    log_x <- data$log()
    if (!.item(torch::torch_isfinite(log_x)$all())) {
        rho <- .corr_general(data, threshold, eps)
        if (is.null(out)) {
            return(rho)
        }
        return(out$copy_(rho))
    }
    f64 <- torch::torch_float64()
    G <- if (is.null(out)) torch::torch_empty(d, d, dtype = f64) else out
    Tm <- if (is.null(scratch)) torch::torch_empty(d, d, dtype = f64) else
        scratch
    Yc <- log_x - log_x$nanmean(dim = 1L, keepdim = TRUE)
    q <- Yc$square()$sum(dim = 1L)
    G$addmm_(Yc$t(), Yc, beta = 0, alpha = 1)
    G$mul_(-2)$add_(q$view(c(-1L, 1L)))$add_(q$view(c(1L, -1L)))$div_(n)
    G$clamp_(min = 0)
    G$fill_diagonal_(0)
    log_var <- .normalized_log_var(log_x, eps)
    std <- log_var$sqrt()
    G$sub_(log_var$view(c(1L, -1L)))$sub_(log_var$view(c(-1L, 1L)))
    Tm$copy_(std$view(c(1L, -1L)))$mul_(std$view(c(-1L, 1L)))
    Tm$clamp_(min = eps)$mul_(-2)
    G$div_(Tm)
    .corr_finish(G, threshold)
}

# Per-feature variance of the sample-centered log data, normalized as in the
# proportionality-based correlation.
.normalized_log_var <- function(log_x, eps) {
    n <- log_x$size(1)
    d <- log_x$size(2)
    clr <- log_x - log_x$nanmean(dim = 2L, keepdim = TRUE)
    clr_mean <- clr$sum(dim = 1L, keepdim = TRUE) / n
    clr_var <- (clr - clr_mean)$pow(2)$sum(dim = 1L) / n
    .normalize_var(clr_var, d, eps)
}

# Normalization of the feature variances.
.normalize_var <- function(clr_var, d, eps) {
    sum_log_var <- clr_var$nansum() * d / (d - 1)
    log_var <- (clr_var - sum_log_var * (1 / d^2)) * d / (d - 2)
    log_var$nan_to_num(nan = 0, posinf = 0, neginf = 0)$clamp(min = eps)
}

# Cleanup and hard threshold of a correlation matrix, in place.
.corr_finish <- function(G, threshold) {
    G$nan_to_num_(nan = 0, posinf = 0, neginf = 0)
    G$clamp_(min = -1, max = 1)
    G$fill_diagonal_(1)
    G$masked_fill_(G$gt(-threshold)$logical_and(G$lt(threshold)), 0)
    G
}
