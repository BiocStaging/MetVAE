# Initial values for the censored entries on the centered log-ratio scale.

# Replaces each NaN of y (batch x d) by a draw from the lower tail of a
# normal: per feature, 'sample_size' normal variates with the feature mean
# and standard deviation are drawn, the k smallest are kept (k = number of
# zeros of the feature, at least 1), and the t-th NaN of the feature in row
# order receives the t-th uniform choice among them.
.random_initial <- function(y, sample_size, num_zero, mean, sd, gen = NULL) {
    nan_mask <- torch::torch_isnan(y)
    num_nan <- nan_mask$sum(dim = 1L)
    if (.item(num_nan$eq(0L)$all())) {
        return(y$clone())
    }
    d <- y$size(2)
    rand <- .rnorm_t(c(sample_size, d), y$device, gen)
    random_data <- rand$mul_(sd)$add_(mean)
    k0 <- num_zero$to(dtype = torch::torch_long())
    k <- torch::torch_where(k0$gt(0L), k0, torch::torch_ones_like(k0))$
        clamp(min = 1L, max = sample_size)
    k_max <- .item(k$max())
    candidates <- torch::torch_topk(random_data, k_max, dim = 1L,
                                    largest = FALSE, sorted = TRUE)[[1L]]
    m_max <- .item(num_nan$max())
    u <- .runif_f32(c(m_max, d), y$device, gen)
    r <- (u * k$view(c(1L, -1L)))$floor()$to(dtype = torch::torch_long())
    fills <- candidates$gather(1L, r$add(1L))
    rank <- nan_mask$cumsum(dim = 1L)$sub(1L)$clamp(min = 0L, max = m_max - 1L)
    picked <- fills$gather(1L, rank$add(1L))
    torch::torch_where(nan_mask, picked, y)
}
