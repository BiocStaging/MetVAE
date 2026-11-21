# Imputation of the censored entries and the multiple-imputation estimate of
# the correlation matrix.

# Completed CLR data (n x d tensor): random initial values for the censored
# entries, then their reconstruction by the autoencoder. 'gen' is a torch
# generator or NULL for the global generator.
.impute_zeros_t <- function(m, pp, gen = NULL) {
    y <- pp$clr_data
    if (!.item(pp$num_zero$ne(0)$any())) {
        return(y)
    }
    was_training <- m$training
    m$eval()
    on.exit(if (was_training) m$train(), add = TRUE)
    nan_mask <- torch::torch_isnan(y)
    complete_y <- .random_initial(y, y$size(1), pp$num_zero, pp$clr_mean,
                                  pp$clr_sd, gen = gen)
    recon <- torch::with_no_grad(.vae_forward(m, complete_y, gen)$recon)
    torch::torch_where(nan_mask, recon, y)
}

# Mean over 'num_sim' imputations of the imputed log data and of the
# thresholded correlation matrix; imputation s uses a generator seeded with
# base_seed + s - 1.
.multiple_imputation <- function(m, pp, num_sim, threshold, base_seed) {
    d <- pp$clr_data$size(2)
    if (!.item(pp$num_zero$ne(0)$any())) {
        il <- pp$clr_data + pp$shift
        return(list(impute_log = il,
                    estimate = .corr_dense(il$exp(), threshold)))
    }
    f64 <- torch::torch_float64()
    acc <- torch::torch_zeros(d, d, dtype = f64)
    G <- torch::torch_empty(d, d, dtype = f64)
    Tm <- torch::torch_empty(d, d, dtype = f64)
    ilog <- NULL
    torch::with_no_grad({
        for (s in seq_len(num_sim)) {
            gen <- .new_gen(base_seed + s - 1L)
            il <- .impute_zeros_t(m, pp, gen) + pp$shift
            .corr_dense(il$exp(), threshold, out = G, scratch = Tm)
            acc$add_(G)
            ilog <- if (is.null(ilog)) il else ilog$add_(il)
        }
    })
    list(impute_log = ilog / num_sim, estimate = acc / num_sim)
}
