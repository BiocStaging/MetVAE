#' Estimate the correlation matrix by multiple imputation
#'
#' @description
#' Imputes the zeros \code{num_sim} times with the trained autoencoder,
#' computes the correlation matrix of each completed data set from the
#' variance of the log ratios, sets entries with absolute value below
#' \code{threshold} to zero, and averages the matrices over the imputations.
#'
#' @param fit An object returned by \code{\link{metvae}}.
#' @param num_sim Number of imputations. Default 100.
#' @param threshold Entries of each imputation's correlation matrix with
#' absolute value below this value are set to zero before averaging. Default
#' 0.2.
#' @param seed Integer seed; imputation s uses the seed \code{seed + s - 1}.
#' NULL (default) uses the seed of \code{fit}.
#'
#' @return \code{fit} with the component \code{corr}: a list with
#' \code{impute_log_data} (samples x features matrix, mean of the imputed log
#' abundances), \code{estimate} (features x features sparse matrix of class
#' \code{dgCMatrix}, mean correlation), and the settings \code{num_sim},
#' \code{threshold}, and \code{seed}. Assign the result:
#' \code{fit <- get_corr(fit)}.
#'
#' @details
#' Results are identical for a given seed on one machine with one version of
#' the package torch. The number of threads used by torch, set with
#' \code{torch::torch_set_num_threads()}, can change the last digits.
#'
#' @seealso \code{\link{metvae}}, \code{\link{sparse_by_p}},
#' \code{\link{sparse_by_sec}}
#'
#' @examples
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     torch::torch_is_installed()) {
#'     sim <- metvae_sim_data(n = 40, d = 10, cor_pairs = 2, mu = 10:12,
#'                            seed = 1)
#'     y <- sim$y
#'     y[y < apply(y, 2, stats::quantile, 0.2)[col(y)]] <- 0
#'     fit <- metvae(y, max_epochs = 20, verbose = FALSE)
#'     fit <- get_corr(fit, num_sim = 5)
#'     fit$corr$estimate[1:4, 1:4]
#' }
#'
#' @export
get_corr <- function(fit, num_sim = 100L, threshold = 0.2, seed = NULL) {
    .check_fit(fit)
    num_sim <- .check_count(num_sim, "num_sim")
    threshold <- .check_number(threshold, "threshold", lower = 0, upper = 1)
    seed <- if (is.null(seed)) fit$seed else .check_count(seed, "seed", 0L)
    .metvae_require_torch("get_corr()")
    m <- .fit_module(fit)
    pp <- .fit_tensors(fit)
    res <- .multiple_imputation(m, pp, num_sim, threshold, seed)
    impute_log <- .as_r(res$impute_log)
    dimnames(impute_log) <- list(fit$sample_names, fit$feature_names)
    fit$corr <- list(impute_log_data = impute_log,
                     estimate = .as_sparse(res$estimate, fit$feature_names),
                     num_sim = num_sim, threshold = threshold, seed = seed)
    fit
}

#' Impute the zeros
#'
#' @description
#' Completes the centered log-ratio (CLR) data of a fit once: the zeros are
#' given random initial values from the lower tail of the estimated feature
#' distributions and then replaced by their reconstruction by the trained
#' autoencoder. Observed entries are unchanged.
#'
#' @param fit An object returned by \code{\link{metvae}}.
#' @param seed Integer seed of the random draws. NULL (default) uses the
#' global torch random number generator.
#'
#' @return A samples x features matrix of completed CLR values, after removal
#' of the covariate effects.
#'
#' @seealso \code{\link{metvae}}, \code{\link{get_corr}}
#'
#' @examples
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     torch::torch_is_installed()) {
#'     sim <- metvae_sim_data(n = 40, d = 10, cor_pairs = 2, mu = 10:12,
#'                            seed = 1)
#'     y <- sim$y
#'     y[y < apply(y, 2, stats::quantile, 0.2)[col(y)]] <- 0
#'     fit <- metvae(y, max_epochs = 20, verbose = FALSE)
#'     impute_zeros(fit, seed = 1)[1:3, 1:4]
#' }
#'
#' @export
impute_zeros <- function(fit, seed = NULL) {
    .check_fit(fit)
    .metvae_require_torch("impute_zeros()")
    gen <- if (is.null(seed)) NULL else .new_gen(.check_count(seed, "seed", 0L))
    m <- .fit_module(fit)
    out <- .as_r(.impute_zeros_t(m, .fit_tensors(fit), gen))
    dimnames(out) <- list(fit$sample_names, fit$feature_names)
    out
}
