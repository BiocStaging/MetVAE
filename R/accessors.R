#' Covariate coefficients and effects
#'
#' @description
#' \code{confound_coef} returns the estimated covariate coefficients on the
#' centered log-ratio (CLR) scale, and \code{confound_es} returns the
#' covariate effect of every sample and feature, i.e. the scaled covariate
#' matrix times the coefficients. Continuous covariates are centered and
#' scaled, so a coefficient is the change in CLR abundance per standard
#' deviation of the covariate.
#'
#' @param fit An object returned by \code{\link{metvae}}.
#'
#' @return \code{confound_coef}: a features x covariates matrix, or NULL when
#' the model has no covariates. \code{confound_es}: a samples x features
#' matrix, or NULL.
#'
#' @seealso \code{\link{metvae}}
#'
#' @examples
#' fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
#'                             package = "MetVAE"))
#' head(confound_coef(fits$example))
#' confound_es(fits$example)[1:3, 1:4]
#'
#' @name confound
NULL

#' @rdname confound
#' @export
confound_coef <- function(fit) {
    .check_fit(fit)
    if (is.null(fit$clr_coef)) {
        return(NULL)
    }
    t(fit$clr_coef)
}

#' @rdname confound
#' @export
confound_es <- function(fit) {
    .check_fit(fit)
    if (is.null(fit$clr_coef)) {
        return(NULL)
    }
    out <- fit$design %*% fit$clr_coef
    dimnames(out) <- list(fit$sample_names, fit$feature_names)
    out
}

#' Decoder loadings and co-occurrence
#'
#' @description
#' \code{clr_loading} returns the weights of the linear decoder, which give
#' the loading of every feature on every latent variable on the centered
#' log-ratio scale. \code{cooccurrence} returns the squared Euclidean distance
#' between the loading vectors of every pair of features, multiplied by
#' (n - 1) / n for n samples; small values indicate features that vary
#' together.
#'
#' @param fit An object returned by \code{\link{metvae}}.
#'
#' @return \code{clr_loading}: a features x latent variables matrix with
#' columns latent_0, latent_1, .... \code{cooccurrence}: a features x features
#' matrix.
#'
#' @seealso \code{\link{metvae}}
#'
#' @examples
#' fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
#'                             package = "MetVAE"))
#' dim(clr_loading(fits$example))
#' cooccurrence(fits$example)[1:3, 1:3]
#'
#' @name loadings
NULL

#' @rdname loadings
#' @export
clr_loading <- function(fit) {
    .check_fit(fit)
    w <- fit$model$state[["decode_mu.weight"]]
    dimnames(w) <- list(fit$feature_names,
                        paste0("latent_", seq_len(ncol(w)) - 1L))
    w
}

#' @rdname loadings
#' @export
cooccurrence <- function(fit) {
    n <- length(fit$sample_names)
    out <- as.matrix(stats::dist(clr_loading(fit)))^2 * (n - 1) / n
    dimnames(out) <- list(fit$feature_names, fit$feature_names)
    out
}
