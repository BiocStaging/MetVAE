#' Print and summarize a MetVAE fit
#'
#' @description
#' \code{print} reports the data dimensions before and after filtering, the
#' proportion of zeros, the covariates, the model size, the training, and
#' whether correlations have been estimated. \code{summary} returns these
#' quantities as a list.
#'
#' @param x,object An object returned by \code{\link{metvae}}.
#' @param ... Unused.
#'
#' @return \code{print} returns \code{x} invisibly. \code{summary} returns an
#' object of class \code{"summary.metvae"}, a list with the components
#' n_samples, n_features, n_input_samples, n_input_features, zero_fraction,
#' covariates, latent_dim, hidden_dims, epochs, final_loss, and
#' n_edges (number of nonzero off-diagonal pairs of the correlation estimate,
#' or NA).
#'
#' @examples
#' fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
#'                             package = "MetVAE"))
#' fits$example
#' summary(fits$example)$n_edges
#'
#' @name metvae-methods
NULL

#' @rdname metvae-methods
#' @export
summary.metvae <- function(object, ...) {
    .check_fit(object)
    pre <- object$preprocess
    n_edges <- NA_integer_
    if (!is.null(object$corr)) {
        est <- object$corr$estimate
        n_edges <- as.integer((Matrix::nnzero(est) -
                                   sum(Matrix::diag(est) != 0)) / 2)
    }
    structure(list(
        n_samples = pre$n_kept, n_features = pre$d_kept,
        n_input_samples = pre$n_input, n_input_features = pre$d_input,
        zero_fraction = sum(object$num_zero) / (pre$n_kept * pre$d_kept),
        covariates = object$confound_names,
        latent_dim = object$model$config$latent_dim,
        hidden_dims = object$model$config$hidden_dims,
        epochs = length(object$train_loss),
        final_loss = object$train_loss[length(object$train_loss)],
        n_edges = n_edges), class = "summary.metvae")
}

#' @rdname metvae-methods
#' @export
print.summary.metvae <- function(x, ...) {
    cat("MetVAE fit\n")
    cat("  samples:  ", x$n_samples, " (of ", x$n_input_samples, ")\n",
        sep = "")
    cat("  features: ", x$n_features, " (of ", x$n_input_features, ")\n",
        sep = "")
    cat("  zeros imputed: ", sprintf("%.1f%%", 100 * x$zero_fraction), "\n",
        sep = "")
    cat("  covariates: ", if (length(x$covariates)) {
        paste(x$covariates, collapse = ", ")
    } else {
        "none"
    }, "\n", sep = "")
    cat("  latent dimension: ", x$latent_dim, "; encoder: ",
        if (length(x$hidden_dims)) {
            paste("hidden layers", paste(x$hidden_dims, collapse = ", "))
        } else {
            "linear"
        }, "\n", sep = "")
    cat("  training: ", x$epochs, " epochs, final loss ",
        format(x$final_loss, digits = 6), "\n", sep = "")
    if (is.na(x$n_edges)) {
        cat("  correlations: not estimated; run fit <- get_corr(fit)\n")
    } else {
        cat("  correlations: ", x$n_edges, " nonzero pairs\n", sep = "")
    }
    invisible(x)
}

#' @rdname metvae-methods
#' @export
print.metvae <- function(x, ...) {
    print(summary(x))
    invisible(x)
}
