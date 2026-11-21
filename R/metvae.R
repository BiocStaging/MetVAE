#' Fit the MetVAE model
#'
#' @description
#' Preprocesses an abundance table and trains the variational autoencoder of
#' MetVAE. Features whose proportion of zeros exceeds
#' \code{feature_zero_threshold} are removed, the data are transformed to the
#' centered log-ratio (CLR) scale, the feature means, standard deviations, and
#' covariate coefficients are estimated by a left-censored normal model in
#' which zeros are values below the detection limit, the covariate effects are
#' removed, and the autoencoder is trained on the result. The correlations are
#' estimated afterwards by \code{\link{get_corr}}.
#'
#' @param x A \code{SummarizedExperiment} with features in rows and samples in
#' columns, or a numeric matrix or data.frame with samples in rows and
#' features in columns (see \code{features_as_rows}). Values are non-negative
#' abundances; zeros mark values below the detection limit and missing values
#' are treated as zeros.
#' @param meta A data.frame of sample metadata with the sample names as row
#' names, or NULL. Every sample of \code{x} must be present. Must be NULL when
#' \code{x} is a \code{SummarizedExperiment}, whose \code{colData} is used.
#' @param assay_name Name of the assay of a \code{SummarizedExperiment}; NULL
#' uses the first assay.
#' @param features_as_rows Logical; TRUE when the rows of a matrix or
#' data.frame \code{x} are features. Default FALSE.
#' @param continuous_covariate_keys Character vector of numeric metadata
#' columns to adjust for, or NULL. They are centered and scaled.
#' @param categorical_covariate_keys Character vector of categorical metadata
#' columns to adjust for, or NULL. Each is coded by indicator variables with
#' the first level as reference (factor levels as given, other values sorted).
#' @param feature_zero_threshold Features with a proportion of zeros greater
#' than this value are removed. NULL keeps every feature. Default 0.3.
#' @param sample_zero_threshold Samples with a proportion of zeros greater than
#' this value are removed after the feature filter. NULL (default) keeps every
#' sample.
#' @param latent_dim Number of latent variables. NULL (default) uses the
#' smaller of the numbers of samples and features.
#' @param hidden_dims Integer vector of hidden layer sizes of the encoder, or
#' NULL (default) for a linear encoder.
#' @param activation Activation between hidden layers: "relu" (default),
#' "tanh", "gelu", "silu", or "none".
#' @param batch_size Number of samples per minibatch. Default 128.
#' @param max_epochs Number of training epochs. Default 1000.
#' @param learning_rate Initial learning rate of the AdamW optimizer; a cosine
#' schedule with warm restarts varies it between this value and half of it.
#' Default 1e-3.
#' @param max_grad_norm Maximum 2-norm of the gradient, or NULL for no
#' clipping. Default 1.
#' @param shuffle Logical; TRUE (default) permutes the samples every epoch.
#' @param seed Integer seed of the model initialization, the batch order, and
#' the default seed of \code{\link{get_corr}}. Default 0.
#' @param verbose Logical; TRUE (default) reports the preprocessing steps and
#' shows a progress bar in interactive sessions.
#'
#' @return An object of class \code{"metvae"}: a list holding the sample,
#' feature, and covariate names, the preprocessed CLR data (\code{clr_data},
#' samples x features, NA at the zeros), the per-sample offsets
#' (\code{shift}), the zero counts, the estimated CLR means
#' (\code{clr_mean}), standard deviations (\code{clr_sd}) and covariate
#' coefficients (\code{clr_coef}, covariates x features, or NULL), the model
#' weights, and the training loss of each epoch (\code{train_loss}). The
#' object contains only R objects and can be saved with \code{saveRDS()}.
#'
#' @seealso \code{\link{get_corr}}, \code{\link{sparse_by_p}},
#' \code{\link{sparse_by_sec}}
#'
#' @examples
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     torch::torch_is_installed()) {
#'     sim <- metvae_sim_data(n = 40, d = 10, cor_pairs = 2, mu = 10:12,
#'                            seed = 1)
#'     fit <- metvae(sim$y, max_epochs = 20, verbose = FALSE)
#'     fit
#' }
#'
#' @export
metvae <- function(x, meta = NULL, assay_name = NULL,
                   features_as_rows = FALSE,
                   continuous_covariate_keys = NULL,
                   categorical_covariate_keys = NULL,
                   feature_zero_threshold = 0.3, sample_zero_threshold = NULL,
                   latent_dim = NULL, hidden_dims = NULL,
                   activation = c("relu", "tanh", "gelu", "silu", "none"),
                   batch_size = 128L, max_epochs = 1000L,
                   learning_rate = 1e-3, max_grad_norm = 1, shuffle = TRUE,
                   seed = 0L, verbose = TRUE) {
    call <- match.call()
    settings <- .metvae_settings(latent_dim, hidden_dims, match.arg(activation),
                                 batch_size, max_epochs, learning_rate,
                                 max_grad_norm, shuffle, seed)
    input <- .metvae_input(x, meta, assay_name, features_as_rows,
                           continuous_covariate_keys,
                           categorical_covariate_keys, feature_zero_threshold,
                           sample_zero_threshold, verbose)
    .metvae_require_torch("metvae()")
    n <- nrow(input$data)
    d <- ncol(input$data)
    settings$model <- list(input_dim = d,
                           latent_dim = if (is.null(latent_dim)) min(n, d) else
                               settings$model$latent_dim,
                           hidden_dims = settings$model$hidden_dims,
                           activation = settings$model$activation)
    torch::torch_manual_seed(settings$seed)
    pp <- .clr_preprocess(input)
    m <- .vae_module(d, settings$model$latent_dim, settings$model$hidden_dims,
                     settings$model$activation)
    losses <- .train_vae(m, pp, settings$train, settings$seed, verbose)
    .new_fit(input, pp, m, settings, losses, call)
}

# Validated model and training settings.
.metvae_settings <- function(latent_dim, hidden_dims, activation, batch_size,
                             max_epochs, learning_rate, max_grad_norm,
                             shuffle, seed) {
    if (!is.null(latent_dim)) {
        latent_dim <- .check_count(latent_dim, "latent_dim")
    }
    if (!is.null(hidden_dims)) {
        if (!is.numeric(hidden_dims) || anyNA(hidden_dims) ||
            any(hidden_dims < 1) || any(hidden_dims != round(hidden_dims))) {
            stop("'hidden_dims' must be NULL or a vector of positive whole ",
                 "numbers.", call. = FALSE)
        }
        hidden_dims <- as.integer(hidden_dims)
    }
    if (!is.null(max_grad_norm)) {
        max_grad_norm <- .check_number(max_grad_norm, "max_grad_norm",
                                       lower = .Machine$double.xmin)
    }
    .check_flag(shuffle, "shuffle")
    list(model = list(latent_dim = latent_dim, hidden_dims = hidden_dims,
                      activation = activation),
         train = list(batch_size = .check_count(batch_size, "batch_size"),
                      max_epochs = .check_count(max_epochs, "max_epochs"),
                      learning_rate = .check_number(
                          learning_rate, "learning_rate",
                          lower = .Machine$double.xmin),
                      max_grad_norm = max_grad_norm, shuffle = shuffle),
         seed = .check_count(seed, "seed", lower = 0L))
}
