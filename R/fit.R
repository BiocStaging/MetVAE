# The 'metvae' fit object: a list of R objects (no torch tensors), so that it
# can be saved with saveRDS().

# Fit object from the input record, the preprocessed tensors, the trained
# module, and the settings.
.new_fit <- function(input, pp, m, settings, losses, call) {
    state <- lapply(.vae_params(m), .as_r)
    structure(list(
        call = call,
        versions = list(MetVAE = as.character(utils::packageVersion("MetVAE")),
                        torch = as.character(utils::packageVersion("torch")),
                        R = R.version.string),
        sample_names = input$sample_names,
        feature_names = input$feature_names,
        confound_names = input$confound_names,
        preprocess = input$preprocess,
        design = input$design,
        clr_data = .named(.as_r(pp$clr_data), input),
        shift = stats::setNames(as.numeric(.as_r(pp$shift)),
                                input$sample_names),
        num_zero = stats::setNames(as.numeric(.as_r(pp$num_zero)),
                                   input$feature_names),
        clr_mean = stats::setNames(as.numeric(.as_r(pp$clr_mean)),
                                   input$feature_names),
        clr_sd = stats::setNames(as.numeric(.as_r(pp$clr_sd)),
                                 input$feature_names),
        clr_coef = .coef_matrix(pp$clr_coef, input),
        model = list(config = settings$model, state = state),
        train_settings = settings$train,
        train_loss = losses,
        seed = settings$seed,
        corr = NULL),
        class = "metvae")
}

# Samples x features matrix with names.
.named <- function(x, input) {
    dimnames(x) <- list(input$sample_names, input$feature_names)
    x
}

# Covariate coefficients (covariates x features) with names, or NULL.
.coef_matrix <- function(coef, input) {
    if (is.null(coef)) {
        return(NULL)
    }
    out <- .as_r(coef)
    dimnames(out) <- list(input$confound_names, input$feature_names)
    out
}

# Stops unless 'fit' is a metvae fit.
.check_fit <- function(fit) {
    if (!inherits(fit, "metvae")) {
        stop("'fit' must be an object returned by metvae().", call. = FALSE)
    }
    invisible(fit)
}

# Stops unless correlations have been estimated.
.check_corr <- function(fit) {
    if (is.null(fit$corr)) {
        stop("No correlation estimates. Run fit <- get_corr(fit) first.",
             call. = FALSE)
    }
    invisible(fit)
}

# Torch tensors of the preprocessed data of a fit.
.fit_tensors <- function(fit) {
    list(clr_data = .as_t(fit$clr_data),
         shift = .as_t(fit$shift)$view(c(-1L, 1L)),
         num_zero = .as_t(fit$num_zero, dtype = torch::torch_float32()),
         clr_mean = .as_t(fit$clr_mean),
         clr_sd = .as_t(fit$clr_sd))
}

# Autoencoder module of a fit, rebuilt from the stored weights. The global
# torch generator state is left unchanged.
.fit_module <- function(fit) {
    cfg <- fit$model$config
    rng <- torch::torch_get_rng_state()
    on.exit(torch::torch_set_rng_state(rng), add = TRUE)
    m <- .vae_module(cfg$input_dim, cfg$latent_dim, cfg$hidden_dims,
                     cfg$activation)
    params <- .vae_params(m)
    torch::with_no_grad({
        for (k in names(params)) {
            params[[k]]$copy_(.as_t(fit$model$state[[k]]))
        }
    })
    m$eval()
    m
}

# Sparse symmetric matrix (dgCMatrix) with feature names from a dense tensor
# or matrix.
.as_sparse <- function(x, names) {
    if (inherits(x, "torch_tensor")) {
        x <- .as_r(x)
    }
    dimnames(x) <- list(names, names)
    methods::as(Matrix::Matrix(x, sparse = TRUE), "generalMatrix")
}
