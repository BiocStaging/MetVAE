# Centered log-ratio transform, censored-normal estimation of the feature
# means, standard deviations and covariate coefficients, and removal of the
# covariate effects.

# Tensors used by the model from the output of .metvae_input().
.clr_preprocess <- function(input) {
    f64 <- torch::torch_float64()
    x <- .as_t(input$data)
    design <- if (is.null(input$design)) NULL else .as_t(input$design)
    nan <- torch::torch_tensor(NaN, dtype = f64)
    log_data <- torch::torch_where(x$gt(0), x$log(), nan)
    shift <- log_data$nanmean(dim = 2L, keepdim = TRUE)
    clr <- log_data - shift
    th_raw <- torch::torch_where(x$gt(0), x,
                                 torch::torch_tensor(Inf, dtype = f64))$
        min(dim = 1L)[[1L]]
    th_raw <- torch::torch_where(th_raw$eq(Inf),
                                 torch::torch_tensor(1e-5, dtype = f64),
                                 th_raw)
    clr_th <- th_raw$log() - shift
    num_zero <- x$eq(0)$sum(dim = 1L)$to(dtype = torch::torch_float32())
    params <- .ols_estimate(clr, design)
    if (.item(num_zero$ne(0)$any())) {
        params <- .fit_censored_normal(clr, design, clr_th, params,
                                       max_iter = 100L)
    }
    p <- if (is.null(design)) 1L else design$size(2) + 1L
    clr_coef <- NULL
    if (!is.null(design)) {
        clr_coef <- params[2:p, , drop = FALSE]
        clr <- clr - design$matmul(clr_coef)
    }
    list(clr_data = clr, design = design, num_zero = num_zero, shift = shift,
         clr_mean = params[1L, ], clr_sd = params[p + 1L, ]$exp(),
         clr_coef = clr_coef)
}
