# Fitting, correlation estimation, and reproducibility (LibTorch required).

small_fit <- function(seed = 0L, epochs = 30L) {
    sim <- metvae_sim_data(n = 40, d = 10, cor_pairs = 2, mu = 10:12,
                           seed = 1)
    y <- sim$y
    y[sweep(y, 2, apply(y, 2, stats::quantile, 0.2), "<")] <- 0
    metvae(y, batch_size = 16L, max_epochs = epochs, learning_rate = 1e-2,
           seed = seed, verbose = FALSE)
}

test_that("a fit holds only R objects and survives saveRDS", {
    skip_if_no_libtorch()
    fit <- small_fit()
    expect_s3_class(fit, "metvae")
    expect_length(fit$train_loss, 30L)
    expect_true(all(is.finite(fit$train_loss)))
    is_tensor <- vapply(unlist(fit, recursive = TRUE),
                        function(v) inherits(v, "torch_tensor"), logical(1L))
    expect_false(any(is_tensor))
    path <- withr::local_tempfile(fileext = ".rds")
    saveRDS(fit, path)
    a <- get_corr(fit, num_sim = 3L)
    b <- get_corr(readRDS(path), num_sim = 3L)
    expect_identical(a$corr, b$corr)
})

test_that("results are reproducible within a session", {
    skip_if_no_libtorch()
    threads <- torch::torch_get_num_threads()
    a <- get_corr(small_fit(), num_sim = 4L)
    expect_identical(torch::torch_get_num_threads(), threads)
    b <- get_corr(small_fit(), num_sim = 4L)
    expect_identical(a$train_loss, b$train_loss)
    expect_identical(a$model$state, b$model$state)
    expect_identical(a$corr, b$corr)
    c <- get_corr(small_fit(seed = 1L), num_sim = 4L)
    expect_false(identical(a$train_loss, c$train_loss))
})

test_that("get_corr and impute_zeros keep the observed entries", {
    skip_if_no_libtorch()
    fit <- get_corr(small_fit(), num_sim = 3L, seed = 2L)
    est <- as.matrix(fit$corr$estimate)
    expect_true(isSymmetric(est))
    expect_true(all(diag(est) == 1))
    imp <- impute_zeros(fit, seed = 1L)
    observed <- !is.na(fit$clr_data)
    expect_identical(imp[observed], fit$clr_data[observed])
    expect_false(anyNA(imp))
    expect_identical(imp, impute_zeros(fit, seed = 1L))
    expect_error(get_corr(fit, num_sim = 0), "num_sim")
})

test_that("sparse_by_sec accepts a fixed penalty", {
    skip_if_no_libtorch()
    fit <- get_corr(small_fit(), num_sim = 3L)
    res <- sparse_by_sec(fit, rho = 0.5)
    expect_identical(res$best_rho, 0.5)
    expect_null(res$scores_by_rho)
    sp <- as.matrix(res$sparse_estimate)
    expect_true(isSymmetric(sp))
    expect_identical(dimnames(sp), list(fit$feature_names, fit$feature_names))
})

test_that("a SummarizedExperiment and a matrix give the same fit", {
    skip_if_no_libtorch()
    sim <- metvae_sim_data(n = 30, d = 8, cor_pairs = 2, mu = 10:12,
                           seed = 1)
    meta <- data.frame(batch = rep(c("A", "B"), 15),
                       row.names = rownames(sim$y))
    se <- SummarizedExperiment::SummarizedExperiment(
        assays = list(abundance = t(sim$y)), colData = meta)
    a <- metvae(se, categorical_covariate_keys = "batch", max_epochs = 5L,
                verbose = FALSE)
    b <- metvae(sim$y, meta = meta, categorical_covariate_keys = "batch",
                max_epochs = 5L, verbose = FALSE)
    expect_identical(a$train_loss, b$train_loss)
    expect_identical(a$clr_coef, b$clr_coef)
})
