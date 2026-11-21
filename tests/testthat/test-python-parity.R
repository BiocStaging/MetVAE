# Agreement with the Python package metvae on its quickstart workload. The
# reference outputs are written by inst/scripts/make_pyref.py; a subset is
# shipped in tests/testthat/fixtures/python, and the stage outputs are used
# when METVAE_PYREF_DIR points to a full set.

quickstart_fit <- function() {
    inp <- pyref_inputs()
    fit <- metvae(inp$data, meta = inp$meta,
                  continuous_covariate_keys = "age",
                  categorical_covariate_keys = "batch", learning_rate = 1e-2,
                  verbose = FALSE)
    get_corr(fit)
}

edge_metrics <- function(est, truth) {
    lt <- lower.tri(truth)
    e <- as.matrix(est)[lt] != 0
    t <- truth[lt] != 0
    c(TPR = sum(e & t) / sum(t), FPR = sum(e & !t) / sum(!t),
      FDR = sum(e & !t) / max(sum(e), 1))
}

test_that("the quickstart reproduces the Python results", {
    skip_if_no_libtorch()
    skip_if_no_pyref(c("data", "train_loss", "estimate", "p_sparse",
                       "sec_cv_sparse", "sec_cv_best_rho", "true_cor"))
    fit <- quickstart_fit()
    loss_ref <- read_pyref("train_loss")
    expect_lt(max(abs(fit$train_loss - loss_ref) / abs(loss_ref)), 1e-6)
    est <- as.matrix(fit$corr$estimate)
    ref <- read_pyref("estimate")
    expect_lt(max_abs_diff(est, ref), 1e-6)
    expect_identical(est != 0, ref != 0, ignore_attr = TRUE)
    truth <- read_pyref("true_cor")
    rp <- sparse_by_p(fit)
    expect_equal(edge_metrics(rp$sparse_estimate, truth),
                 edge_metrics(read_pyref("p_sparse"), truth))
    rs <- sparse_by_sec(fit)
    expect_identical(rs$best_rho, read_pyref("sec_cv_best_rho"))
    expect_equal(edge_metrics(rs$sparse_estimate, truth),
                 edge_metrics(read_pyref("sec_cv_sparse"), truth))
})

test_that("each stage reproduces Python from the Python inputs", {
    skip_if_no_libtorch()
    stages <- c("clr_raw", "design", "clr_th", "ols", "clr_params",
                "random_initial_123", "clr_data", "clr_mean", "clr_sd",
                "num_zero", "impute_log_sim0", "corr_sim0",
                "final.decode_rho")
    skip_if_no_pyref(stages)
    t64 <- function(name) MetVAE:::.as_t(read_pyref(name))
    expect_lt(max_abs_diff(MetVAE:::.as_r(MetVAE:::.ols_estimate(
        t64("clr_raw"), t64("design"))), read_pyref("ols")), 1e-12)
    fit <- MetVAE:::.fit_censored_normal(t64("clr_raw"), t64("design"),
                                         t64("clr_th"), t64("ols"))
    expect_lt(max_abs_diff(MetVAE:::.as_r(fit), read_pyref("clr_params")),
              1e-9)
    nz <- MetVAE:::.as_t(read_pyref("num_zero"),
                         dtype = torch::torch_float32())
    ri <- MetVAE:::.random_initial(t64("clr_data"), 100L, nz, t64("clr_mean"),
                                   t64("clr_sd"), gen = MetVAE:::.new_gen(123L))
    expect_identical(MetVAE:::.as_r(ri), read_pyref("random_initial_123"))
    corr <- MetVAE:::.corr_dense(t64("impute_log_sim0")$exp())
    expect_lt(max_abs_diff(MetVAE:::.as_r(corr), read_pyref("corr_sim0")),
              1e-13)
})
