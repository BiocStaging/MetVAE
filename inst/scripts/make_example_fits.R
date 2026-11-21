# Builds inst/extdata/metvae_example_fits.rds, the precomputed fits used by
# the help page examples and by the vignette when LibTorch is not installed.
# The code of each object repeats the corresponding vignette chunks.
#
# Run from the package root with LibTorch installed:
#   Rscript inst/scripts/make_example_fits.R

library(MetVAE)

censor <- function(y, prop) {
    th <- apply(y, 2L, stats::quantile, probs = prop)
    y[sweep(y, 2L, th, "<")] <- 0
    y
}

# Small fit for the help page examples.
set.seed(1)
meta_ex <- data.frame(age = stats::rnorm(60L, 50, 8),
                      batch = rep(c("A", "B"), 30L),
                      row.names = paste0("s", 0:59))
sim_ex <- metvae_sim_data(n = 60L, d = 15L, cor_pairs = 3L, mu = 10:14,
                          x = meta_ex, cont_list = "age",
                          cat_list = "batch", seed = 1L)
fit_ex <- metvae(censor(sim_ex$y, 0.3), meta = meta_ex,
                 continuous_covariate_keys = "age",
                 categorical_covariate_keys = "batch",
                 learning_rate = 1e-2, max_epochs = 300L, verbose = FALSE)
fit_ex <- get_corr(fit_ex, num_sim = 20L)

# Vignette: quickstart.
set.seed(7)
meta <- data.frame(age = stats::rnorm(100L, mean = 50, sd = 8),
                   batch = sample(c("A", "B"), 100L, replace = TRUE),
                   row.names = paste0("s", 0:99))
sim <- metvae_sim_data(n = 100L, d = 50L, cor_pairs = 10L, mu = 10:14,
                       x = meta, cont_list = "age", cat_list = "batch",
                       seed = 7L)
abundance <- censor(sim$y, 0.3)
se <- SummarizedExperiment::SummarizedExperiment(
    assays = list(abundance = t(abundance)), colData = meta)
fit <- metvae(se, assay_name = "abundance",
              continuous_covariate_keys = "age",
              categorical_covariate_keys = "batch", learning_rate = 1e-2,
              verbose = FALSE)
fit <- get_corr(fit)
res_sec <- sparse_by_sec(fit)

# Vignette: covariate adjustment.
set.seed(11)
meta2 <- data.frame(batch = rep(c("A", "B"), each = 50L),
                    row.names = paste0("s", 0:99))
sim2 <- metvae_sim_data(n = 100L, d = 40L, cor_pairs = 8L, mu = 10:14,
                        x = meta2, cat_list = "batch", da_prop = 0.5,
                        seed = 11L)
abundance2 <- censor(sim2$y, 0.2)
fit_adj <- metvae(abundance2, meta = meta2,
                  categorical_covariate_keys = "batch",
                  learning_rate = 1e-2, max_epochs = 500L, verbose = FALSE)
fit_adj <- get_corr(fit_adj)
fit_unadj <- metvae(abundance2, learning_rate = 1e-2, max_epochs = 500L,
                    verbose = FALSE)
fit_unadj <- get_corr(fit_unadj)

fits <- list(example = fit_ex, quickstart = fit, quickstart_sec = res_sec,
             confounded_adjusted = fit_adj,
             confounded_unadjusted = fit_unadj)
saveRDS(fits, file.path("inst", "extdata", "metvae_example_fits.rds"),
        compress = "xz")
