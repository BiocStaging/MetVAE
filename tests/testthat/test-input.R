prep <- function(...) MetVAE:::.metvae_input(..., verbose = FALSE)

test_that("a matrix with samples in rows is used as given", {
    x <- toy_abundance()
    out <- prep(x)
    expect_identical(out$data, x)
    expect_identical(out$sample_names, rownames(x))
    expect_identical(out$feature_names, colnames(x))
    expect_null(out$design)
    expect_equal(unname(out$num_zero), c(1, 1, 0, 0, 0))
})

test_that("features_as_rows transposes the input", {
    x <- toy_abundance()
    expect_identical(prep(t(x), features_as_rows = TRUE)$data, x)
})

test_that("a SummarizedExperiment gives the same data as the matrix", {
    x <- toy_abundance()
    meta <- toy_meta()
    se <- SummarizedExperiment::SummarizedExperiment(
        assays = list(abundance = t(x)), colData = meta)
    a <- prep(se, assay_name = "abundance",
              continuous_covariate_keys = "age",
              categorical_covariate_keys = "batch")
    b <- prep(x, meta = meta, continuous_covariate_keys = "age",
              categorical_covariate_keys = "batch")
    expect_identical(a$data, b$data)
    expect_identical(a$design, b$design)
    expect_error(prep(se, meta = meta), "colData")
    expect_error(prep(se, assay_name = "counts"), "abundance")
})

test_that("unnamed input receives default names and duplicates fail", {
    x <- unname(toy_abundance())
    out <- prep(x)
    expect_identical(out$sample_names[1:2], c("sample1", "sample2"))
    expect_identical(out$feature_names[1:2], c("feature1", "feature2"))
    y <- toy_abundance()
    colnames(y)[2] <- colnames(y)[1]
    expect_error(prep(y), "Feature names must be unique")
})

test_that("non-numeric input is rejected", {
    df <- as.data.frame(toy_abundance())
    df$f1 <- as.character(df$f1)
    expect_error(prep(df), "numeric")
    expect_error(prep(list(1, 2)), "SummarizedExperiment")
})

test_that("metadata are aligned to the samples and must cover them", {
    x <- toy_abundance()
    meta <- toy_meta()[8:1, ]
    out <- prep(x, meta = meta, continuous_covariate_keys = "age")
    expect_identical(rownames(out$design), rownames(x))
    expect_error(prep(x, meta = meta[-1, ], continuous_covariate_keys = "age"),
                 "missing in the sample metadata: s8")
    expect_error(prep(x, continuous_covariate_keys = "age"), "'meta' is NULL")
    expect_error(prep(x, meta = meta, continuous_covariate_keys = "height"),
                 "not found")
    expect_error(prep(x, meta = meta, continuous_covariate_keys = "batch"),
                 "must be numeric")
})

test_that("continuous covariates are scaled with the population SD", {
    x <- toy_abundance()
    meta <- toy_meta()
    out <- prep(x, meta = meta, continuous_covariate_keys = "age")
    v <- meta$age
    expected <- (v - mean(v)) / sqrt(mean((v - mean(v))^2))
    expect_equal(unname(out$design[, "age"]), expected)
})

test_that("categorical covariates drop the first level in byte order", {
    x <- toy_abundance()
    meta <- toy_meta()
    out <- prep(x, meta = meta, categorical_covariate_keys = "batch")
    expect_identical(colnames(out$design), "batch_B")
    expect_equal(unname(out$design[, 1]), as.numeric(meta$batch == "B"))
    meta$batch <- factor(meta$batch, levels = c("B", "A"))
    out <- prep(x, meta = meta, categorical_covariate_keys = "batch")
    expect_identical(colnames(out$design), "batch_A")
    meta$batch[1] <- NA
    expect_error(prep(x, meta = meta, categorical_covariate_keys = "batch"),
                 "missing values")
})

test_that("zero filters run on features first, then samples", {
    x <- toy_abundance()
    x[1:3, 3] <- 0
    out <- prep(x, feature_zero_threshold = 0.3)
    expect_false("f3" %in% out$feature_names)
    x2 <- toy_abundance()
    x2[1, 2:3] <- 0
    out <- prep(x2, feature_zero_threshold = NULL,
                sample_zero_threshold = 0.5)
    expect_false("s1" %in% out$sample_names)
    expect_error(prep(x, feature_zero_threshold = 2), "\\[0, 1\\]")
})

test_that("missing, negative, and all-zero values are cleaned", {
    x <- toy_abundance()
    x[3, 3] <- NA
    x[4, 4] <- -1
    x[5, ] <- 0
    expect_warning(out <- prep(x, feature_zero_threshold = NULL),
                   "1 negative values")
    expect_equal(out$data["s3", "f3"], 0)
    expect_equal(out$data["s4", "f4"], 0)
    expect_false("s5" %in% out$sample_names)
    x[6, 1] <- Inf
    expect_error(suppressWarnings(prep(x, feature_zero_threshold = NULL)),
                 "finite")
})

test_that("too few features after filtering is an error", {
    x <- toy_abundance(d = 3L)
    x[1:4, 3] <- 0
    expect_error(prep(x, feature_zero_threshold = 0.3), "At least 3 features")
})
