# Functions that run on a stored fit without torch.

fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
                            package = "MetVAE"))
fit <- fits$example

test_that("the stored fits are complete", {
    for (f in fits[c("example", "quickstart", "confounded_adjusted",
                     "confounded_unadjusted")]) {
        expect_s3_class(f, "metvae")
        expect_false(is.null(f$corr))
        expect_identical(dim(f$corr$estimate), rep(length(f$feature_names),
                                                   2L))
    }
})

test_that("sparse_by_p returns consistent matrices", {
    res <- sparse_by_p(fit)
    est <- as.matrix(res$estimate)
    sp <- as.matrix(res$sparse_estimate)
    expect_true(isSymmetric(est))
    expect_true(isSymmetric(sp))
    expect_true(all(diag(sp) == 1))
    expect_true(all(sp[est == 0] == 0))
    expect_true(all(sp[sp != 0] == est[sp != 0]))
    expect_true(all(res$q_value[sp == 0 & est != 0] > 0.05))
    expect_identical(dimnames(sp), list(fit$feature_names, fit$feature_names))
    edges <- sparse_by_p(fit, pq = "edges")
    expect_identical(nrow(edges$p_value),
                     as.integer(sum(est[upper.tri(est)] != 0)))
    i <- match(edges$q_value$feature1[1], fit$feature_names)
    j <- match(edges$q_value$feature2[1], fit$feature_names)
    expect_equal(edges$q_value$q_value[1], res$q_value[i, j])
    expect_null(sparse_by_p(fit, pq = "none")$p_value)
    expect_error(sparse_by_p(fits$example[-which(names(fit) == "corr")]),
                 "metvae")
})

test_that("a fit without correlations points to get_corr", {
    f <- fit
    f["corr"] <- list(NULL)
    expect_error(sparse_by_p(f), "get_corr")
})

test_that("accessors return named matrices of the right size", {
    d <- length(fit$feature_names)
    n <- length(fit$sample_names)
    expect_identical(dim(confound_coef(fit)), c(d, 2L))
    expect_identical(colnames(confound_coef(fit)), c("age", "batch_B"))
    expect_identical(dim(confound_es(fit)), c(n, d))
    expect_equal(confound_es(fit), fit$design %*% fit$clr_coef,
                 ignore_attr = TRUE)
    load <- clr_loading(fit)
    expect_identical(colnames(load)[1], "latent_0")
    co <- cooccurrence(fit)
    expect_true(isSymmetric(co))
    expect_true(all(diag(co) == 0))
    expect_null(confound_coef(fits$confounded_unadjusted))
})

test_that("print and summary report the fit", {
    expect_output(print(fit), "MetVAE fit")
    expect_output(print(fit), "correlations: [0-9]+ nonzero pairs")
    s <- summary(fit)
    expect_s3_class(s, "summary.metvae")
    expect_identical(s$n_features, length(fit$feature_names))
    f <- fit
    f["corr"] <- list(NULL)
    expect_output(print(f), "run fit <- get_corr\\(fit\\)")
})

test_that("export_graphml builds and writes the networks", {
    res <- sparse_by_p(fit)
    dir <- withr::local_tempdir()
    g <- export_graphml(res, cutoffs = c(0.3, 0, -1), output_dir = dir)
    expect_identical(names(g), "Correlation_cutoff0.3")
    sp <- as.matrix(res$sparse_estimate)
    n_edges <- sum(abs(sp[upper.tri(sp)]) >= 0.3)
    expect_equal(igraph::ecount(g[[1]]), n_edges)
    e <- igraph::as_data_frame(g[[1]], what = "edges")
    expect_true(all(c("weight", "correlation", "EdgeScore", "EdgeType",
                      "id") %in% names(e)))
    expect_equal(e$correlation, sp[cbind(e$from, e$to)])
    file <- file.path(dir, "correlation_graph_cutoff0.3.graphml")
    expect_true(file.exists(file))
    back <- igraph::read_graph(file, format = "graphml")
    expect_identical(igraph::ecount(back), igraph::ecount(g[[1]]))
    expect_length(export_graphml(res, cutoffs = 2), 0L)
})
