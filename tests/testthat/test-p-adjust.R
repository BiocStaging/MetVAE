padj <- MetVAE:::.p_adjust_sm

test_that("adjustments agree with stats::p.adjust where both exist", {
    set.seed(1)
    p <- c(stats::runif(200)^3, 0, 1, 1, 0.5, 0.5)
    pairs <- list(bonferroni = "bonferroni", holm = "holm",
                  "simes-hochberg" = "hochberg", hommel = "hommel",
                  fdr_bh = "BH", fdr_by = "BY")
    for (m in names(pairs)) {
        expect_equal(padj(p, m), stats::p.adjust(p, pairs[[m]]),
                     tolerance = 1e-12, info = m)
    }
})

test_that("adjusted p-values are in [0, 1] and at least the raw values", {
    set.seed(2)
    p <- stats::runif(300)^2
    for (m in MetVAE:::.p_adjust_methods) {
        q <- padj(p, m)
        expect_true(all(q >= 0 & q <= 1), info = m)
        if (!m %in% c("fdr_tsbh", "fdr_tsbky")) {
            expect_true(all(q >= p - 1e-15), info = m)
        }
    }
    expect_identical(padj(numeric(0), "fdr_bh"), numeric(0))
    expect_error(padj(p, "BH"), "should be one of")
})

test_that("all ten methods reproduce statsmodels", {
    skip_if_no_pyref(c("pvec", "qvec_all", "p_adj_methods"))
    p <- read_pyref("pvec")
    q <- read_pyref("qvec_all")
    methods <- read_pyref("p_adj_methods")
    for (i in seq_along(methods)) {
        expect_lt(max_abs_diff(padj(p, methods[i]), q[i, ]), 1e-15)
    }
})
