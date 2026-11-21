# Numerical components that need LibTorch.

test_that("random initial values fill only the censored entries", {
    skip_if_no_libtorch()
    y <- matrix(stats::rnorm(60), 12, 5)
    y[c(1, 4, 7), 2] <- NaN
    y[2, 5] <- NaN
    yt <- MetVAE:::.as_t(y)
    nz <- MetVAE:::.as_t(c(0, 3, 0, 0, 1), dtype = torch::torch_float32())
    out <- MetVAE:::.random_initial(yt, 12L, nz, MetVAE:::.as_t(rep(0, 5)),
                                    MetVAE:::.as_t(rep(1, 5)),
                                    gen = MetVAE:::.new_gen(1L))
    o <- MetVAE:::.as_r(out)
    expect_false(anyNA(o))
    expect_identical(o[!is.na(y)], y[!is.na(y)])
    again <- MetVAE:::.as_r(MetVAE:::.random_initial(
        yt, 12L, nz, MetVAE:::.as_t(rep(0, 5)), MetVAE:::.as_t(rep(1, 5)),
        gen = MetVAE:::.new_gen(1L)))
    expect_identical(o, again)
})

test_that("the two correlation paths agree on positive data", {
    skip_if_no_libtorch()
    set.seed(3)
    x <- MetVAE:::.as_t(exp(matrix(stats::rnorm(200), 40, 5)))
    a <- MetVAE:::.as_r(MetVAE:::.corr_dense(x, threshold = 0))
    b <- MetVAE:::.as_r(MetVAE:::.corr_general(x, threshold = 0))
    expect_equal(a, b, tolerance = 1e-12)
    expect_true(isSymmetric(a))
    expect_true(all(diag(a) == 1))
    expect_true(all(abs(a) <= 1))
    expect_error(MetVAE:::.corr_dense(x[, 1:2]), "d >= 3")
})

test_that("LBFGS minimizes a quadratic", {
    skip_if_no_libtorch()
    A <- MetVAE:::.as_t(diag(c(1, 4, 9)))
    target <- MetVAE:::.as_t(c(1, -2, 3))
    fn <- function(x) {
        p <- x$detach()$requires_grad_(TRUE)
        r <- p - target
        loss <- (r$matmul(A) * r)$sum()
        list(loss = loss$item(),
             grad = torch::autograd_grad(loss, p)[[1L]]$detach())
    }
    x <- MetVAE:::.lbfgs(fn, MetVAE:::.as_t(c(0, 0, 0)), max_iter = 50L,
                         tolerance_grad = 1e-10, tolerance_change = 1e-12)
    expect_equal(as.numeric(x), c(1, -2, 3), tolerance = 1e-6)
})

test_that("SEC returns a sparse correlation matrix", {
    skip_if_no_libtorch()
    set.seed(4)
    x <- MetVAE:::.as_t(exp(matrix(stats::rnorm(400), 50, 8)))
    Rn <- MetVAE:::.corr_dense(x, threshold = 0)
    R <- MetVAE:::.as_r(MetVAE:::.sec_dense(Rn, 0.5, n_samples = 50L))
    expect_true(isSymmetric(R))
    expect_equal(diag(R), rep(1, 8), tolerance = 1e-4)
    expect_gt(min(eigen(R, symmetric = TRUE)$values), -1e-8)
    expect_error(MetVAE:::.sec_dense(Rn, 0.5, max_iter = 0L), "max_iter")
})

test_that("cross-validation does not evaluate a value of c twice", {
    skip_if_no_libtorch()
    set.seed(5)
    X <- MetVAE:::.as_t(exp(matrix(stats::rnorm(600), 60, 10)))
    Rn <- MetVAE:::.corr_dense(X)
    cv <- suppressWarnings(MetVAE:::.sec_cv(
        X, Rn, c_grid = c(1, 2, 3), n_splits = 3L, seed = 0L, refine = TRUE,
        refine_points = 5L, sec_args = list(n_samples = 60L)))
    expect_false(anyDuplicated(cv$scores$rho) > 0)
    expect_false(is.unsorted(cv$scores$rho))
    expect_true(cv$best_rho %in% cv$scores$rho)
})
