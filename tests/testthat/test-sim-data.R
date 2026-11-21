test_that("simulated data have the documented structure", {
    sim <- metvae_sim_data(n = 30, d = 10, cor_pairs = 3, mu = 10:12,
                           seed = 1)
    expect_identical(dim(sim$y), c(30L, 10L))
    expect_identical(rownames(sim$y)[1:2], c("s0", "s1"))
    expect_identical(colnames(sim$y)[1:2], c("f0", "f1"))
    expect_true(all(sim$y > 0))
    cm <- sim$cor_matrix
    expect_true(isSymmetric(unname(cm)))
    expect_true(all(diag(cm) == 1))
    expect_identical(sum(cm[upper.tri(cm)] != 0), 3L)
    expect_true(all(cm[cbind(c(1, 3, 5), c(2, 4, 6))] != 0))
    expect_null(sim$x)
    expect_null(sim$beta)
})

test_that("a seed gives identical draws and leaves the global RNG alone", {
    set.seed(99)
    before <- .Random.seed
    a <- metvae_sim_data(n = 20, d = 6, cor_pairs = 2, mu = 10, seed = 3)
    expect_identical(.Random.seed, before)
    b <- metvae_sim_data(n = 20, d = 6, cor_pairs = 2, mu = 10, seed = 3)
    expect_identical(a, b)
})

test_that("covariates add effects to every correlated feature", {
    meta <- data.frame(age = seq_len(40), batch = rep(c("A", "B"), 20))
    sim <- metvae_sim_data(n = 40, d = 12, cor_pairs = 3, mu = 10:12,
                           x = meta, cont_list = "age", cat_list = "batch",
                           da_prop = 0, seed = 2)
    expect_identical(colnames(sim$x), c("age", "batch_B"))
    expect_identical(dim(sim$beta), c(12L, 2L))
    expect_true(all(sim$beta[1:6, ] != 0))
    expect_true(all(sim$beta[7:12, ] == 0))
})

test_that("invalid arguments are rejected", {
    expect_error(metvae_sim_data(n = 10, d = 4, cor_pairs = 3, mu = 10),
                 "maximum number of correlated pairs is 2")
    expect_error(metvae_sim_data(n = 10, d = 4), "'mu'")
    expect_error(metvae_sim_data(n = 10, d = 4, mu = 10,
                                 x = data.frame(a = 1:10)),
                 "cont_list")
})
