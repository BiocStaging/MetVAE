test_that("an absent torch gives the installation commands", {
    skip_if(MetVAE:::.metvae_torch_available(), "torch is available")
    expect_error(MetVAE:::.metvae_require_torch("metvae()"), "install")
})

test_that("an available torch passes the check", {
    skip_if_no_libtorch()
    expect_true(MetVAE:::.metvae_torch_available())
    expect_true(MetVAE:::.metvae_require_torch())
})
