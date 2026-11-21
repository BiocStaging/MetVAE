# Shared helpers for the tests.

# Skips a test unless torch and its LibTorch library are installed.
skip_if_no_libtorch <- function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "LibTorch is not installed")
}

# Small abundance matrix (samples x features) with zeros and names.
toy_abundance <- function(n = 8L, d = 5L, seed = 1L) {
    set.seed(seed)
    x <- matrix(exp(stats::rnorm(n * d, mean = 5)), nrow = n,
                dimnames = list(paste0("s", seq_len(n)),
                                paste0("f", seq_len(d))))
    x[1L, 1L] <- 0
    x[2L, 2L] <- 0
    x
}

# Sample metadata matching toy_abundance().
toy_meta <- function(n = 8L) {
    data.frame(age = seq(20, by = 5, length.out = n),
               batch = rep(c("B", "A"), length.out = n),
               row.names = paste0("s", seq_len(n)))
}
