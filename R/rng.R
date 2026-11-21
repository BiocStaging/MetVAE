# Random draws that reproduce the streams of the Python package. Draws come
# from the global torch generator or from a seeded torch generator; draws on
# a CUDA device are made on the CPU and moved, so CPU and CUDA runs agree.

# Seeded CPU generator.
.new_gen <- function(seed) {
    g <- torch::torch_generator()
    g$set_current_seed(as.integer(seed))
    g
}

# Standard normal float64 draws of a given shape.
.rnorm_t <- function(shape, device = "cpu", gen = NULL) {
    f64 <- torch::torch_float64()
    x <- if (is.null(gen)) {
        torch::torch_randn(shape, dtype = f64)
    } else {
        torch::torch_empty(shape, dtype = f64)$normal_(generator = gen)
    }
    x$to(device = device)
}

# Uniform float32 draws on [0, 1) of a given shape.
.runif_f32 <- function(shape, device = "cpu", gen = NULL) {
    f32 <- torch::torch_float32()
    x <- if (is.null(gen)) {
        torch::torch_rand(shape, dtype = f32)
    } else {
        torch::torch_empty(shape, dtype = f32)$uniform_(generator = gen)
    }
    x$to(device = device)
}

# State of the batch-order generator: the global generator state right after
# torch_manual_seed(seed). The global generator is reseeded afterwards.
.batch_rng_init <- function(seed) {
    torch::torch_manual_seed(as.integer(seed))
    state <- torch::torch_get_rng_state()
    torch::torch_manual_seed(as.integer(seed))
    state
}

# One epoch of 0-based sample indices with the draws of a torch DataLoader
# with a RandomSampler: one int64 base-seed draw and two permutations. The
# global generator state is left unchanged. Returns the permutation and the
# updated batch-order state.
.epoch_permutation <- function(n, shuffle, state) {
    global <- torch::torch_get_rng_state()
    on.exit(torch::torch_set_rng_state(global), add = TRUE)
    torch::torch_set_rng_state(state)
    invisible(torch::torch_empty(1L, dtype = torch::torch_long())$random_())
    if (shuffle) {
        perm <- torch::torch_randperm(n, dtype = torch::torch_long())
        invisible(torch::torch_randperm(n, dtype = torch::torch_long()))
    } else {
        perm <- torch::torch_arange(0L, n - 1L, dtype = torch::torch_long())
    }
    list(perm = perm, state = torch::torch_get_rng_state())
}

# Permutation of 0:(n - 1) from a generator seeded with 'seed'; the global
# generator state is left unchanged.
.randperm_seeded <- function(n, seed) {
    global <- torch::torch_get_rng_state()
    on.exit(torch::torch_set_rng_state(global), add = TRUE)
    torch::torch_manual_seed(as.integer(seed))
    torch::torch_randperm(n, dtype = torch::torch_long())
}
