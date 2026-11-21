# Variational autoencoder: layer-normalized linear or multilayer encoder for
# the latent mean and scale, linear decoder, and a scalar log-scale of the
# reconstruction standard deviation.

# Activation module for a name.
.activation <- function(name) {
    switch(name,
           relu = torch::nn_relu(),
           tanh = torch::nn_tanh(),
           gelu = torch::nn_gelu(),
           silu = torch::nn_silu(),
           none = torch::nn_identity(),
           stop("Unknown activation: ", name, call. = FALSE))
}

# Linear layers with activations between them.
.mlp <- function(in_dim, hidden_dims, out_dim, activation) {
    dims <- c(in_dim, hidden_dims)
    layers <- list()
    for (i in seq_len(length(dims) - 1L)) {
        layers <- c(layers, list(torch::nn_linear(dims[i], dims[i + 1L]),
                                 .activation(activation)))
    }
    layers <- c(layers, list(torch::nn_linear(dims[length(dims)], out_dim)))
    do.call(torch::nn_sequential, layers)
}

# VAE module with float64 parameters. Construction draws the initial weights
# from the global torch random number generator in the order of the Python
# package: encode_mu, encode_rho, decode_mu.
.vae_module <- function(input_dim, latent_dim, hidden_dims = NULL,
                        activation = "relu") {
    generator <- torch::nn_module(
        "metvae_vae",
        initialize = function(input_dim, latent_dim, hidden_dims,
                              activation) {
            self$encnorm <- torch::nn_layer_norm(input_dim)
            self$encode_mu <- .mlp(input_dim, hidden_dims, latent_dim,
                                   activation)
            self$encode_rho <- .mlp(input_dim, hidden_dims, latent_dim,
                                    activation)
            self$decode_mu <- torch::nn_linear(latent_dim, input_dim)
            self$decode_rho <- torch::nn_parameter(
                torch::torch_tensor(-2, dtype = torch::torch_float64()))
        })
    .with_float64(generator(input_dim, latent_dim, hidden_dims, activation))
}

# Parameters in the order of the Python state_dict, which lists the scalar
# decode_rho first. Child modules of nn_sequential are numbered from 0 in both
# packages, so the parameter names agree.
.vae_params <- function(m) {
    p <- m$named_parameters()
    keys <- names(p)
    p[c("decode_rho", setdiff(keys, "decode_rho"))]
}

# Latent mean and standard deviation.
.vae_encode <- function(m, y) {
    h <- m$encnorm(y)
    list(mu = m$encode_mu(h),
         std = torch::nnf_softplus(m$encode_rho(h))$add(1e-4))
}

# Latent draw and reconstruction; 'gen' is a torch generator or NULL for the
# global random number generator.
.vae_forward <- function(m, y, gen = NULL) {
    enc <- .vae_encode(m, y)
    eps <- .rnorm_t(enc$std$size(), y$device, gen)
    z <- enc$mu + eps * enc$std
    list(mu = enc$mu, std = enc$std, recon = m$decode_mu(z))
}

# Negative evidence lower bound averaged over the batch.
.training_step <- function(m, y) {
    out <- .vae_forward(m, y)
    encode_std <- out$std$clamp(min = 1e-3, max = 10)
    encode_logvar <- encode_std$log()$mul(2)
    decode_std <- torch::nnf_softplus(m$decode_rho)$add(1e-4)$
        clamp(min = 1e-3, max = 10)
    s <- decode_std$expand(out$recon$size())
    var <- s$pow(2)
    recon_ll <- ((y - out$recon)$pow(2)$neg() / var$mul(2) - s$log())$
        sub(log(sqrt(2 * pi)))$sum(dim = -1L)
    neg_kl <- (encode_logvar$add(1) - out$mu$pow(2) - encode_logvar$exp())$
        sum(dim = -1L)$mul(0.5)
    (recon_ll + neg_kl)$mean()$neg()
}
