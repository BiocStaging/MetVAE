# Training of the variational autoencoder by minibatch AdamW.

# One epoch: batches of the permuted samples, random initial values for the
# censored entries of batches that contain them, and one optimizer step per
# batch. Returns the mean batch loss and the optimizer state.
.train_epoch <- function(m, params, pp, perm, settings, lr, state) {
    n <- pp$clr_data$size(1)
    bs <- settings$batch_size
    running <- 0
    num_batches <- 0L
    for (s in seq(1L, n, by = bs)) {
        idx <- perm[s:min(s + bs - 1L, n)]$add(1L)
        y <- pp$clr_data$index_select(1L, idx)
        batch_nan <- .item(pp$rows_with_nan$index_select(1L, idx)$any())
        if (pp$has_zero && batch_nan) {
            y <- .random_initial(y, n, pp$num_zero, pp$clr_mean, pp$clr_sd)
        }
        loss <- .training_step(m, y)
        for (p in params) {
            p$grad <- NULL
        }
        loss$backward()
        if (!is.null(settings$max_grad_norm)) {
            .clip_grad_norm(params, settings$max_grad_norm)
        }
        state <- .adamw_step(params, state, lr)
        running <- running + .item(loss)
        num_batches <- num_batches + 1L
    }
    list(loss = running / num_batches, state = state)
}

# Trains the module 'm' on the preprocessed tensors 'pp'. The global torch
# generator is reseeded with 'seed'; the batch order uses a separate stream
# started from the same seed. Returns the loss of each epoch.
.train_vae <- function(m, pp, settings, seed, verbose) {
    torch::torch_manual_seed(as.integer(seed))
    batch_state <- .batch_rng_init(seed)
    params <- .vae_params(m)
    state <- .adamw_init(params)
    lrs <- .lr_schedule(settings$learning_rate, settings$max_epochs)
    pp$has_zero <- .item(pp$num_zero$ne(0)$any())
    pp$rows_with_nan <- torch::torch_isnan(pp$clr_data)$any(dim = 2L)
    n <- pp$clr_data$size(1)
    losses <- numeric(settings$max_epochs)
    bar <- .progress_start(settings$max_epochs, verbose)
    m$train()
    for (e in seq_len(settings$max_epochs)) {
        ep <- .epoch_permutation(n, settings$shuffle, batch_state)
        batch_state <- ep$state
        perm <- ep$perm
        res <- .train_epoch(m, params, pp, perm, settings, lrs[e], state)
        state <- res$state
        losses[e] <- res$loss
        .progress_update(bar, e)
    }
    .progress_close(bar)
    losses
}

# Text progress bar shown in interactive sessions when 'verbose' is TRUE.
.progress_start <- function(total, verbose) {
    if (!(isTRUE(verbose) && interactive())) {
        return(NULL)
    }
    utils::txtProgressBar(min = 0, max = total, style = 3)
}
.progress_update <- function(bar, value) {
    if (!is.null(bar)) {
        utils::setTxtProgressBar(bar, value)
    }
    invisible(NULL)
}
.progress_close <- function(bar) {
    if (!is.null(bar)) {
        close(bar)
    }
    invisible(NULL)
}
