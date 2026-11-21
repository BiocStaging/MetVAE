# Optimizer, gradient clipping, and learning-rate schedule of the training
# loop, following torch.optim.AdamW (single-tensor update, weight decay 0),
# torch.nn.utils.clip_grad_norm_, and CosineAnnealingWarmRestarts.

# AdamW state for a list of parameters.
.adamw_init <- function(params) {
    list(exp_avg = lapply(params, torch::torch_zeros_like),
         exp_avg_sq = lapply(params, torch::torch_zeros_like),
         step = 0)
}

# One AdamW update with learning rate 'lr'; returns the updated state.
.adamw_step <- function(params, state, lr, beta1 = 0.9, beta2 = 0.999,
                        eps = 1e-8) {
    state$step <- state$step + 1
    step_size <- lr / (1 - beta1^state$step)
    bias_correction2_sqrt <- (1 - beta2^state$step)^0.5
    torch::with_no_grad({
        for (i in seq_along(params)) {
            g <- params[[i]]$grad
            state$exp_avg[[i]]$lerp_(g, 1 - beta1)
            state$exp_avg_sq[[i]]$mul_(beta2)$addcmul_(g, g,
                                                       value = 1 - beta2)
            denom <- state$exp_avg_sq[[i]]$sqrt()$div(bias_correction2_sqrt)$
                add_(eps)
            params[[i]]$addcdiv_(state$exp_avg[[i]], denom,
                                 value = -step_size)
        }
    })
    state
}

# Scales the gradients so that their joint 2-norm is at most max_norm.
.clip_grad_norm <- function(params, max_norm) {
    torch::with_no_grad({
        grads <- lapply(params, function(p) p$grad)
        norms <- lapply(grads, function(g) torch::linalg_vector_norm(g, 2))
        total <- torch::linalg_vector_norm(torch::torch_stack(norms), 2)
        coef <- .rdiv(max_norm, total$add(1e-6))$clamp(max = 1)
        for (g in grads) {
            g$mul_(coef)
        }
    })
    invisible(NULL)
}

# Learning rate of CosineAnnealingWarmRestarts(T_0 = 20, T_mult = 2,
# eta_min = lr0 / 2) for the epochs 1, ..., max_epochs.
.lr_schedule <- function(lr0, max_epochs, T_0 = 20, T_mult = 2) {
    eta_min <- lr0 / 2
    lr <- numeric(max_epochs)
    T_cur <- 0
    T_i <- T_0
    for (e in seq_len(max_epochs)) {
        lr[e] <- eta_min + (lr0 - eta_min) * (1 + cos(pi * T_cur / T_i)) / 2
        T_cur <- T_cur + 1
        if (T_cur >= T_i) {
            T_cur <- T_cur - T_i
            T_i <- T_i * T_mult
        }
    }
    lr
}
