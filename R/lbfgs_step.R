# Outer iteration of LBFGS (torch.optim.LBFGS.step with line_search_fn =
# "strong_wolfe", lr = 1, history_size = 100, one call).

# Two-loop recursion: search direction from the gradient and the history.
.lbfgs_direction <- function(flat_grad, hist, H_diag) {
    num_old <- length(hist$dirs)
    al <- numeric(num_old)
    q <- flat_grad$neg()
    for (i in rev(seq_len(num_old))) {
        al[i] <- .item(hist$stps[[i]]$dot(q)) * hist$ro[i]
        q$add_(hist$dirs[[i]], alpha = -al[i])
    }
    r <- q$mul(H_diag)
    for (i in seq_len(num_old)) {
        be_i <- .item(hist$dirs[[i]]$dot(r)) * hist$ro[i]
        r$add_(hist$stps[[i]], alpha = al[i] - be_i)
    }
    r
}

# History update with the curvature pair (y, s) when y's > 1e-10.
.lbfgs_update <- function(hist, y, s, history_size) {
    ys <- .item(y$dot(s))
    if (ys > 1e-10) {
        if (length(hist$dirs) == history_size) {
            hist$dirs <- hist$dirs[-1L]
            hist$stps <- hist$stps[-1L]
            hist$ro <- hist$ro[-1L]
        }
        hist$dirs <- c(hist$dirs, list(y))
        hist$stps <- c(hist$stps, list(s))
        hist$ro <- c(hist$ro, 1 / ys)
        hist$H_diag <- ys / .item(y$dot(y))
    }
    hist
}

# Minimizes fn (returning list(loss, grad) at a flat tensor) from x. Returns
# the final flat tensor.
.lbfgs <- function(fn, x, max_iter = 100L, tolerance_grad = 1e-6,
                   tolerance_change = 1e-6, history_size = 100L) {
    max_eval <- (max_iter * 5L) %/% 4L
    x <- x$clone()
    ev <- fn(x)
    loss <- ev$loss
    flat_grad <- ev$grad
    if (.item(flat_grad$abs()$max()) <= tolerance_grad) {
        return(x)
    }
    hist <- list(dirs = list(), stps = list(), ro = numeric(), H_diag = 1)
    current_evals <- 1L
    for (n_iter in seq_len(max_iter)) {
        if (n_iter == 1L) {
            d <- flat_grad$neg()
            t_step <- min(1, 1 / .item(flat_grad$abs()$sum()))
        } else {
            hist <- .lbfgs_update(hist, flat_grad$sub(prev_grad),
                                  d$mul(t_step), history_size)
            d <- .lbfgs_direction(flat_grad, hist, hist$H_diag)
            t_step <- 1
        }
        prev_grad <- flat_grad$clone()
        prev_loss <- loss
        gtd <- .item(flat_grad$dot(d))
        if (gtd > -tolerance_change) {
            break
        }
        ls <- .strong_wolfe(fn, x$clone(), t_step, d, loss, flat_grad, gtd)
        loss <- ls$f
        flat_grad <- ls$g
        t_step <- ls$t
        x$add_(d, alpha = t_step)
        current_evals <- current_evals + ls$evals
        if (n_iter == max_iter || current_evals >= max_eval ||
            .item(flat_grad$abs()$max()) <= tolerance_grad ||
            .item(d$mul(t_step)$abs()$max()) <= tolerance_change ||
            abs(loss - prev_loss) < tolerance_change) {
            break
        }
    }
    x
}
