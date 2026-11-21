# Limited-memory BFGS with a strong Wolfe line search for one flat parameter
# tensor, following torch.optim.LBFGS (torch 2.5.1). Scalars are R doubles.

# Minimizer of the cubic interpolating (x1, f1, g1) and (x2, f2, g2) within
# 'bounds'.
.cubic_interpolate <- function(x1, f1, g1, x2, f2, g2, bounds = NULL) {
    if (is.null(bounds)) {
        bounds <- if (x1 <= x2) c(x1, x2) else c(x2, x1)
    }
    d1 <- g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square <- d1^2 - g1 * g2
    if (d2_square >= 0) {
        d2 <- sqrt(d2_square)
        if (x1 <= x2) {
            min_pos <- x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        } else {
            min_pos <- x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        }
        return(min(max(min_pos, bounds[1L]), bounds[2L]))
    }
    (bounds[1L] + bounds[2L]) / 2
}

# Evaluation of the objective at x + t * d.
.directional_eval <- function(fn, x, t, d) {
    fn(x$add(d, alpha = t))
}

# Bracketing phase of the strong Wolfe line search.
.wolfe_bracket <- function(fn, x, t, d, f, g, gtd, c1, c2, max_ls) {
    ev <- .directional_eval(fn, x, t, d)
    evals <- 1L
    gtd_new <- .item(ev$grad$dot(d))
    prev <- list(t = 0, f = f, g = g, gtd = gtd)
    ls_iter <- 0L
    while (ls_iter < max_ls) {
        cur <- list(t = t, f = ev$loss, g = ev$grad, gtd = gtd_new)
        if (cur$f > (f + c1 * t * gtd) || (ls_iter > 1L && cur$f >= prev$f)) {
            return(list(br = list(prev, cur), done = FALSE, evals = evals,
                        ls_iter = ls_iter))
        }
        if (abs(gtd_new) <= -c2 * gtd) {
            return(list(br = list(cur), done = TRUE, evals = evals,
                        ls_iter = ls_iter))
        }
        if (gtd_new >= 0) {
            return(list(br = list(prev, cur), done = FALSE, evals = evals,
                        ls_iter = ls_iter))
        }
        min_step <- t + 0.01 * (t - prev$t)
        t <- .cubic_interpolate(prev$t, prev$f, prev$gtd, t, cur$f, gtd_new,
                                bounds = c(min_step, t * 10))
        prev <- cur
        ev <- .directional_eval(fn, x, t, d)
        evals <- evals + 1L
        gtd_new <- .item(ev$grad$dot(d))
        ls_iter <- ls_iter + 1L
    }
    list(br = list(list(t = 0, f = f, g = g, gtd = NA_real_),
                   list(t = t, f = ev$loss, g = ev$grad, gtd = gtd_new)),
         done = FALSE, evals = evals, ls_iter = ls_iter)
}

# Step size from the safeguarded interpolation inside a bracket.
.zoom_step <- function(br, insuf) {
    lo <- min(br[[1L]]$t, br[[2L]]$t)
    hi <- max(br[[1L]]$t, br[[2L]]$t)
    t <- .cubic_interpolate(br[[1L]]$t, br[[1L]]$f, br[[1L]]$gtd,
                            br[[2L]]$t, br[[2L]]$f, br[[2L]]$gtd)
    eps <- 0.1 * (hi - lo)
    if (min(hi - t, t - lo) < eps) {
        if (insuf || t >= hi || t <= lo) {
            t <- if (abs(t - hi) < abs(t - lo)) hi - eps else lo + eps
            insuf <- FALSE
        } else {
            insuf <- TRUE
        }
    } else {
        insuf <- FALSE
    }
    list(t = t, insuf = insuf)
}

# Strong Wolfe line search. Returns the accepted loss, gradient, step, and
# number of evaluations.
.strong_wolfe <- function(fn, x, t, d, f, g, gtd, c1 = 1e-4, c2 = 0.9,
                          tolerance_change = 1e-9, max_ls = 25L) {
    d_norm <- .item(d$abs()$max())
    b <- .wolfe_bracket(fn, x, t, d, f, g, gtd, c1, c2, max_ls)
    br <- b$br
    done <- b$done
    evals <- b$evals
    ls_iter <- b$ls_iter
    insuf <- FALSE
    low <- if (br[[1L]]$f <= br[[length(br)]]$f) 1L else 2L
    while (!done && ls_iter < max_ls) {
        if (abs(br[[2L]]$t - br[[1L]]$t) * d_norm < tolerance_change) {
            break
        }
        z <- .zoom_step(br, insuf)
        insuf <- z$insuf
        ev <- .directional_eval(fn, x, z$t, d)
        evals <- evals + 1L
        cur <- list(t = z$t, f = ev$loss, g = ev$grad,
                    gtd = .item(ev$grad$dot(d)))
        ls_iter <- ls_iter + 1L
        high <- 3L - low
        if (cur$f > (f + c1 * cur$t * gtd) || cur$f >= br[[low]]$f) {
            br[[high]] <- cur
            low <- if (br[[1L]]$f <= br[[2L]]$f) 1L else 2L
        } else {
            if (abs(cur$gtd) <= -c2 * gtd) {
                done <- TRUE
            } else if (cur$gtd * (br[[high]]$t - br[[low]]$t) >= 0) {
                br[[high]] <- br[[low]]
            }
            br[[low]] <- cur
        }
    }
    list(f = br[[low]]$f, g = br[[low]]$g, t = br[[low]]$t, evals = evals)
}
