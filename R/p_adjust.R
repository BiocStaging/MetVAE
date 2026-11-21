# Multiple-testing adjustments of statsmodels.stats.multitest.multipletests
# (alpha = 0.05), with the same arithmetic.

.p_adjust_methods <- c("bonferroni", "sidak", "holm-sidak", "holm",
                       "simes-hochberg", "hommel", "fdr_bh", "fdr_by",
                       "fdr_tsbh", "fdr_tsbky")

# Adjusted p-values of the vector p.
.p_adjust_sm <- function(p, method = "fdr_bh", alpha = 0.05) {
    method <- match.arg(method, .p_adjust_methods)
    m <- length(p)
    if (m == 0L) {
        return(numeric(0L))
    }
    o <- order(p, method = "radix")
    ps <- p[o]
    q <- switch(method,
                "bonferroni" = ps * m,
                "sidak" = -expm1(m * log1p(-ps)),
                "holm-sidak" = cummax(-expm1((m:1) * log1p(-ps))),
                "holm" = cummax(ps * (m:1)),
                "simes-hochberg" = rev(cummin(rev((m:1) * ps))),
                "hommel" = .hommel(ps),
                "fdr_bh" = .fdr_bh(ps),
                "fdr_by" = .fdr_bh(ps, by = TRUE),
                "fdr_tsbh" = .fdr_twostage(ps, alpha, "bh"),
                "fdr_tsbky" = .fdr_twostage(ps, alpha, "bky"))
    q[q > 1] <- 1
    out <- numeric(m)
    out[o] <- q
    out
}

# Benjamini-Hochberg (or Benjamini-Yekutieli) adjustment of sorted p-values,
# capped at 1.
.fdr_bh <- function(ps, by = FALSE) {
    m <- length(ps)
    ecdf <- seq_len(m) / m
    if (by) {
        ecdf <- ecdf / sum(1 / seq_len(m))
    }
    q <- rev(cummin(rev(ps / ecdf)))
    q[q > 1] <- 1
    q
}

# Number of rejections of the Benjamini-Hochberg step-up rule at level alpha.
.bh_rejections <- function(ps, alpha) {
    hit <- which(ps <= (seq_along(ps) / length(ps)) * alpha)
    if (length(hit) == 0L) 0L else max(hit)
}

# Two-stage adaptive adjustment (Benjamini, Krieger and Yekutieli 'bky', or
# Benjamini and Hochberg 'bh') with one iteration.
.fdr_twostage <- function(ps, alpha, method) {
    m <- length(ps)
    fact <- if (method == "bky") 1 + alpha else 1
    alpha_prime <- alpha / fact
    q <- .fdr_bh(ps)
    r1 <- .bh_rejections(ps, alpha_prime)
    if (r1 == 0L || r1 == m) {
        return(q * fact)
    }
    m0 <- 1 * m - r1
    q <- q * (m0 * 1 / m)
    if (method == "bky") {
        q <- q * (1 + alpha)
    }
    q
}

# Hommel adjustment of sorted p-values.
.hommel <- function(ps) {
    m <- length(ps)
    a <- ps
    for (k in rev(seq_len(m))[-m]) {
        tail <- (m - k + 1L):m
        cim <- min(k * ps[tail] / seq_len(k))
        a[tail] <- pmax(a[tail], cim)
        if (k < m) {
            head <- seq_len(m - k)
            a[head] <- pmax(a[head], pmin(k * ps[head], cim))
        }
    }
    a
}
