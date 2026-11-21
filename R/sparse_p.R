#' Sparsify the correlation matrix by Fisher z-tests
#'
#' @description
#' Tests every off-diagonal correlation of the estimate from
#' \code{\link{get_corr}} with the Fisher z-transformation, adjusts the
#' p-values for multiplicity, and sets the correlations whose adjusted p-value
#' exceeds \code{cutoff} to zero.
#'
#' @param fit An object returned by \code{\link{get_corr}}.
#' @param p_adj_method Multiplicity adjustment, with the names and formulas of
#' statsmodels: "fdr_bh" (Benjamini-Hochberg, default), "fdr_by"
#' (Benjamini-Yekutieli), "bonferroni", "sidak", "holm", "holm-sidak",
#' "simes-hochberg", "hommel", "fdr_tsbh" and "fdr_tsbky" (two-stage
#' procedures at level 0.05).
#' @param cutoff Correlations with an adjusted p-value greater than this value
#' are set to zero. Default 0.05.
#' @param pq Form of the p-values and adjusted p-values: "dense" (default)
#' returns features x features matrices; "edges" returns a data.frame with one
#' row per feature pair with a nonzero estimate, which needs much less memory
#' for many features; "none" omits them.
#'
#' @return A list with \code{estimate} (sparse features x features matrix of
#' class \code{dgCMatrix}; the estimate of \code{get_corr} with entries
#' restricted to [-1 + 1e-7, 1 - 1e-7] and unit diagonal), \code{p_value} and
#' \code{q_value} (unadjusted and adjusted p-values, see \code{pq}; zero on
#' the diagonal), and \code{sparse_estimate} (sparse matrix of the retained
#' correlations).
#'
#' @details
#' The test statistic of a pair with correlation r is
#' \code{atanh(r) * sqrt(n - 3)}, where n is the number of samples. Pairs with
#' r = 0 have p-value 1 and are included in the multiplicity adjustment.
#'
#' @seealso \code{\link{get_corr}}, \code{\link{sparse_by_sec}},
#' \code{\link{export_graphml}}
#'
#' @examples
#' fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
#'                             package = "MetVAE"))
#' res <- sparse_by_p(fits$example, cutoff = 0.05)
#' res$sparse_estimate[1:4, 1:4]
#'
#' @export
sparse_by_p <- function(fit, p_adj_method = "fdr_bh", cutoff = 0.05,
                        pq = c("dense", "edges", "none")) {
    .check_fit(fit)
    .check_corr(fit)
    p_adj_method <- match.arg(p_adj_method, .p_adjust_methods)
    cutoff <- .check_number(cutoff, "cutoff", lower = 0, upper = 1)
    pq <- match.arg(pq)
    n <- length(fit$sample_names)
    if (n <= 3L) {
        stop("Sample size must be > 3 for Fisher's z-test.", call. = FALSE)
    }
    names <- fit$feature_names
    Rn <- as.matrix(fit$corr$estimate)
    Rn <- pmin(pmax(Rn, -1 + 1e-7), 1 - 1e-7)
    diag(Rn) <- 1
    lower <- which(lower.tri(Rn))
    r <- Rn[lower]
    z <- 0.5 * (log1p(r) - log1p(-r)) / (1 / sqrt(n - 3))
    p <- 2 * (1 - stats::pnorm(abs(z)))
    q <- .p_adjust_sm(p, p_adj_method)
    sparse <- Rn
    sparse[lower[q > cutoff]] <- 0
    sparse[upper.tri(sparse)] <- t(sparse)[upper.tri(sparse)]
    list(estimate = .as_sparse(Rn, names),
         p_value = .pq_output(p, lower, Rn, names, pq, "p_value"),
         q_value = .pq_output(q, lower, Rn, names, pq, "q_value"),
         sparse_estimate = .as_sparse(sparse, names))
}

# p-values in the form requested by 'pq'.
.pq_output <- function(v, lower, Rn, names, pq, label) {
    if (pq == "none") {
        return(NULL)
    }
    d <- nrow(Rn)
    if (pq == "dense") {
        out <- matrix(0, d, d, dimnames = list(names, names))
        out[lower] <- v
        return(out + t(out))
    }
    keep <- Rn[lower] != 0
    rows <- (lower[keep] - 1L) %% d + 1L
    cols <- (lower[keep] - 1L) %/% d + 1L
    out <- data.frame(feature1 = names[cols], feature2 = names[rows],
                      estimate = Rn[lower][keep], value = v[keep])
    names(out)[4L] <- label
    out
}
