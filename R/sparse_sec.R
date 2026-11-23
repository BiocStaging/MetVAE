#' Sparsify the correlation matrix by SEC
#'
#' @description
#' Computes the sparse estimate of the correlation matrix of Cui, Leng and Sun
#' (2016), which minimizes \code{0.5 * ||R - Rn||_F^2 + rho * sum(W * |R|)}
#' over correlation matrices R with eigenvalues at least \code{epsilon}, where
#' Rn is the estimate from \code{\link{get_corr}} and
#' \code{W = 1 / |Rn|}. The penalty is \code{rho} when given, and otherwise
#' chosen by K-fold cross-validation over
#' \code{rho = c * sqrt(log(p) / n)} for c in \code{c_grid}.
#'
#' @param fit An object returned by \code{\link{get_corr}}.
#' @param rho Penalty. NULL (default) selects it by cross-validation.
#' @param c_grid Values of c for the cross-validation. Default 1, ..., 10.
#' @param n_splits Number of folds. Default 5.
#' @param refine Logical; TRUE (default) evaluates a finer grid between the
#' neighbors of the best value of c.
#' @param refine_points Number of points of the finer grid. Default 10.
#' @param threshold Off-diagonal entries of the estimate with absolute value
#' below this value are set to zero. Default 0.1.
#' @param epsilon Lower bound on the eigenvalues. Default 1e-5.
#' @param tol Convergence tolerance of the relative gradient residual.
#' Default 1e-3.
#' @param max_iter Maximum number of iterations. Default 1000.
#' @param restart Period of the momentum restart; NULL or 0 disables it.
#' Default 50.
#' @param line_search_apg Logical; TRUE (default) adapts the step size.
#' @param delta Entries of Rn with absolute value below delta are fixed at
#' zero; NULL (default) uses \code{c_delta * sqrt(log(p) / n)}.
#' @param c_delta Scale of the default delta. Default 0.1.
#' @param seed Integer seed of the fold assignment. Default 0.
#'
#' @return A list with \code{estimate} (the estimate of \code{get_corr}),
#' \code{sparse_estimate} (sparse features x features matrix of class
#' \code{dgCMatrix}), \code{best_rho} (the penalty used), and
#' \code{scores_by_rho} (data.frame with columns c, rho, and the mean
#' validation error score, ordered by rho; NULL when \code{rho} is given).
#'
#' @details
#' The columns of Rn are centered and scaled before the iterations and the
#' result is symmetrized, as in the Python package metvae. In Cui, Leng and
#' Sun (2016) the standardization applies to the data variables.
#'
#' The implementation is adapted from the MATLAB code of Cui, Leng and Sun
#' (2016) and is distributed with the permission of Chenlei Leng.
#'
#' @references
#' Cui Y, Leng C, Sun D. Sparse estimation of high-dimensional correlation
#' matrices. Computational Statistics and Data Analysis 93 (2016) 390-403.
#' \doi{10.1016/j.csda.2014.10.001}
#'
#' @seealso \code{\link{get_corr}}, \code{\link{sparse_by_p}}
#'
#' @examples
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     torch::torch_is_installed()) {
#'     fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
#'                                 package = "MetVAE"))
#'     res <- sparse_by_sec(fits$example, c_grid = c(1, 2, 3),
#'                          refine = FALSE)
#'     res$best_rho
#' }
#'
#' @export
sparse_by_sec <- function(fit, rho = NULL, c_grid = seq_len(10L),
                          n_splits = 5L, refine = TRUE, refine_points = 10L,
                          threshold = 0.1, epsilon = 1e-5, tol = 1e-3,
                          max_iter = 1000L, restart = 50L,
                          line_search_apg = TRUE, delta = NULL,
                          c_delta = 0.1, seed = 0L) {
    .check_fit(fit)
    .check_corr(fit)
    .check_flag(refine, "refine")
    .check_flag(line_search_apg, "line_search_apg")
    .metvae_require_torch("sparse_by_sec()")
    n <- length(fit$sample_names)
    sec_args <- list(epsilon = epsilon, tol = tol,
                     max_iter = .check_count(max_iter, "max_iter"),
                     restart = restart, line_search_apg = line_search_apg,
                     delta = delta, n_samples = n, c_delta = c_delta,
                     threshold = threshold)
    Rn <- .as_t(as.matrix(fit$corr$estimate))
    if (!is.null(rho)) {
        rho <- .check_number(rho, "rho", lower = 0)
        R_hat <- do.call(.sec_dense, c(list(Rn = Rn, rho = rho), sec_args))
        res <- list(best_rho = rho, scores = NULL, estimate = R_hat)
    } else {
        X <- .as_t(fit$corr$impute_log_data)$exp()
        res <- .sec_cv(X, Rn, c_grid, .check_count(n_splits, "n_splits", 2L),
                       .check_count(seed, "seed", 0L), refine,
                       .check_count(refine_points, "refine_points"), sec_args)
    }
    names <- fit$feature_names
    list(estimate = fit$corr$estimate,
         sparse_estimate = .as_sparse(res$estimate, names),
         best_rho = res$best_rho, scores_by_rho = res$scores)
}
