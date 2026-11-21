#' MetVAE: sparse correlation networks for untargeted metabolomics data
#'
#' @description
#' MetVAE estimates a sparse correlation network among metabolites from an
#' abundance table with samples in rows and metabolites in columns, or from a
#' \code{SummarizedExperiment} with metabolites in rows. Zeros are treated as
#' values below the detection limit and are imputed by a variational
#' autoencoder fitted on the centered log-ratio scale. Covariate effects are
#' removed before the correlations are estimated, and the correlation matrix is
#' sparsified by Fisher z-tests with a multiplicity correction or by the sparse
#' estimation of correlation (SEC) estimator.
#'
#' The model is fitted with the R package \code{torch}. Its LibTorch library is
#' downloaded once by running \code{torch::install_torch()}.
#'
#' @references
#' Lin H, Zhang L, Lotfi A, Jarmusch A, Lee I, Kim A, Morton JT, Aksenov A.
#' Protocol for constructing correlation-based molecular networks from
#' large-scale untargeted metabolomics data. bioRxiv (2026).
#' \doi{10.1101/2025.04.26.649581}
#'
#' Cui Y, Leng C, Sun D. Sparse estimation of high-dimensional correlation
#' matrices. Computational Statistics and Data Analysis 93 (2016) 390-403.
#' \doi{10.1016/j.csda.2014.10.001}
#'
#' @keywords internal
"_PACKAGE"
