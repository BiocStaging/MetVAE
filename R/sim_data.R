#' Simulate metabolomics abundance data
#'
#' @description
#' Simulates log-normal abundances of d features in n samples. The first
#' \code{2 * cor_pairs} features form disjoint consecutive pairs (1, 2),
#' (3, 4), ... with correlations drawn from +/-0.5, +/-0.6, and +/-0.7; the
#' other features are independent. The mean of each log abundance is drawn
#' from \code{mu}. When covariates are supplied, every correlated feature and
#' a proportion \code{da_prop} of the other features receive covariate
#' effects drawn from -2, -1, 1, 2.
#'
#' @param n Number of samples.
#' @param d Number of features.
#' @param cor_pairs Number of correlated pairs, at most \code{d \%/\% 2}.
#' Default 0.
#' @param mu Candidate means of the log abundances.
#' @param sigma Standard deviation of the log abundances. Default 1.
#' @param x A data.frame of covariates with n rows, or NULL.
#' @param cont_list Names of the continuous covariates in \code{x}; they are
#' centered and scaled.
#' @param cat_list Names of the categorical covariates in \code{x}; they are
#' coded by indicators with the first level dropped.
#' @param da_prop Proportion of uncorrelated features with covariate effects.
#' Default 0.1.
#' @param seed Integer seed, or NULL to use the current random number
#' generator state. The global state is restored after a seeded call.
#'
#' @return A list with \code{y} (n x d abundance matrix with samples s0, s1,
#' ... and features f0, f1, ...), \code{cor_matrix} (true correlation matrix
#' of the log abundances), \code{x} (covariate matrix used, or NULL), and
#' \code{beta} (d x covariates effect matrix, or NULL).
#'
#' @examples
#' sim <- metvae_sim_data(n = 50, d = 20, cor_pairs = 4, mu = 10:14,
#'                        seed = 1)
#' dim(sim$y)
#' which(sim$cor_matrix[1:4, 1:4] != 0, arr.ind = TRUE)
#'
#' @export
metvae_sim_data <- function(n, d, cor_pairs = 0L, mu, sigma = 1, x = NULL,
                            cont_list = NULL, cat_list = NULL, da_prop = 0.1,
                            seed = NULL) {
    n <- .check_count(n, "n")
    d <- .check_count(d, "d")
    cor_pairs <- .check_count(cor_pairs, "cor_pairs", lower = 0L)
    if (missing(mu) || !is.numeric(mu) || length(mu) == 0L) {
        stop("'mu' must be a numeric vector of candidate means.",
             call. = FALSE)
    }
    if (cor_pairs > d %/% 2L) {
        stop("The maximum number of correlated pairs is ", d %/% 2L, ".",
             call. = FALSE)
    }
    if (!is.null(seed)) {
        .with_seed(seed, .simulate(n, d, cor_pairs, mu, sigma, x, cont_list,
                                   cat_list, da_prop))
    } else {
        .simulate(n, d, cor_pairs, mu, sigma, x, cont_list, cat_list,
                  da_prop)
    }
}

# Evaluates 'expr' after set.seed(seed) and restores the previous state.
.with_seed <- function(seed, expr) {
    old <- if (exists(".Random.seed", envir = globalenv())) {
        get(".Random.seed", envir = globalenv())
    } else {
        NULL
    }
    on.exit({
        if (is.null(old)) {
            rm(".Random.seed", envir = globalenv())
        } else {
            assign(".Random.seed", old, envir = globalenv())
        }
    }, add = TRUE)
    set.seed(seed)
    force(expr)
}

# Simulation of metvae_sim_data().
.simulate <- function(n, d, cor_pairs, mu, sigma, x, cont_list, cat_list,
                      da_prop) {
    mu_vec <- mu[sample.int(length(mu), d, replace = TRUE)]
    cor_matrix <- diag(d)
    if (cor_pairs > 0L) {
        values <- sample(c(-0.7, -0.6, -0.5, 0.5, 0.6, 0.7), cor_pairs,
                         replace = TRUE)
        i1 <- seq(1L, 2L * cor_pairs, by = 2L)
        cor_matrix[cbind(i1, i1 + 1L)] <- values
        cor_matrix[cbind(i1 + 1L, i1)] <- values
    }
    z <- matrix(stats::rnorm(n * d), n, d)
    log_y <- z %*% chol(cor_matrix * sigma^2) +
        matrix(mu_vec, n, d, byrow = TRUE)
    design <- .sim_design(x, cont_list, cat_list)
    beta <- NULL
    if (!is.null(design)) {
        beta <- .sim_effects(d, cor_pairs, ncol(design), da_prop)
        log_y <- log_y + design %*% t(beta)
    }
    y <- exp(log_y)
    dimnames(y) <- list(paste0("s", seq_len(n) - 1L),
                        paste0("f", seq_len(d) - 1L))
    dimnames(cor_matrix) <- list(colnames(y), colnames(y))
    list(cor_matrix = cor_matrix, y = y, x = design, beta = beta)
}

# Covariate matrix of the simulation, or NULL.
.sim_design <- function(x, cont_list, cat_list) {
    if (is.null(x)) {
        return(NULL)
    }
    if (is.null(cont_list) && is.null(cat_list)) {
        stop("At least one of 'cont_list' and 'cat_list' must be given.",
             call. = FALSE)
    }
    x <- as.data.frame(x)
    .covariate_design(.scale_continuous(x, cont_list), cont_list, cat_list)
}

# Effects: every correlated feature has nonzero effects; the other features
# have a nonzero effect with probability da_prop.
.sim_effects <- function(d, cor_pairs, p, da_prop) {
    es <- c(0, -2, -1, 1, 2)
    b1 <- sample(es, 2L * cor_pairs * p, replace = TRUE,
                 prob = c(0, 0.25, 0.25, 0.25, 0.25))
    b2 <- sample(es, (d - 2L * cor_pairs) * p, replace = TRUE,
                 prob = c(1 - da_prop, rep(da_prop / 4, 4L)))
    rbind(matrix(b1, ncol = p, byrow = TRUE),
          matrix(b2, ncol = p, byrow = TRUE))
}
