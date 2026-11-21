#' Export correlation networks as GraphML
#'
#' @description
#' Builds one undirected network per cutoff from a sparse correlation matrix:
#' the nodes are features and an edge joins two features whose correlation
#' has absolute value at least the cutoff. The networks are returned as
#' \code{igraph} objects and optionally written as GraphML files, which can be
#' opened in Cytoscape.
#'
#' @param x A features x features correlation matrix (matrix, data.frame, or
#' \code{Matrix}), or a result of \code{\link{sparse_by_p}} or
#' \code{\link{sparse_by_sec}}, whose \code{sparse_estimate} is used.
#' @param cutoffs Positive cutoffs of the absolute correlation. Default 0.7.
#' @param output_dir Directory for the GraphML files, created if absent; NULL
#' (default) writes no files.
#' @param file_prefix Prefix of the file names; the cutoff and ".graphml" are
#' appended. Default "correlation_graph_cutoff".
#'
#' @return A named list of \code{igraph} graphs, one per cutoff with at least
#' one edge, named "Correlation_cutoff" followed by the cutoff. Every edge has
#' the attributes weight, correlation, and EdgeScore (all equal to the
#' correlation) and EdgeType and id (the graph name).
#'
#' @seealso \code{\link{sparse_by_p}}, \code{\link{sparse_by_sec}}
#'
#' @examples
#' fits <- readRDS(system.file("extdata", "metvae_example_fits.rds",
#'                             package = "MetVAE"))
#' res <- sparse_by_p(fits$example)
#' graphs <- export_graphml(res, cutoffs = c(0.3, 0.5))
#' graphs
#'
#' @export
export_graphml <- function(x, cutoffs = 0.7, output_dir = NULL,
                           file_prefix = "correlation_graph_cutoff") {
    edges <- .upper_edges(.square_matrix(x))
    cutoffs <- sort(unique(as.numeric(cutoffs[cutoffs > 0])),
                    decreasing = TRUE)
    graphs <- list()
    if (!is.null(output_dir)) {
        dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    }
    for (cutoff in cutoffs) {
        sub <- edges[abs(edges$correlation) >= cutoff, , drop = FALSE]
        if (nrow(sub) == 0L) {
            next
        }
        label <- paste0("Correlation_cutoff", formatC(cutoff, format = "g"))
        sub$weight <- sub$correlation
        sub$EdgeScore <- sub$correlation
        sub$EdgeType <- label
        sub$id <- label
        g <- igraph::graph_from_data_frame(sub, directed = FALSE)
        graphs[[label]] <- g
        if (!is.null(output_dir)) {
            path <- file.path(output_dir, paste0(
                file_prefix, formatC(cutoff, format = "g"), ".graphml"))
            igraph::write_graph(g, path, format = "graphml")
        }
    }
    graphs
}

# Square numeric matrix with names from the accepted inputs.
.square_matrix <- function(x) {
    if (is.list(x) && !is.data.frame(x) && !is.null(x$sparse_estimate)) {
        x <- x$sparse_estimate
    }
    x <- as.matrix(x)
    if (!is.numeric(x) || nrow(x) != ncol(x)) {
        stop("'x' must be a square numeric matrix or a result of ",
             "sparse_by_p() or sparse_by_sec().", call. = FALSE)
    }
    if (is.null(rownames(x))) {
        rownames(x) <- paste0("feature", seq_len(nrow(x)))
    }
    x
}

# Nonzero, non-missing entries of the strict upper triangle as an edge list,
# in row-major order.
.upper_edges <- function(x) {
    idx <- which(upper.tri(x) & !is.na(x) & x != 0, arr.ind = TRUE)
    idx <- idx[order(idx[, 1L], idx[, 2L]), , drop = FALSE]
    names <- rownames(x)
    data.frame(node1 = names[idx[, 1L]], node2 = names[idx[, 2L]],
               correlation = as.numeric(x[idx]), stringsAsFactors = FALSE)
}
