# Input handling: abundance table and sample metadata, zero filters, cleaning,
# and the covariate design matrix. Pure R; no torch.

# Abundance matrix with samples in rows and metadata data.frame (or NULL)
# from a SummarizedExperiment, a matrix, or a data.frame.
.as_abundance <- function(x, meta, assay_name, features_as_rows) {
    if (methods::is(x, "SummarizedExperiment")) {
        if (!is.null(meta)) {
            stop("'meta' must be NULL when 'x' is a SummarizedExperiment; ",
                 "covariates are read from colData(x).", call. = FALSE)
        }
        assay_name <- .select_assay(x, assay_name)
        data <- t(as.matrix(SummarizedExperiment::assay(x, assay_name)))
        meta <- as.data.frame(SummarizedExperiment::colData(x),
                              optional = TRUE)
        rownames(meta) <- colnames(x)
    } else if (is.matrix(x) || is.data.frame(x)) {
        if (is.data.frame(x) &&
            !all(vapply(x, is.numeric, logical(1L)))) {
            stop("All columns of the data.frame 'x' must be numeric.",
                 call. = FALSE)
        }
        data <- as.matrix(x)
        if (!is.numeric(data)) {
            stop("The matrix 'x' must be numeric.", call. = FALSE)
        }
        if (features_as_rows) {
            data <- t(data)
        }
    } else {
        stop("'x' must be a SummarizedExperiment, a matrix, or a ",
             "data.frame.", call. = FALSE)
    }
    storage.mode(data) <- "double"
    list(data = .name_abundance(data), meta = meta)
}

# Name of the assay to use; the first assay when 'assay_name' is NULL.
.select_assay <- function(x, assay_name) {
    available <- SummarizedExperiment::assayNames(x)
    if (length(SummarizedExperiment::assays(x)) == 0L) {
        stop("The SummarizedExperiment 'x' has no assay.", call. = FALSE)
    }
    if (is.null(assay_name)) {
        return(1L)
    }
    if (!is.character(assay_name) || length(assay_name) != 1L ||
        !assay_name %in% available) {
        stop("'assay_name' must be one of: ",
             paste(available, collapse = ", "), ".", call. = FALSE)
    }
    assay_name
}

# Assigns default names to an unnamed abundance matrix and rejects
# duplicated names.
.name_abundance <- function(data) {
    if (is.null(rownames(data))) {
        rownames(data) <- paste0("sample", seq_len(nrow(data)))
    }
    if (is.null(colnames(data))) {
        colnames(data) <- paste0("feature", seq_len(ncol(data)))
    }
    if (anyDuplicated(rownames(data))) {
        stop("Sample names must be unique.", call. = FALSE)
    }
    if (anyDuplicated(colnames(data))) {
        stop("Feature names must be unique.", call. = FALSE)
    }
    data
}

# Checks that the covariate keys are columns of 'meta' with valid values.
.check_meta <- function(meta, cont_keys, cat_keys) {
    keys <- c(cont_keys, cat_keys)
    if (is.null(meta)) {
        if (length(keys) > 0L) {
            stop("Covariate keys were given but 'meta' is NULL.",
                 call. = FALSE)
        }
        return(invisible(NULL))
    }
    if (!is.data.frame(meta)) {
        stop("'meta' must be a data.frame with sample names as row names.",
             call. = FALSE)
    }
    if (anyDuplicated(c(cont_keys, cat_keys))) {
        stop("A covariate cannot be both continuous and categorical.",
             call. = FALSE)
    }
    absent <- setdiff(keys, colnames(meta))
    if (length(absent) > 0L) {
        stop("Covariate keys not found in the metadata: ",
             .head_list(absent), ".", call. = FALSE)
    }
    for (key in cont_keys) {
        if (!is.numeric(meta[[key]])) {
            stop("The continuous covariate '", key, "' must be numeric.",
                 call. = FALSE)
        }
    }
    invisible(NULL)
}

# Centers and scales the continuous covariates over all rows of 'meta',
# using the population standard deviation; constant columns keep scale 1.
.scale_continuous <- function(meta, cont_keys) {
    if (is.null(meta) || is.null(cont_keys)) {
        return(meta)
    }
    for (key in cont_keys) {
        v <- as.numeric(meta[[key]])
        centered <- v - mean(v, na.rm = TRUE)
        s <- sqrt(mean(centered^2, na.rm = TRUE))
        if (!is.finite(s) || s == 0) {
            s <- 1
        }
        meta[[key]] <- centered / s
    }
    meta
}

# Reorders 'meta' to the samples of 'data'; every sample must be present.
.align_meta <- function(data, meta) {
    if (is.null(meta)) {
        return(NULL)
    }
    missing <- setdiff(rownames(data), rownames(meta))
    if (length(missing) > 0L) {
        stop("The following sample names are missing in the sample ",
             "metadata: ", .head_list(missing), ".", call. = FALSE)
    }
    meta[rownames(data), , drop = FALSE]
}
