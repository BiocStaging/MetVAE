# Zero filters, cleaning of the abundance values, and the covariate design.

# Drops features, then samples, whose proportion of zeros exceeds the
# thresholds. Missing values count as zeros.
.zero_filter <- function(data, feature_zero_threshold, sample_zero_threshold,
                         verbose) {
    is_zero <- is.na(data) | data == 0
    if (!is.null(feature_zero_threshold)) {
        drop <- colMeans(is_zero) > feature_zero_threshold
        thr <- sprintf("%.2f", feature_zero_threshold)
        if (any(drop)) {
            data <- data[, !drop, drop = FALSE]
            is_zero <- is_zero[, !drop, drop = FALSE]
            .inform(verbose, "Filtered features: removed ", sum(drop),
                    " with zero proportion > ", thr)
        } else {
            .inform(verbose, "Filtered features: removed 0 (threshold ",
                    thr, ")")
        }
    }
    if (!is.null(sample_zero_threshold)) {
        drop <- rowMeans(is_zero) > sample_zero_threshold
        thr <- sprintf("%.2f", sample_zero_threshold)
        if (any(drop)) {
            data <- data[!drop, , drop = FALSE]
            .inform(verbose, "Filtered samples: removed ", sum(drop),
                    " with zero proportion > ", thr)
        } else {
            .inform(verbose, "Filtered samples: removed 0 (threshold ",
                    thr, ")")
        }
    } else {
        .inform(verbose, "Filtered samples: none (no sample_zero_threshold ",
                "provided)")
    }
    data
}

# Sets missing values to zero and negative values to zero (with a warning),
# then drops samples whose values are all zero.
.clean_abundance <- function(data, verbose) {
    data[is.na(data)] <- 0
    if (any(!is.finite(data))) {
        stop("The abundance values must be finite.", call. = FALSE)
    }
    num_neg <- sum(data < 0)
    if (num_neg > 0L) {
        warning("The dataset contains ", num_neg, " negative values. They ",
                "have been converted to zeros, but please double-check that ",
                "this preprocessing step is appropriate for your data.",
                call. = FALSE)
        data[data < 0] <- 0
    }
    all_zero <- rowSums(data != 0) == 0L
    if (any(all_zero)) {
        data <- data[!all_zero, , drop = FALSE]
        .inform(verbose, "Removed ", sum(all_zero), " all-zero samples after ",
                "cleaning.")
    }
    data
}

# Indicator columns of a categorical covariate, first level dropped. Factor
# levels are kept as given; other values are sorted in byte order.
.dummy_columns <- function(v, key) {
    if (anyNA(v)) {
        stop("The categorical covariate '", key, "' has missing values.",
             call. = FALSE)
    }
    if (is.factor(v)) {
        levels_v <- levels(v)
        v <- as.character(v)
    } else {
        levels_v <- as.character(sort(unique(v), method = "radix"))
        v <- as.character(v)
    }
    kept <- levels_v[-1L]
    out <- vapply(kept, function(l) as.numeric(v == l), numeric(length(v)))
    out <- matrix(out, nrow = length(v), ncol = length(kept))
    colnames(out) <- .make_valid_column_name(paste0(key, "_", kept))
    out
}

# Covariate design matrix (samples x covariates): continuous covariates
# first, then the indicators of the categorical covariates. NULL without
# covariates.
.covariate_design <- function(meta, cont_keys, cat_keys) {
    if (is.null(meta) || (is.null(cont_keys) && is.null(cat_keys))) {
        return(NULL)
    }
    parts <- list()
    for (key in cont_keys) {
        v <- as.numeric(meta[[key]])
        if (anyNA(v)) {
            stop("The continuous covariate '", key, "' has missing values.",
                 call. = FALSE)
        }
        parts[[key]] <- matrix(v, ncol = 1L, dimnames = list(NULL, key))
    }
    for (key in cat_keys) {
        parts[[key]] <- .dummy_columns(meta[[key]], key)
    }
    design <- do.call(cbind, unname(parts))
    if (anyDuplicated(colnames(design))) {
        stop("Covariate columns have duplicated names after encoding: ",
             .head_list(unique(colnames(design)[duplicated(
                 colnames(design))])), ".", call. = FALSE)
    }
    rownames(design) <- rownames(meta)
    design
}
