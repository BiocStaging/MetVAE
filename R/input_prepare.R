# Entry point of the pure-R input stage.

# Validated abundance matrix (samples x features), covariate design, names,
# zero counts, and a record of the filtering, in the order of the Python
# package: covariate scaling on the full metadata, sample alignment, zero
# filters, cleaning, then the design.
.metvae_input <- function(x, meta = NULL, assay_name = NULL,
                          features_as_rows = FALSE,
                          continuous_covariate_keys = NULL,
                          categorical_covariate_keys = NULL,
                          feature_zero_threshold = 0.3,
                          sample_zero_threshold = NULL, verbose = TRUE) {
    .check_flag(features_as_rows, "features_as_rows")
    .check_flag(verbose, "verbose")
    cont_keys <- .check_keys(continuous_covariate_keys,
                             "continuous_covariate_keys")
    cat_keys <- .check_keys(categorical_covariate_keys,
                            "categorical_covariate_keys")
    feature_zero_threshold <- .check_proportion_or_null(
        feature_zero_threshold, "feature_zero_threshold")
    sample_zero_threshold <- .check_proportion_or_null(
        sample_zero_threshold, "sample_zero_threshold")

    ab <- .as_abundance(x, meta, assay_name, features_as_rows)
    .check_meta(ab$meta, cont_keys, cat_keys)
    meta <- .scale_continuous(ab$meta, cont_keys)
    data <- ab$data
    meta <- .align_meta(data, meta)
    n0 <- nrow(data)
    d0 <- ncol(data)
    .inform(verbose, "Start: samples=", n0, ", features=", d0)

    data <- .zero_filter(data, feature_zero_threshold, sample_zero_threshold,
                         verbose)
    .inform(verbose, "After zero filtering: samples=", nrow(data),
            ", features=", ncol(data))
    data <- .clean_abundance(data, verbose)
    .inform(verbose, "Post-cleaning (convert negative values to zeros and ",
            "drop all-zero samples): samples=", nrow(data), ", features=",
            ncol(data))
    .check_dimensions(data)

    design <- .covariate_design(.align_meta(data, meta), cont_keys, cat_keys)
    list(data = data, design = design,
         sample_names = rownames(data), feature_names = colnames(data),
         confound_names = colnames(design),
         num_zero = colSums(data == 0),
         preprocess = list(n_input = n0, d_input = d0,
                           n_kept = nrow(data), d_kept = ncol(data),
                           feature_zero_threshold = feature_zero_threshold,
                           sample_zero_threshold = sample_zero_threshold))
}

# Stops unless at least 2 samples and 3 features remain.
.check_dimensions <- function(data) {
    if (nrow(data) < 2L) {
        stop("At least 2 samples are required after filtering; ",
             nrow(data), " remain.", call. = FALSE)
    }
    if (ncol(data) < 3L) {
        stop("At least 3 features are required after filtering; ",
             ncol(data), " remain. Raise 'feature_zero_threshold' or set ",
             "it to NULL.", call. = FALSE)
    }
    invisible(NULL)
}
