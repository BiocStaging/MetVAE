# Argument checks and small helpers shared across the package.

# Stops unless x is a single logical value.
.check_flag <- function(x, name) {
    if (!is.logical(x) || length(x) != 1L || is.na(x)) {
        stop("'", name, "' must be TRUE or FALSE.", call. = FALSE)
    }
    invisible(x)
}

# Stops unless x is a single whole number >= lower. Returns it as an integer.
.check_count <- function(x, name, lower = 1L) {
    ok <- is.numeric(x) && length(x) == 1L && !is.na(x) &&
        is.finite(x) && x == round(x) && x >= lower
    if (!ok) {
        stop("'", name, "' must be a single whole number >= ", lower, ".",
             call. = FALSE)
    }
    as.integer(x)
}

# Stops unless x is a single finite number in [lower, upper].
.check_number <- function(x, name, lower = -Inf, upper = Inf) {
    ok <- is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x) &&
        x >= lower && x <= upper
    if (!ok) {
        stop("'", name, "' must be a single number in [", lower, ", ",
             upper, "].", call. = FALSE)
    }
    as.numeric(x)
}

# Stops unless x is NULL or a single number in [0, 1].
.check_proportion_or_null <- function(x, name) {
    if (is.null(x)) {
        return(NULL)
    }
    .check_number(x, name, lower = 0, upper = 1)
}

# Stops unless x is NULL or a character vector of distinct names.
.check_keys <- function(x, name) {
    if (is.null(x)) {
        return(NULL)
    }
    if (!is.character(x) || length(x) == 0L || anyNA(x) ||
        anyDuplicated(x)) {
        stop("'", name, "' must be NULL or a character vector of distinct ",
             "column names.", call. = FALSE)
    }
    x
}

# Message gated by a verbosity flag.
.inform <- function(verbose, ...) {
    if (isTRUE(verbose)) {
        message(...)
    }
    invisible(NULL)
}

# Lists at most 'k' elements of x for an error message.
.head_list <- function(x, k = 5L) {
    shown <- paste(utils::head(x, k), collapse = ", ")
    if (length(x) > k) {
        shown <- paste0(shown, ", ... (", length(x), " in total)")
    }
    shown
}

# Column name with spaces and punctuation replaced, as in the Python package.
.make_valid_column_name <- function(name) {
    name <- as.character(name)
    name <- gsub("[ .(),/\\\\-]", "_", name)
    name <- gsub("&", "and", name, fixed = TRUE)
    gsub("[^A-Za-z0-9_]", "", name)
}
