# Reader for the reference outputs of the Python package written by
# inst/scripts/make_pyref.py: raw little-endian float64 or int64 arrays in row
# major order, and character vectors with one value per line.

# Directory of a reference workload: METVAE_PYREF_DIR/<workload> when the
# variable is set (all stage outputs), otherwise the subset shipped in
# tests/testthat/fixtures/python/<workload>, or NULL.
pyref_dir <- function(workload = "quickstart") {
    root <- Sys.getenv("METVAE_PYREF_DIR", "")
    if (nzchar(root) && dir.exists(file.path(root, workload))) {
        return(file.path(root, workload))
    }
    shipped <- test_path("fixtures", "python", workload)
    if (dir.exists(shipped)) shipped else NULL
}

# TRUE when every array in 'names' is available for the workload.
has_pyref <- function(names, workload = "quickstart") {
    dir <- pyref_dir(workload)
    if (is.null(dir)) {
        return(FALSE)
    }
    man <- utils::read.csv(file.path(dir, "manifest.csv"),
                           stringsAsFactors = FALSE)
    all(names %in% man$name)
}

skip_if_no_pyref <- function(names, workload = "quickstart") {
    skip_if_not(has_pyref(names, workload),
                "Python reference outputs are not available")
}

# Array 'name' of a reference workload as an R vector, matrix, or array.
read_pyref <- function(name, workload = "quickstart") {
    dir <- pyref_dir(workload)
    man <- utils::read.csv(file.path(dir, "manifest.csv"),
                           stringsAsFactors = FALSE)
    row <- man[man$name == name, ]
    if (nrow(row) != 1L) {
        stop("No reference array named ", name)
    }
    if (row$type == "character") {
        return(readLines(file.path(dir, paste0(name, ".txt"))))
    }
    dims <- as.integer(strsplit(row$dims, "x", fixed = TRUE)[[1L]])
    con <- file(file.path(dir, paste0(name, ".bin")), "rb")
    on.exit(close(con))
    n <- prod(dims)
    if (row$type == "int64") {
        words <- readBin(con, "integer", n = 2L * n, size = 4L,
                         endian = "little")
        v <- words[seq(1L, 2L * n, by = 2L)]
    } else {
        v <- readBin(con, "double", n = n, size = 8L, endian = "little")
    }
    if (length(dims) <= 1L) {
        return(v)
    }
    if (length(dims) == 2L) {
        return(matrix(v, nrow = dims[1L], ncol = dims[2L], byrow = TRUE))
    }
    aperm(array(v, dim = rev(dims)), rev(seq_along(dims)))
}

# Abundance matrix and metadata of a reference workload, as R objects.
pyref_inputs <- function(workload = "quickstart") {
    data <- read_pyref("data", workload)
    dimnames(data) <- list(read_pyref("sample_names", workload),
                           read_pyref("feature_names", workload))
    man <- utils::read.csv(file.path(pyref_dir(workload), "manifest.csv"),
                           stringsAsFactors = FALSE)
    cols <- setdiff(sub("^meta_", "", grep("^meta_", man$name, value = TRUE)),
                    "names")
    meta <- data.frame(row.names = read_pyref("meta_names", workload))
    for (col in cols) {
        meta[[col]] <- read_pyref(paste0("meta_", col), workload)
    }
    list(data = data, meta = meta)
}

# Maximum absolute difference, ignoring positions where both are NaN.
max_abs_diff <- function(a, b) {
    a <- as.numeric(a)
    b <- as.numeric(b)
    both_nan <- is.nan(a) & is.nan(b)
    max(abs(a[!both_nan] - b[!both_nan]), 0)
}
