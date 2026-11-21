# Availability of the package torch and its LibTorch library.

# TRUE when the package torch can be loaded and its LibTorch library is
# installed.
.metvae_torch_available <- function() {
    requireNamespace("torch", quietly = TRUE) &&
        isTRUE(tryCatch(torch::torch_is_installed(),
                        error = function(e) FALSE))
}

# Stops with the installation commands when torch or its LibTorch library is
# absent. 'what' names the calling function in the message.
.metvae_require_torch <- function(what = "MetVAE") {
    if (!requireNamespace("torch", quietly = TRUE)) {
        stop(what, " requires the package 'torch', which is not installed. ",
             "Install it with install.packages(\"torch\") and then run ",
             "torch::install_torch() once to download the LibTorch library.",
             call. = FALSE)
    }
    installed <- tryCatch(torch::torch_is_installed(),
                          error = function(e) FALSE)
    if (!isTRUE(installed)) {
        stop(what, " requires the LibTorch library of the package 'torch', ",
             "which is not installed. Run torch::install_torch() once.",
             call. = FALSE)
    }
    invisible(TRUE)
}
