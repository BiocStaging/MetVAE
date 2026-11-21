# Conversions between R objects and torch tensors, and torch helpers.
# Scalars always appear on the right of a tensor operation so that the result
# keeps the tensor's dtype.

# float64 tensor (or 'dtype') from an R vector or matrix; cloned so that the
# tensor owns its memory.
.as_t <- function(x, dtype = NULL, device = "cpu") {
    if (is.null(dtype)) {
        dtype <- torch::torch_float64()
    }
    torch::torch_tensor(x, dtype = dtype, device = device)$clone()
}

# R array from a tensor.
.as_r <- function(t) {
    as.array(t$detach()$cpu())
}

# R numeric scalar from a 0-dim or one-element tensor.
.item <- function(t) {
    t$item()
}

# Standard normal distribution function and its logarithm.
.ndtr <- function(x) {
    utils::getFromNamespace("torch_special_ndtr", "torch")(x)
}
.log_ndtr <- function(x) {
    utils::getFromNamespace("torch_special_log_ndtr", "torch")(x)
}

# 1 - x and a / x computed as in Python's reflected operators.
.rsub <- function(x, a) {
    x$neg()$add(a)
}
.rdiv <- function(a, x) {
    x$reciprocal()$mul(a)
}

# Evaluates 'expr' with float64 as the default torch dtype.
.with_float64 <- function(expr) {
    previous <- torch::torch_get_default_dtype()
    torch::torch_set_default_dtype(torch::torch_float64())
    on.exit(torch::torch_set_default_dtype(previous), add = TRUE)
    force(expr)
}
