# MetVAE

MetVAE is available for R, for Python, and through a graphical interface.

| Platform | Install | Source |
| --- | --- | --- |
| R (this package) | `BiocManager::install("MATRIX-Research-Lab/MetVAE")` | [GitHub](https://github.com/MATRIX-Research-Lab/MetVAE) |
| Python | `pip install metvae` | [GitHub](https://github.com/MATRIX-Research-Lab/MetVAE-PyPI), [PyPI](https://pypi.org/project/metvae/) |
| Graphical interface | See its README | [GNPS_Local](https://github.com/Some-Earth-2514/GNPS_Local) |

MetVAE builds a correlation network from untargeted metabolomics data. The
input is a table of metabolite abundances, with one row per sample and one
column per metabolite, and an optional table of sample information such as
batch, age, or diet. Zeros in the abundance table are treated as values below
the detection limit, not as true absences, and they are filled in by a
variational autoencoder. The effects of the sample covariates you name are
removed before the correlations are estimated. The output is a sparse
correlation matrix: most metabolite pairs are set to exactly zero, and the
remaining pairs form the edges of a network.

Two methods decide which pairs to keep: Fisher z-tests with a multiple-testing
correction (`sparse_by_p()`), and the sparse estimation of correlation (SEC)
estimator with a penalty chosen by cross-validation (`sparse_by_sec()`).

## Table of contents

- [Installation](#installation)
- [Preparing your data](#preparing-your-data)
- [A complete example](#a-complete-example)
- [Arguments that matter](#arguments-that-matter)
- [Reading the results](#reading-the-results)
- [Reproducibility](#reproducibility)
- [Run time and memory](#run-time-and-memory)
- [Troubleshooting](#troubleshooting)
- [Getting help](#getting-help)
- [Citation](#citation)
- [License](#license)

## Installation

MetVAE requires R 4.5 or later. Install it from GitHub until it is available
from Bioconductor:

```r
install.packages("BiocManager")
BiocManager::install("MATRIX-Research-Lab/MetVAE")
```

MetVAE fits its model with the R package `torch`, which needs the LibTorch
library. Download it once, after installing MetVAE:

```r
torch::install_torch()
```

The download is several hundred megabytes. Check the installation with:

```r
torch::torch_is_installed()   # should print TRUE
library(MetVAE)
```

The model runs on the CPU.

## Preparing your data

### File layout

The abundance file has samples as rows and metabolites as columns. The first
column holds the sample names.

```
sample_id,M001,M002,M003,M004,M005
S01,1204.5,0,88.2,3310.0,15.7
S02,980.1,42.6,91.0,2870.5,0
S03,1102.8,39.4,0,3050.2,14.9
S04,1355.0,45.1,95.6,3402.7,16.2
```

The metadata file has one row per sample. Its first column holds the same
sample names, in any order.

```
sample_id,age,sex,batch
S01,54,F,B1
S02,61,M,B1
S03,48,F,B2
S04,57,M,B2
```

### Rules for the abundance table

| Rule | Reason |
| --- | --- |
| Keep zeros as 0; do not fill them in yourself. | MetVAE models zeros as values below the detection limit. |
| No negative values. | Negative values are set to zero with a warning. |
| Missing values (empty cells, NA) are allowed. | They are treated as zeros. |
| Do not log-transform or normalize. | MetVAE applies the centered log-ratio transformation itself. |
| Every sample must appear in the metadata. | Covariates are matched to samples by name. |

### Loading the tables

```r
data <- read.csv("abundance.csv", row.names = 1, check.names = FALSE)
meta <- read.csv("metadata.csv", row.names = 1)
all(rownames(data) %in% rownames(meta))   # should print TRUE
```

A `SummarizedExperiment` with metabolites in rows and samples in columns also
works: pass it as the first argument, and MetVAE reads the covariates from
`colData()`.

## A complete example

This example simulates a data set, fits the model, and runs both
sparsification methods. It takes about 10 seconds.

```r
library(MetVAE)

set.seed(7)
meta <- data.frame(age = rnorm(100, mean = 50, sd = 8),
                   batch = sample(c("A", "B"), 100, replace = TRUE),
                   row.names = paste0("s", 0:99))
sim <- metvae_sim_data(n = 100, d = 50, cor_pairs = 10, mu = 10:14,
                       x = meta, cont_list = "age", cat_list = "batch",
                       seed = 7)

# Set the lowest 30 percent of each metabolite to zero.
data <- sim$y
thresholds <- apply(data, 2, quantile, probs = 0.3)
data[sweep(data, 2, thresholds, "<")] <- 0

fit <- metvae(data, meta = meta,
              continuous_covariate_keys = "age",
              categorical_covariate_keys = "batch",
              learning_rate = 1e-2)
fit <- get_corr(fit)

res_p <- sparse_by_p(fit)
res_sec <- sparse_by_sec(fit)
res_sec$best_rho
```

`metvae()` prints its preprocessing steps:

```
Start: samples=100, features=50
Filtered features: removed 0 (threshold 0.30)
Filtered samples: none (no sample_zero_threshold provided)
After zero filtering: samples=100, features=50
Post-cleaning (convert negative values to zeros and drop all-zero samples): samples=100, features=50
```

Note that `get_corr()` returns the fit with the correlations added, so its
result is assigned back to `fit`.

The vignette walks through the same example with figures:

```r
vignette("MetVAE", package = "MetVAE")
```

## Arguments that matter

Every argument has a default that works for a first analysis.

### `metvae()`

| Argument | What it controls | Default | When to change it |
| --- | --- | --- | --- |
| `continuous_covariate_keys` | Numeric metadata columns to adjust for | none | Adjust for variables such as age or BMI. |
| `categorical_covariate_keys` | Categorical metadata columns to adjust for | none | Adjust for batch, diet, or sex. |
| `feature_zero_threshold` | Metabolites with a larger proportion of zeros are removed | 0.3 | Raise it to keep sparser metabolites; NULL keeps all. |
| `sample_zero_threshold` | Samples with a larger proportion of zeros are removed | none | Set it to drop poor-quality samples. |
| `latent_dim` | Number of latent variables | smaller of samples and metabolites | Lower it to speed up large data sets. |
| `max_epochs` | Training passes over the data | 1000 | Lower it for a quick look. |
| `learning_rate` | Step size of the optimizer | 0.001 | 0.01 trains faster on small data sets. |
| `batch_size` | Samples per training step | 128 | Rarely. |
| `seed` | Random seed | 0 | Change it to check stability. |

### `get_corr()`

| Argument | What it controls | Default |
| --- | --- | --- |
| `num_sim` | Number of imputations averaged | 100 |
| `threshold` | Correlations below this value in each imputation are set to zero | 0.2 |
| `seed` | Seed of the imputations | seed of the fit |

### `sparse_by_p()` and `sparse_by_sec()`

| Argument | What it controls | Default |
| --- | --- | --- |
| `p_adj_method` | Multiple-testing correction of `sparse_by_p()` | "fdr_bh" (Benjamini-Hochberg) |
| `cutoff` | Correlations with a larger adjusted p-value are set to zero | 0.05 |
| `rho` | Penalty of `sparse_by_sec()`; NULL chooses it by cross-validation | NULL |
| `c_grid` | Candidate values of c in `rho = c * sqrt(log(p) / n)` | 1, ..., 10 |
| `threshold` | Final cutoff of `sparse_by_sec()` | 0.1 |

### How to choose

- `feature_zero_threshold`: metabolites with many zeros carry little
  information about correlation. The default 0.3 is a reasonable start.
- `num_sim`: 100 imputations are enough for stable estimates; fewer give a
  quick preview.
- `cutoff`: use a smaller value such as 0.001 for data sets with thousands of
  metabolites, where 0.05 keeps many weak edges.
- `rho`: let cross-validation choose it. If you see a warning that the best c
  is at the edge of `c_grid`, extend the grid in that direction.

## Reading the results

`sparse_by_p()` returns a list:

| Element | Content |
| --- | --- |
| `estimate` | Correlation matrix from `get_corr()` |
| `p_value` | p-values of the Fisher z-tests |
| `q_value` | Adjusted p-values |
| `sparse_estimate` | Correlations kept after the correction; all others are 0 |

`sparse_by_sec()` returns `estimate`, `sparse_estimate`, `best_rho` (the
penalty used), and `scores_by_rho` (the cross-validation error of every
penalty).

A zero in `sparse_estimate` means that the data give no evidence of an
association between the two metabolites. The correlation matrices are stored
as sparse matrices; `as.matrix()` converts them.

### Saving the results

```r
write.csv(as.matrix(res_p$sparse_estimate), "sparse_correlation.csv")
saveRDS(fit, "metvae_fit.rds")   # reload with readRDS()
```

### Networks

```r
graphs <- export_graphml(res_p, cutoffs = 0.7, output_dir = "results")
```

This writes `results/correlation_graph_cutoff0.7.graphml`. To open it in
Cytoscape:

1. Choose File > Import > Network from File.
2. Select the `.graphml` file.
3. Map the edge attribute `EdgeScore` to edge color or width.

An edge list for a spreadsheet:

```r
edges <- igraph::as_data_frame(graphs[[1]], what = "edges")
write.csv(edges[, c("from", "to", "correlation")], "edges.csv",
          row.names = FALSE)
```

### Other outputs

| Function | Returns |
| --- | --- |
| `impute_zeros(fit, seed)` | One completed data set on the centered log-ratio scale |
| `confound_coef(fit)` | Covariate coefficients of each metabolite |
| `confound_es(fit)` | Covariate effect of each sample and metabolite |
| `clr_loading(fit)` | Decoder weights (metabolites x latent variables) |
| `cooccurrence(fit)` | Squared distances between the loadings of metabolite pairs |

## Reproducibility

On one machine with one version of MetVAE and of the package `torch`, the same
seeds give identical results. The seed of `metvae()` controls the model
initialization and the order of the training batches, and is also the default
seed of `get_corr()`.

Results can differ in the last digits across machines, linear algebra
libraries, `torch` versions, or numbers of threads. MetVAE does not change the
number of threads; set it at the start of a session with
`torch::torch_set_num_threads()` if needed. Such differences matter only for
correlations that lie almost exactly at a cutoff.

For the same input data, the R and Python implementations draw the same
random numbers. On the quickstart data of the Python package, they give the
same network.

## Run time and memory

Measured on an Apple M2 Max laptop (12 cores, 64 GB) on the CPU:

| Data set | Step | Time |
| --- | --- | --- |
| 100 samples x 50 metabolites | `metvae()`, 1000 epochs | 4.5 s |
| | `get_corr()`, 100 imputations | 0.3 s |
| | `sparse_by_sec()`, cross-validation | 0.8 s |

`get_corr()` holds three dense metabolite x metabolite matrices of 8-byte
numbers, about 1.3 GB for 7,000 metabolites. `sparse_by_p()` with the default
`pq = "dense"` holds several more; `pq = "edges"` reduces memory for large
data sets.

## Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `requires the LibTorch library of the package 'torch'` | LibTorch was not downloaded. | Run `torch::install_torch()` once. |
| `The following sample names are missing in the sample metadata` | Some samples of the abundance table have no metadata row. | Add them, or keep only shared samples. |
| `The dataset contains N negative values` | Negative abundances, often from background subtraction or a log scale. | Supply raw, non-negative abundances. |
| `At least 3 features are required after filtering` | Too many metabolites were removed. | Raise `feature_zero_threshold` or set it to NULL. |
| `Sample size must be > 3 for Fisher's z-test` | Fewer than 4 samples. | Use `sparse_by_sec()`. |
| `No correlation estimates. Run fit <- get_corr(fit) first.` | `sparse_by_p()` or `sparse_by_sec()` was called before `get_corr()`. | Run `fit <- get_corr(fit)`. |
| `Best c = ... occurs at the UPPER edge of c_grid` | The penalty grid is too narrow. | Use a wider `c_grid`, for example `1:20`. |
| R stops responding or runs out of memory | Very many metabolites. | Raise `feature_zero_threshold`, or use `pq = "edges"` in `sparse_by_p()`. |

## Getting help

Report bugs and ask questions on the
[GitHub issue page](https://github.com/MATRIX-Research-Lab/MetVAE/issues).
Include the code you ran, the full error message, and the output of
`sessionInfo()`.

## Citation

If you use MetVAE, please cite:

Lin H, Zhang L, Lotfi A, Jarmusch A, Lee I, Kim A, Morton JT, Aksenov A
(2026). Protocol for constructing correlation-based molecular networks from
large-scale untargeted metabolomics data. bioRxiv.
https://doi.org/10.1101/2025.04.26.649581

`citation("MetVAE")` prints the BibTeX entry.

The SEC implementation is adapted from the MATLAB reference code released by
Leng's group at the University of Warwick, which accompanies Cui, Leng and Sun
(2016), Computational Statistics and Data Analysis 93, 390-403,
https://doi.org/10.1016/j.csda.2014.10.001.

## License

MIT. See the `LICENSE` file.
