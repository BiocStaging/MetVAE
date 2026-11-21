"""Reference outputs of the Python package metvae for the R port.

Usage: python make_pyref.py <output_dir> [quickstart|sim3]

Each array is written as raw little-endian bytes (float64 or int64, row
major) and listed in manifest.csv with its shape and type; character vectors
are written one value per line. The R tests read the files with readBin().
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

import metvae
import metvae.impute_missing as im
import metvae.model as mm
from metvae.model import MetVAE
from metvae.sim import sim_data

P_ADJ_METHODS = ["bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg",
                 "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky"]


class Writer:
    """Writes arrays and character vectors into one directory."""

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.rows = []

    def array(self, name, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.dtype.kind in "iub":
            arr, kind = x.astype("<i8"), "int64"
        else:
            arr, kind = x.astype("<f8"), "float64"
        arr.tofile(os.path.join(self.path, name + ".bin"))
        dims = "x".join(str(s) for s in arr.shape) if arr.ndim > 0 else "1"
        self.rows.append({"name": name, "type": kind, "dims": dims})

    def strings(self, name, values):
        with open(os.path.join(self.path, name + ".txt"), "w") as f:
            for v in values:
                f.write(str(v) + "\n")
        self.rows.append({"name": name, "type": "character",
                          "dims": str(len(values))})

    def close(self, info):
        with open(os.path.join(self.path, "manifest.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "type", "dims"])
            w.writeheader()
            w.writerows(self.rows)
        with open(os.path.join(self.path, "info.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            for k, v in info.items():
                w.writerow([k, repr(v) if isinstance(v, float) else v])


class Recorder:
    """Records the arguments and results of internal metvae functions."""

    def __init__(self):
        self.calls = {}

    def wrap(self, module, name):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            self.calls.setdefault(name, []).append((args, kwargs, out))
            return out

        setattr(module, name, wrapper)


def quickstart_data():
    np.random.seed(7)
    n, d = 100, 50
    ids = [f"sample_{i:02d}" for i in range(1, n + 1)]
    meta = pd.DataFrame({
        "age": np.random.normal(loc=50.0, scale=8.0, size=n),
        "batch": np.random.choice(["A", "B"], size=n)}, index=ids)
    sim = sim_data(n=n, d=d, cor_pairs=10, mu=[10, 11, 12, 13, 14], x=meta,
                   cont_list=["age"], cat_list=["batch"])
    raw = sim["y"].copy()
    data = raw.mask(raw.lt(raw.quantile(0.30, axis=0)), 0.0)
    data.index = ids
    return data, meta, sim["cor_matrix"], dict(
        continuous_covariate_keys=["age"], categorical_covariate_keys=["batch"]
    ), dict(learning_rate=1e-2), dict(rho=2.2)


def sim3_data():
    n, d, zero_prop, seed = 100, 500, 0.3, 42
    np.random.seed(seed)
    smd = pd.DataFrame({"x1": np.random.randn(n),
                        "x2": np.random.choice(["a", "b"], size=n,
                                               replace=True)},
                       index=[f"s{i}" for i in range(n)])
    sim = sim_data(n=n, d=d, cor_pairs=int(0.2 * d), mu=list(range(10, 15)),
                   da_prop=0.1, x=smd, cont_list=["x1"], cat_list=["x2"])
    y = sim["y"]
    log_data = (np.log(y) + np.log(np.random.uniform(1e-3, 1e-1, size=n))
                [:, None] + np.log(np.random.uniform(1e-1, 1, size=d))
                .reshape(1, d))
    data = np.exp(log_data)
    th = np.quantile(data, zero_prop, axis=0)
    data = pd.DataFrame(np.where(data < th, 0, data), index=y.index,
                        columns=y.columns)
    return data, smd, sim["cor_matrix"], dict(
        continuous_covariate_keys=["x1"], categorical_covariate_keys=["x2"],
        latent_dim=min(n, d)
    ), dict(batch_size=100, learning_rate=1e-2), dict(rho=2.2)


def write_state(w, prefix, model):
    for k, v in model.model.state_dict().items():
        w.array(prefix + k, v)


def main(out_dir, workload):
    torch.set_num_threads(8)
    data, meta, true_cor, init_kw, train_kw, sec_kw = {
        "quickstart": quickstart_data, "sim3": sim3_data}[workload]()
    w = Writer(out_dir)
    w.array("data", data.to_numpy())
    w.strings("sample_names", data.index)
    w.strings("feature_names", data.columns)
    for col in meta.columns:
        if meta[col].dtype.kind == "f":
            w.array("meta_" + col, meta[col].to_numpy())
        else:
            w.strings("meta_" + col, meta[col].tolist())
    w.strings("meta_names", meta.index)
    w.array("true_cor", true_cor)

    rec = Recorder()
    rec.wrap(mm, "_ols_estimate")
    rec.wrap(mm, "_fit_censored_normal")
    rec.wrap(im, "_tobit_em_warmstart")
    rec.wrap(mm, "_epoch_index_batches")

    model = MetVAE(data=data, meta=meta, **init_kw)
    ols_args, _, ols_out = rec.calls["_ols_estimate"][0]
    w.array("clr_raw", ols_args[0])
    w.array("design", model.meta)
    w.array("ols", ols_out)
    _, em_kw, em_out = rec.calls["_tobit_em_warmstart"][0]
    w.array("clr_th", em_kw["th"])
    w.array("em_beta", em_out[2])
    w.array("em_log_sigma", em_out[3])
    w.array("clr_params", rec.calls["_fit_censored_normal"][0][2])
    for k in ["clr_data", "shift", "num_zero", "clr_mean", "clr_sd",
              "clr_coef"]:
        w.array(k, getattr(model, k))
    write_state(w, "init.", model)

    gen = torch.Generator(device="cpu").manual_seed(123)
    w.array("random_initial_123", mm._random_initial(
        model.clr_data, model.sample_dim, model.num_zero, model.clr_mean,
        model.clr_sd, generator=gen))

    model.train(**train_kw)
    calls = [c for c in rec.calls["_epoch_index_batches"]
             if c[0][0] == model.sample_dim]
    perms = [torch.cat([b[0] for b in c[2]]) for c in calls[:3]]
    w.array("perm_epochs_1_3", torch.stack(perms))
    w.array("train_loss", np.asarray(model.train_loss))
    write_state(w, "final.", model)

    gen = torch.Generator(device="cpu").manual_seed(model.base_seed)
    il0 = model.impute_zeros(generator=gen) + model.shift
    w.array("impute_log_sim0", il0)
    w.array("corr_sim0", mm._compute_correlation_dense(torch.exp(il0)))

    out = model.get_corr(num_sim=100, workers=1)
    w.array("impute_log", out["impute_log_data"])
    w.array("estimate", out["estimate"].to_dense())

    res_p = model.sparse_by_p()
    w.array("p_value", res_p["p_value"].to_numpy())
    w.array("q_value", res_p["q_value"].to_numpy())
    w.array("p_sparse", res_p["sparse_estimate"].to_numpy())
    p = res_p["p_value"].to_numpy()
    pvec = p[np.tril_indices_from(p, k=-1)]
    w.array("pvec", pvec)
    w.array("qvec_all", np.stack(
        [multipletests(pvec, method=m)[1] for m in P_ADJ_METHODS]))
    w.strings("p_adj_methods", P_ADJ_METHODS)

    res_f = model.sparse_by_sec(**sec_kw)
    w.array("sec_fixed_sparse", res_f["sparse_estimate"].to_numpy())
    res_cv = model.sparse_by_sec(workers=1)
    w.array("sec_cv_sparse", res_cv["sparse_estimate"].to_numpy())
    w.array("sec_cv_scores", res_cv["scores_by_rho"].to_numpy(dtype=float))
    w.array("sec_cv_best_rho", np.array([res_cv["best_rho"]]))

    w.array("cooccurrence", model.cooccurrence().to_numpy())
    w.close({"workload": workload, "metvae": metvae.__version__,
             "torch": torch.__version__, "numpy": np.__version__,
             "threads": torch.get_num_threads(),
             "best_rho": float(res_cv["best_rho"])})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "quickstart")
