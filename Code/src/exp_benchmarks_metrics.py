# File: Code/exp_benchmarks_metrics.py
# Purpose: Unified benchmarking of OLS / Ridge (closed-form) and GD variants
#          with extra metrics: execution time, iterations to convergence,
#          FLOPs estimate, memory footprint, numerical stability, bias-variance.
#
# How to run (examples):
#   python Code/exp_benchmarks_metrics.py
#   python Code/exp_benchmarks_metrics.py --n-points 1000 --max-degree 12 --methods ols ridge gd-vanilla gd-momentum gd-adam
#   python Code/exp_benchmarks_metrics.py --noise --bootstrap 50
#
# What you get:
#   - outputs/tables/benchmarks_<timestamp>.csv              (main grid, 1 row per degree×method)
#   - outputs/tables/bias_variance_<timestamp>.csv          (bias-variance per degree)
#   - outputs/logs/benchmarks_<timestamp>.log               (full log)
#   - clear terminal summary table
#
# Notes:
#   - Uses your ml_core functions: runge_function, polynomial_features_scaled,
#     OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, R2_score, MSE_Bias_Variance
#   - FLOPs & memory are rough engineering estimates (good enough for comparisons).
#   - "Converged" = (actual iterations < n_iter_max). You can tighten this by checking
#     ||Δθ|| < tol inside ml_core if you want a definitive flag returned.

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression, Ridge as SkRidge, Lasso as SkLasso


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# --- make local Code importable (this file lives in Code/) ---
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
OUT_DIR = ROOT_DIR / "../outputs/benchmarks"
TABLES_DIR = OUT_DIR / "tables"
LOGS_DIR = OUT_DIR / "logs"

# Add Code/ to sys.path so we can `import ml_core`
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ---- import your project helpers ----
from ml_core import (
    runge_function,
    polynomial_features_scaled,
    OLS_parameters,
    Ridge_parameters,
    Gradient_descent_advanced,
    MSE,
    R2_score,
    MSE_Bias_Variance,
)

# --------------------------
# Utilities
# --------------------------
def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

class Logger:
    """Very small tee-logger (stdout + file)."""
    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
    def write(self, msg: str):
        sys.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()
    def close(self):
        self.file.close()

def human(n: float) -> str:
    """Humanize large numbers."""
    if n is None or math.isnan(n) or math.isinf(n): return str(n)
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"

def bytes_to_mb(b: float) -> float:
    return b / (1024.0 * 1024.0)

def print_table(rows: List[Dict[str, object]], columns: List[str], max_rows: int | None = None):
    """Minimal, dependency-free table printing."""
    data = rows if max_rows is None else rows[:max_rows]
    col_widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in data)) for c in columns}
    sep = " | "
    header = sep.join(f"{c:{col_widths[c]}}" for c in columns)
    line = "-+-".join("-" * col_widths[c] for c in columns)
    print(header)
    print(line)
    for r in data:
        print(sep.join(f"{str(r.get(c, '')):{col_widths[c]}}" for c in columns))


def run_sklearn_method(method: str, Xtr, Xte, y_tr, y_te, lam: float):
    """
    Fit/predict with scikit-learn equivalents and return timings + metrics.
    Returns: (theta_like, metrics_dict)
    Note: returns a 'theta_like' vector to keep interface consistent, but it's not used further.
    """
    t0 = time.perf_counter()

    if method == "sklearn-ols":
        model = LinearRegression(fit_intercept=False)
    elif method == "sklearn-ridge":
        model = SkRidge(alpha=lam, fit_intercept=False)  # solver='auto'
    elif method == "sklearn-lasso":
        model = SkLasso(alpha=lam, fit_intercept=False, max_iter=10000)
    else:
        raise ValueError(f"Unknown sklearn method: {method}")

    model.fit(Xtr, y_tr)
    t1 = time.perf_counter()
    fit_ms = (t1 - t0) * 1000.0

    t2 = time.perf_counter()
    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    t3 = time.perf_counter()
    pred_ms = (t3 - t2) * 1000.0

    # metrics
    mse_tr = MSE(y_tr, yhat_tr)
    mse_te = MSE(y_te, yhat_te)
    r2_tr = R2_score(y_tr, yhat_tr)
    r2_te = R2_score(y_te, yhat_te)

    # iterations, if exposed
    iters = 0
    if hasattr(model, "n_iter_"):
        niter = model.n_iter_
        # sklearn can expose int or array; normalize
        if isinstance(niter, (list, tuple, np.ndarray)):
            iters = int(np.max(niter))
        else:
            iters = int(niter)

    # Fake theta to keep interfaces aligned (not used afterwards)
    theta_like = np.zeros(Xtr.shape[1])

    metrics = dict(
        fit_time_ms=fit_ms,
        predict_time_ms=pred_ms,
        mse_train=mse_tr,
        mse_test=mse_te,
        r2_train=r2_tr,
        r2_test=r2_te,
        iters=iters,
    )
    return theta_like, metrics

# --------------------------
# FLOPs & memory estimates
# --------------------------
def estimate_flops_closed_form(n: int, p: int, solver: str = "pinv") -> float:
    """
    Rough FLOPs for closed-form:
      - Build X^T X: ~2 n p^2
      - Build X^T y: ~2 n p
      - Invert/pinv: ~ c * p^3   (SVD-based pinv is higher than LU; use c ~ 4/3)
      - Multiply (XTX_inv @ X^T): ~2 p^2 n
      - Multiply ... @ y: ~2 n p
      Total ~ 4 n p^2 + 4 n p + c p^3
    """
    c = 4.0 / 3.0 if solver == "pinv" else 2.0 / 3.0
    return 4.0 * n * (p**2) + 4.0 * n * p + c * (p**3)

def estimate_flops_gd(n: int, p: int, iters: int, method: str, batch_size: int | None = None) -> float:
    """
    GD per iter full-batch: ~4 n p (X@θ + X^T r + updates)
    Mini-batch of size b: ~4 b p per iter.
    Optimizer vector ops (momentum/adam/etc.) add O(p) per iter -> negligible vs n p.
    """
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        per_iter = 4.0 * n * p
    else:
        per_iter = 4.0 * batch_size * p
    return per_iter * max(iters, 1)

def estimate_memory_bytes(method: str, n: int, p: int, batch_size: int | None = None) -> float:
    """
    Very rough upper bound on peak resident data for the algorithm (float64 ~ 8 bytes):
      - Common: X (n×p), y (n), θ (p)
      - Closed-form: plus X^T X (p×p), X^T y (p)
      - GD: plus grad/optimizer states (a few × p)
    """
    BYTES = 8.0  # float64
    X = n * p * BYTES
    y = n * BYTES
    theta = p * BYTES

    if method in ("ols", "ridge"):
        xtx = p * p * BYTES
        xty = p * BYTES
        return X + y + theta + xtx + xty
    else:
        # gd variants: θ, grad, maybe momentum/adagrad/rmsprop/adam states (~ up to 3–4 extra vectors)
        states = 4 * p * BYTES
        # batch doesn't change stored X unless you stream; we assume X resident.
        return X + y + theta + states

# --------------------------
# Data classes
# --------------------------
@dataclass
class BenchRow:
    degree: int
    method: str
    n_points: int
    n_train: int
    n_test: int
    p: int

    fit_time_ms: float
    predict_time_ms: float
    total_time_ms: float
    iters: int
    converged: bool

    mse_train: float
    r2_train: float
    mse_test: float
    r2_test: float

    flops_est: float
    mem_mb_est: float
    cond_XTX: float
    scaled_vs_unscaled_mse_delta: float  # stability proxy (test MSE scaled - unscaled)

# --------------------------
# Core benchmarking
# --------------------------
def build_poly_train_test(x: np.ndarray, y: np.ndarray, degree: int, test_size=0.33, rs: int = 314):
    """
    Split, then build scaled polynomial features using training stats.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=rs)
    # Compute scaling stats on train
    Xtr, means, stds = polynomial_features_scaled(x_tr, degree, intercept=True, return_stats=True)
    Xte = polynomial_features_scaled(x_te, degree, intercept=True, col_means=means, col_stds=stds, return_stats=False)
    return Xtr, Xte, y_tr, y_te

def stability_delta_unscaled(Xtr_scaled, Xte_scaled, y_tr, y_te, x_tr_raw, x_te_raw, degree: int) -> float:
    """
    Numerical stability proxy:
      - solve OLS with scaled polynomial features and with unscaled polynomial features,
      - compare test MSEs (scaled - unscaled). If unscaled blows up, return +inf.
    """
    try:
        # scaled
        theta_s = OLS_parameters(Xtr_scaled, y_tr)
        yhat_s = Xte_scaled @ theta_s
        mse_s = MSE(y_te, yhat_s)

        # unscaled
        # build raw poly without scaling, same split
        def raw_poly(x_raw, deg):
            X = np.zeros((len(x_raw), deg + 1))
            X[:, 0] = 1.0
            for i in range(1, deg + 1):
                X[:, i] = x_raw ** i
            return X

        Xtr_raw = raw_poly(x_tr_raw, degree)
        Xte_raw = raw_poly(x_te_raw, degree)
        theta_u = OLS_parameters(Xtr_raw, y_tr)
        yhat_u = Xte_raw @ theta_u
        mse_u = MSE(y_te, yhat_u)

        return mse_s - mse_u
    except Exception:
        return float("inf")

def run_one_method(
    method: str,
    degree: int,
    Xtr: np.ndarray,
    Xte: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    lam: float,
    gd_opts: dict,
) -> Tuple[BenchRow, np.ndarray]:
    """
    Fit a single method and collect metrics. Returns (bench_row, theta).
    """
    n_train, p = Xtr.shape
    n_test = Xte.shape[0]
    n_total = n_train + n_test

    # Condition number (on training X^T X)
    try:
        cond = np.linalg.cond(Xtr.T @ Xtr)
    except Exception:
        cond = float("inf")

    # --- Fit ---
    t0 = time.perf_counter()

    iters = 0
    converged = True

    # ===== NEW: sklearn methods =====
    if method in ("sklearn-ols", "sklearn-ridge", "sklearn-lasso"):
        theta, m = run_sklearn_method(method, Xtr, Xte, y_tr, y_te, lam=lam)
        fit_ms = m["fit_time_ms"]
        pred_ms = m["predict_time_ms"]
        total_ms = fit_ms + pred_ms
        mse_tr = m["mse_train"]
        mse_te = m["mse_test"]
        r2_tr = m["r2_train"]
        r2_te = m["r2_test"]
        iters = m["iters"]
        converged = True if iters == 0 else True  # if sklearn returns iters, it converged
        # FLOPs/mem: leave FLOPs NaN (solver-dependent), memory estimate similar to closed-form for ols/ridge; unknown for lasso
        if method in ("sklearn-ols", "sklearn-ridge"):
            flops = float("nan")
            mem_mb = bytes_to_mb(estimate_memory_bytes("ols", n_train, p))
        else:
            flops = float("nan")
            mem_mb = float("nan")

    # ===== Your implementations =====
    else:
        if method == "ols":
            theta = OLS_parameters(Xtr, y_tr)
        elif method == "ridge":
            theta = Ridge_parameters(Xtr, y_tr, lam=lam, intercept=True)
        elif method.startswith("gd-"):
            # Map gd-* to optimizer names in your Gradient_descent_advanced
            opt = method.replace("gd-", "")  # vanilla, momentum, adagrad, rmsprop, adam
            hist = Gradient_descent_advanced(
                Xtr, y_tr,
                Type=0,             # OLS loss
                lam=lam,
                lr=gd_opts["lr"],
                n_iter=gd_opts["n_iter"],
                tol=gd_opts["tol"],
                method=opt,
                beta=gd_opts["beta"],
                epsilon=gd_opts["epsilon"],
                batch_size=gd_opts["batch_size"],
                use_sgd=gd_opts["use_sgd"],
                theta_history=True
            )
            if hist is None or len(hist) == 0:
                theta = np.zeros(p)
                iters = 0
                converged = False
            else:
                theta = hist[-1]
                iters = len(hist)
                converged = iters < gd_opts["n_iter"]
        else:
            raise ValueError(f"Unknown method: {method}")

        t1 = time.perf_counter()
        fit_ms = (t1 - t0) * 1000.0

        # Predict
        t2 = time.perf_counter()
        yhat_tr = Xtr @ theta
        yhat_te = Xte @ theta
        t3 = time.perf_counter()
        pred_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        # Metrics
        mse_tr = MSE(y_tr, yhat_tr)
        mse_te = MSE(y_te, yhat_te)
        r2_tr = R2_score(y_tr, yhat_tr)
        r2_te = R2_score(y_te, yhat_te)

        # FLOPs & memory
        if method in ("ols", "ridge"):
            flops = estimate_flops_closed_form(n_train, p, solver="pinv")
            mem_mb = bytes_to_mb(estimate_memory_bytes(method, n_train, p))
        else:
            flops = estimate_flops_gd(n_train, p, iters=iters, method=method, batch_size=gd_opts["batch_size"] if gd_opts["use_sgd"] else None)
            mem_mb = bytes_to_mb(estimate_memory_bytes("gd", n_train, p, gd_opts["batch_size"] if gd_opts["use_sgd"] else None))

    row = BenchRow(
        degree=degree,
        method=method,
        n_points=n_total,
        n_train=n_train,
        n_test=n_test,
        p=p,

        fit_time_ms=fit_ms,
        predict_time_ms=pred_ms,
        total_time_ms=total_ms,
        iters=iters,
        converged=converged,

        mse_train=mse_tr,
        r2_train=r2_tr,
        mse_test=mse_te,
        r2_test=r2_te,

        flops_est=flops,
        mem_mb_est=mem_mb,
        cond_XTX=cond,
        scaled_vs_unscaled_mse_delta=0.0,  # filled by caller
    )
    return row, theta

def bias_variance_grid(
    x: np.ndarray,
    y: np.ndarray,
    degrees: List[int],
    n_bootstrap: int = 50,
    test_size: float = 0.33,
    rs: int = 314,
) -> List[Dict[str, float]]:
    """
    For each degree, bootstrap train sets, fit OLS closed form, predict on shared test set,
    and compute MSE, bias^2, variance on test split.
    """
    # Fixed test split for all bootstraps (to keep targets constant)
    x_tr_all, x_te, y_tr_all, y_te = train_test_split(x, y, test_size=test_size, random_state=rs)
    results = []
    for deg in degrees:
        # Build test design once (using training stats from each bootstrap, so we rebuild inside loop)
        preds = []
        # Bootstrap over the training indices
        n_tr = len(x_tr_all)
        idx = np.arange(n_tr)
        for _ in range(n_bootstrap):
            b_idx = resample(idx, replace=True, n_samples=n_tr, random_state=None)
            x_tr = x_tr_all[b_idx]
            y_tr = y_tr_all[b_idx]

            # scale from bootstrap-train stats
            Xtr, means, stds = polynomial_features_scaled(x_tr, deg, intercept=True, return_stats=True)
            Xte = polynomial_features_scaled(x_te, deg, intercept=True, col_means=means, col_stds=stds)

            theta = OLS_parameters(Xtr, y_tr)
            preds.append(Xte @ theta)

        P = np.vstack(preds)  # (B, n_test)
        mse, bias2, var = MSE_Bias_Variance(y_te, P)
        results.append({
            "degree": deg,
            "mse": float(mse),
            "bias2": float(bias2),
            "variance": float(var),
        })
    return results


def _ensure_figs_dir():
    figs_dir = OUT_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    return figs_dir

def _group_by_method_and_degree(rows: List[BenchRow]) -> Dict[str, Dict[int, BenchRow]]:
    """
    Returns: { method : { degree : BenchRow } }
    Assumes a single row per (method, degree).
    """
    grid = defaultdict(dict)
    for r in rows:
        grid[r.method][r.degree] = r
    return grid

def _sorted_xy_for_metric(grid: Dict[int, BenchRow], metric: str) -> Tuple[List[int], List[float]]:
    degs = sorted(grid.keys())
    ys = [getattr(grid[d], metric) for d in degs]
    return degs, ys



def make_plots(rows: List[BenchRow], bv_results: List[Dict[str, float]], stamp: str, logger: Logger | None = None):
    """
    Generate and save a set of plots that are useful to discuss in an article:
      1. Test R^2 vs degree (per method)
      2. Fit time vs degree (per method, log-y)
      3. Iterations vs degree (GD methods only)
      4. FLOPs estimate vs degree (per method, log-y)
      5. Memory estimate vs degree (per method)
      6. Condition number of X'X vs degree
      7. Scaled-vs-unscaled Test MSE delta (stability proxy) vs degree
      8. Bias–Variance decomposition vs degree
      9. (Bonus) Accuracy–Speed Pareto: R^2(test) vs Fit time (log-x) scatter
    """
    figs_dir = _ensure_figs_dir()
    saved = []

    grid = _group_by_method_and_degree(rows)

    # 1) R^2(test) vs degree
    plt.figure(figsize=(7, 4.5))
    for m, g in grid.items():
        degs, ys = _sorted_xy_for_metric(g, "r2_test")
        plt.plot(degs, ys, marker="o", label=m)
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"$R^2$ (test)")
    plt.title(r"Test $R^2$ vs Degree")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f1 = figs_dir / f"r2_test_vs_degree.png"
    plt.tight_layout(); plt.savefig(f1, dpi=160); plt.close()
    saved.append(f1)

    # 2) Fit time vs degree (log-y)
    plt.figure(figsize=(7, 4.5))
    for m, g in grid.items():
        degs, ys = _sorted_xy_for_metric(g, "fit_time_ms")
        plt.semilogy(degs, ys, marker="o", label=m)
    plt.xlabel("Polynomial degree")
    plt.ylabel("Fit time (ms, log)")
    plt.title("Fit Time vs Degree")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f2 = figs_dir / f"fit_time_vs_degree.png"
    plt.tight_layout(); plt.savefig(f2, dpi=160); plt.close()
    saved.append(f2)

    # 3) Iterations vs degree (only GD methods)
    plt.figure(figsize=(7, 4.5))
    plotted_any = False
    for m, g in grid.items():
        if not m.startswith("gd-"): 
            continue
        degs, ys = _sorted_xy_for_metric(g, "iters")
        plt.plot(degs, ys, marker="o", label=m)
        plotted_any = True
    if plotted_any:
        plt.xlabel("Polynomial degree")
        plt.ylabel("Iterations")
        plt.title("Iterations to Convergence (GD) vs Degree")
        plt.grid(True, alpha=0.3)
        plt.legend()
        f3 = figs_dir / f"gd_iterations_vs_degree.png"
        plt.tight_layout(); plt.savefig(f3, dpi=160); plt.close()
        saved.append(f3)
    else:
        plt.close()

    # 4) FLOPs estimate vs degree (log-y)
    plt.figure(figsize=(7, 4.5))
    for m, g in grid.items():
        degs, ys = _sorted_xy_for_metric(g, "flops_est")
        # protect against zeros
        ys = [y if (isinstance(y, (int, float)) and y > 0) else np.nan for y in ys]
        plt.semilogy(degs, ys, marker="o", label=m)
    plt.xlabel("Polynomial degree")
    plt.ylabel("FLOPs (log)")
    plt.title("FLOPs Estimate vs Degree")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f4 = figs_dir / f"flops_vs_degree.png"
    plt.tight_layout(); plt.savefig(f4, dpi=160); plt.close()
    saved.append(f4)

    # 5) Memory estimate vs degree
    plt.figure(figsize=(7, 4.5))
    for m, g in grid.items():
        degs, ys = _sorted_xy_for_metric(g, "mem_mb_est")
        plt.plot(degs, ys, marker="o", label=m)
    plt.xlabel("Polynomial degree")
    plt.ylabel("Estimated memory (MB)")
    plt.title("Memory Footprint vs Degree")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f5 = figs_dir / f"memory_vs_degree.png"
    plt.tight_layout(); plt.savefig(f5, dpi=160); plt.close()
    saved.append(f5)

    # 6) Condition number vs degree (use OLS rows once per degree)
    # Pick the first method available (condition number is per design, not method)
    any_method = next(iter(grid))
    degs, conds = _sorted_xy_for_metric(grid[any_method], "cond_XTX")
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(degs, conds, marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("cond(X'X) (log)")
    plt.title("Design Conditioning vs Degree")
    plt.grid(True, alpha=0.3)
    f6 = figs_dir / f"condition_vs_degree.png"
    plt.tight_layout(); plt.savefig(f6, dpi=160); plt.close()
    saved.append(f6)

    # 7) Stability delta (scaled - unscaled Test MSE) vs degree
    # As stored, the delta is duplicated per method; we read it from any_method
    degs, deltas = _sorted_xy_for_metric(grid[any_method], "scaled_vs_unscaled_mse_delta")
    plt.figure(figsize=(7, 4.5))
    plt.plot(degs, deltas, marker="o")
    plt.axhline(0.0, ls="--", alpha=0.5)
    plt.xlabel("Polynomial degree")
    plt.ylabel("Δ Test MSE (scaled − unscaled)")
    plt.title("Numerical Stability Proxy vs Degree")
    plt.grid(True, alpha=0.3)
    f7 = figs_dir / f"stability_delta_vs_degree.png"
    plt.tight_layout(); plt.savefig(f7, dpi=160); plt.close()
    saved.append(f7)

    # 8) Bias–Variance decomposition vs degree
    if bv_results:
        degs_bv = [int(r["degree"]) for r in bv_results]
        mse_bv = [r["mse"] for r in bv_results]
        bias2_bv = [r["bias2"] for r in bv_results]
        var_bv = [r["variance"] for r in bv_results]

        plt.figure(figsize=(7, 4.5))
        plt.plot(degs_bv, mse_bv, marker="o", label="MSE")
        plt.plot(degs_bv, bias2_bv, marker="o", label="Bias²")
        plt.plot(degs_bv, var_bv, marker="o", label="Variance")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Error components")
        plt.title("Bias–Variance Decomposition (OLS)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        f8 = figs_dir / f"bias_variance.png"
        plt.tight_layout(); plt.savefig(f8, dpi=160); plt.close()
        saved.append(f8)

    # 9) Accuracy–Speed Pareto: R^2(test) vs Fit time (log-x), points by (method, degree)
    plt.figure(figsize=(7, 4.8))
    for m, g in grid.items():
        # x = fit_time_ms, y = r2_test per degree
        degs = sorted(g.keys())
        xs = [g[d].fit_time_ms for d in degs]
        ys = [g[d].r2_test for d in degs]
        # scatter with lines by method to show degree evolution
        plt.semilogx(xs, ys, marker="o", linestyle="-", label=m)
    plt.xlabel("Fit time (ms, log scale)")
    plt.ylabel(r"$R^2$ (test)")
    plt.title("Accuracy–Speed Pareto (per method & degree)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    f9 = figs_dir / f"pareto_r2_vs_time.png"
    plt.tight_layout(); plt.savefig(f9, dpi=160); plt.close()
    saved.append(f9)

    if logger:
        for p in saved:
            logger.write(f"[INFO] Saved figure -> {p}\n")

    # --- Helpers to compare our impl vs sklearn for OLS and Ridge ---
    def _pair_series(ours_name: str, skl_name: str, metric: str):
        """Return (degrees_sorted, ours_values, skl_values) for a metric, intersecting degrees present in both."""
        if ours_name not in grid or skl_name not in grid:
            return [], [], []
        g_ours = grid[ours_name]
        g_skl = grid[skl_name]
        common = sorted(set(g_ours.keys()).intersection(g_skl.keys()))
        return common, [getattr(g_ours[d], metric) for d in common], [getattr(g_skl[d], metric) for d in common]

    # 10) Our vs sklearn – accuracy gap (OLS): ΔR2 and ΔMSE vs degree
    degs, r2_o, r2_s = _pair_series("ols", "sklearn-ols", "r2_test")
    _,  mse_o, mse_s = _pair_series("ols", "sklearn-ols", "mse_test")
    if degs:
        plt.figure(figsize=(7, 4.5))
        dr2 = [ro - rs for ro, rs in zip(r2_o, r2_s)]
        dmse = [mo - ms for mo, ms in zip(mse_o, mse_s)]
        plt.plot(degs, dr2, marker="o", label=r"Δ$R^2$ (ours − sklearn)")
        plt.plot(degs, dmse, marker="o", label=r"ΔMSE (ours − sklearn)")
        plt.axhline(0.0, ls="--", alpha=0.5)
        plt.xlabel("Polynomial degree")
        plt.title("OLS: Our vs scikit-learn (Accuracy Gap)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        f10 = figs_dir / f"cmp_ols_accuracy_gap.png"
        plt.tight_layout(); plt.savefig(f10, dpi=160); plt.close()
        saved.append(f10)

    # 11) Our vs sklearn – accuracy gap (Ridge): ΔR2 and ΔMSE vs degree
    degs, r2_o, r2_s = _pair_series("ridge", "sklearn-ridge", "r2_test")
    _,  mse_o, mse_s = _pair_series("ridge", "sklearn-ridge", "mse_test")
    if degs:
        plt.figure(figsize=(7, 4.5))
        dr2 = [ro - rs for ro, rs in zip(r2_o, r2_s)]
        dmse = [mo - ms for mo, ms in zip(mse_o, mse_s)]
        plt.plot(degs, dr2, marker="o", label=r"Δ$R^2$ (ours − sklearn)")
        plt.plot(degs, dmse, marker="o", label=r"ΔMSE (ours − sklearn)")
        plt.axhline(0.0, ls="--", alpha=0.5)
        plt.xlabel("Polynomial degree")
        plt.title("Ridge: Our vs scikit-learn (Accuracy Gap)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        f11 = figs_dir / f"cmp_ridge_accuracy_gap.png"
        plt.tight_layout(); plt.savefig(f11, dpi=160); plt.close()
        saved.append(f11)

    # 12) Our vs sklearn – fit time ratio (ours/sklearn) vs degree (OLS & Ridge)
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for ours_name, skl_name, label in [("ols", "sklearn-ols", "OLS"), ("ridge", "sklearn-ridge", "Ridge")]:
        degs, t_o, t_s = _pair_series(ours_name, skl_name, "fit_time_ms")
        if not degs:
            continue
        ratio = [ (to / ts) if (isinstance(ts, (int, float)) and ts > 0) else np.nan for to, ts in zip(t_o, t_s) ]
        plt.semilogy(degs, ratio, marker="o", label=f"{label} (ours/sklearn)")
        plotted = True
    if plotted:
        plt.xlabel("Polynomial degree")
        plt.ylabel("Fit time ratio (ours / sklearn, log)")
        plt.title("Fit Time Ratio vs Degree")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        f12 = figs_dir / f"time_ratio_vs_degree.png"
        plt.tight_layout(); plt.savefig(f12, dpi=160); plt.close()
        saved.append(f12)
    else:
        plt.close()

 
def _sanitize_method_name(m: str) -> str:
    # Evita caratteri fastidiosi nei nomi file
    return (
        m.lower()
         .replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace("-", "_")
         .replace("(", "")
         .replace(")", "")
         .replace("__", "_")
    )

def write_per_method_csvs(rows: List[BenchRow], stamp: str) -> List[Path]:
    """
    Scrive un CSV per ciascun metodo, filtrando le righe di `rows`.
    Ritorna l'elenco dei percorsi creati (utile per log).
    """
    by_method: Dict[str, List[BenchRow]] = defaultdict(list)
    for r in rows:
        by_method[r.method].append(r)

    written: List[Path] = []
    for method, mrows in by_method.items():
        # Ordina per grado per plotting più semplice in pgfplots
        mrows_sorted = sorted(mrows, key=lambda r: r.degree)
        fieldnames = list(asdict(mrows_sorted[0]).keys()) if mrows_sorted else []

        safe = _sanitize_method_name(method)
        out_path = TABLES_DIR / f"benchmarks_{safe}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in mrows_sorted:
                w.writerow(asdict(r))
        written.append(out_path)
    return written


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Extra-metric benchmarks for OLS/Ridge and GD variants.")
    parser.add_argument("--n-points", type=int, default=100, help="Total points on Runge function in [-1,1].")
    parser.add_argument("--max-degree", type=int, default=15, help="Max polynomial degree.")
    parser.add_argument("--noise", action="store_true", help="Add Gaussian noise to Runge(y).")
    parser.add_argument(
    "--methods",
    nargs="+",
    default=["ols", "ridge", "gd-vanilla", "gd-momentum", "gd-adam", "sklearn-ols", "sklearn-ridge", "sklearn-lasso"],
    help=("Methods to run. "
          "Ours: ols, ridge, gd-vanilla, gd-momentum, gd-adagrad, gd-rmsprop, gd-adam. "
          "Scikit-learn: sklearn-ols, sklearn-ridge, sklearn-lasso"))

    parser.add_argument("--lam", type=float, default=1e-2, help="Lambda for Ridge (and ignored elsewhere).")

    # GD knobs
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-iter", type=int, default=10_000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--use-sgd", action="store_true", help="Use SGD with mini-batches for GD methods.")
    parser.add_argument("--batch-size", type=int, default=32)

    # Bias-variance
    parser.add_argument("--bootstrap", type=int, default=30, help="Bootstrap runs per degree for bias-variance.")
    parser.add_argument("--test-size", type=float, default=0.33)

    args = parser.parse_args()

    # Seed (use env SEED if set, otherwise 314 to mirror ml_core)
    env_seed = os.environ.get("SEED")
    seed = int(env_seed) if env_seed is not None else 314
    np.random.seed(seed)

    ensure_dirs()
    stamp = now_stamp()

    log_path = LOGS_DIR / f"benchmarks.log"
    logger = Logger(log_path)
    logger.write(f"[INFO] Starting benchmarks at {stamp}\n")
    logger.write(f"[INFO] Seed = {seed}\n")
    logger.write(f"[INFO] Args = {vars(args)}\n\n")

    # Generate data on Runge function
    x = np.linspace(-1, 1, args.n_points)
    y = runge_function(x, noise=args.noise)

    degrees = list(range(1, args.max_degree + 1))

    rows: List[BenchRow] = []

    # We need raw split vectors for stability delta
    # We'll redo a consistent split once and reuse raw x_tr/x_te for the delta function
    x_tr_raw, x_te_raw, y_tr_raw, y_te_raw = train_test_split(x, y, test_size=args.test_size, random_state=seed)

    for deg in degrees:
        # Build scaled design matrices (per-degree stats)
        Xtr, Xte, y_tr, y_te = build_poly_train_test(x, y, degree=deg, test_size=args.test_size, rs=seed)

        # Stability proxy: scaled-vs-unscaled test MSE delta using same split
        delta = stability_delta_unscaled(
            Xtr, Xte, y_tr, y_te, x_tr_raw, x_te_raw, degree=deg
        )

        for method in args.methods:
            gd_opts = dict(
                lr=args.lr,
                n_iter=args.n_iter,
                tol=args.tol,
                beta=args.beta,
                epsilon=args.epsilon,
                use_sgd=args.use_sgd,
                batch_size=args.batch_size,
            )

            try:
                row, _ = run_one_method(
                    method=method,
                    degree=deg,
                    Xtr=Xtr, Xte=Xte, y_tr=y_tr, y_te=y_te,
                    lam=args.lam,
                    gd_opts=gd_opts,
                )
                # patch in stability delta
                row.scaled_vs_unscaled_mse_delta = delta
                rows.append(row)
            except Exception as e:
                logger.write(f"[WARN] Failed (degree={deg}, method={method}): {e}\n")
                # Append a placeholder row so the CSV has the full grid
                n_train, p = Xtr.shape
                rows.append(BenchRow(
                    degree=deg, method=method, n_points=len(x), n_train=n_train, n_test=len(x)-n_train, p=p,
                    fit_time_ms=float("nan"), predict_time_ms=float("nan"), total_time_ms=float("nan"),
                    iters=0, converged=False,
                    mse_train=float("nan"), r2_train=float("nan"), mse_test=float("nan"), r2_test=float("nan"),
                    flops_est=float("nan"), mem_mb_est=float("nan"), cond_XTX=float("inf"),
                    scaled_vs_unscaled_mse_delta=float("inf"),
                ))

    # Save main CSV
    csv_path = TABLES_DIR / f"benchmarks.csv"
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    logger.write(f"[INFO] Saved main benchmarks CSV -> {csv_path}\n")

    # Save one CSV per method (comodo per pgfplots/LaTeX)
    method_csvs = write_per_method_csvs(rows, stamp)
    for p in method_csvs:
        logger.write(f"[INFO] Saved per-method CSV -> {p}\n")

    # Bias-variance study (OLS closed-form only; fast and illustrative)
    bv_results = bias_variance_grid(
        x=x, y=y, degrees=degrees, n_bootstrap=args.bootstrap,
        test_size=args.test_size, rs=seed
    )
    bv_csv = TABLES_DIR / f"bias_variance.csv"
    with open(bv_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["degree", "mse", "bias2", "variance"])
        writer.writeheader()
        writer.writerows(bv_results)
    logger.write(f"[INFO] Saved bias-variance CSV -> {bv_csv}\n")

    # -------- Terminal summary --------
    print("\n=== BENCHMARK SUMMARY (aggregated by method) ===")
    # Aggregate across degrees: mean test R2, mean fit time, mean iterations
    agg: Dict[str, Dict[str, float]] = {}
    for r in rows:
        m = r.method
        if m not in agg:
            agg[m] = {
                "count": 0,
                "mean_r2_test": 0.0,
                "mean_mse_test": 0.0,
                "mean_fit_ms": 0.0,
                "mean_iters": 0.0,
                "mean_flops": 0.0,
                "mean_mem_mb": 0.0,
                "mean_cond": 0.0,
            }
        A = agg[m]
        A["count"] += 1
        A["mean_r2_test"] += r.r2_test if not math.isnan(r.r2_test) else 0.0
        A["mean_mse_test"] += r.mse_test if not math.isnan(r.mse_test) else 0.0
        A["mean_fit_ms"] += r.fit_time_ms if not math.isnan(r.fit_time_ms) else 0.0
        A["mean_iters"] += r.iters
        A["mean_flops"] += 0.0 if math.isnan(r.flops_est) else r.flops_est
        A["mean_mem_mb"] += 0.0 if math.isnan(r.mem_mb_est) else r.mem_mb_est
        A["mean_cond"] += 0.0 if math.isinf(r.cond_XTX) else r.cond_XTX

    summary_rows = []
    for m, A in agg.items():
        c = max(A["count"], 1)
        summary_rows.append({
            "method": m,
            "avg R2(test)": f"{A['mean_r2_test']/c:.4f}",
            "avg MSE(test)": f"{A['mean_mse_test']/c:.4e}",
            "avg fit ms": f"{A['mean_fit_ms']/c:.2f}",
            "avg iters": f"{A['mean_iters']/c:.1f}",
            "avg FLOPs": human(A["mean_flops"]/c),
            "avg Mem(MB)": f"{A['mean_mem_mb']/c:.2f}",
            "avg cond(X'X)": f"{A['mean_cond']/c:.2e}" if A["mean_cond"] > 0 else "n/a",
        })

    print_table(summary_rows, ["method", "avg R2(test)", "avg MSE(test)", "avg fit ms", "avg iters", "avg FLOPs", "avg Mem(MB)", "avg cond(X'X)"])

    # Small preview of the full grid (first 12 rows)
    print("\n=== PREVIEW: first 12 rows of full grid (see CSV for complete) ===")
    preview_cols = ["degree", "method", "n_points", "p", "fit_time_ms", "iters", "mse_test", "r2_test", "flops_est", "mem_mb_est", "cond_XTX", "scaled_vs_unscaled_mse_delta"]
    preview = [asdict(r) for r in rows]
    print_table(preview, preview_cols, max_rows=12)
    make_plots(rows, bv_results, stamp, logger=logger)

    logger.write("\n[INFO] Done.\n")
    logger.close()

if __name__ == "__main__":
    main()
