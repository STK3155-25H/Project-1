# ------------------------------------------------------------
# Benchmarks for: polynomial features, OLS (closed-form vs sklearn),
# Ridge (closed-form vs sklearn), and GD variants vs OLS.
# Collects: execution time, memory peak, iteration counts (GD),
# estimated FLOPs, and accuracy gaps.
#
# Run:  python benchmark_ml_impls.py
# Output: ./benchmark_results.csv
# ------------------------------------------------------------

import os
import time
import tracemalloc
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

# --- Import your module AFTER setting the seed so your code picks it at import time ---
os.environ["SEED"] = "314"
import importlib
import src  # noqa
importlib.reload(src)
from src import (
    runge_function, polynomial_features_scaled, split_scale,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced,
    MSE, R2_score
)

# ---------------------------
# Helpers
# ---------------------------
def time_and_memory(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) measuring elapsed wall-clock time and peak memory.
    Uses tracemalloc to track Python allocations (not total RSS, but consistent and portable).
    Returns: (result, elapsed_seconds, peak_kb)
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024.0  # KB

def estimate_flops_ols_normal_eq(n, p):
    """
    Rough FLOPs estimate for normal equations using:
      - XtX: ~2*n*p^2
      - Xty: ~2*n*p
      - Solve (inv or pinv): ~O(p^3)  (we use c*p^3 with c~2 as rough constant)
    Note: your OLS uses pinv; SVD cost is ~O(p^2*n + p^3). We approximate with these dominant terms.
    Returns an integer FLOPs estimate.
    """
    return int(2*n*(p**2) + 2*n*p + 2*(p**3))

def estimate_flops_ridge(n, p):
    """
    Ridge closed-form with (XtX + lam*I) and solve:
      - XtX: ~2*n*p^2
      - Xty: ~2*n*p
      - Solve: ~O(p^3)
    """
    return int(2*n*(p**2) + 2*n*p + 2*(p**3))

def estimate_flops_gd_per_iter(n, p, method='vanilla'):
    """
    FLOPs per GD iteration (very rough):
      - X @ theta: ~2*n*p
      - residual (y - X@theta): ~n
      - X.T @ residual: ~2*n*p
      - update: ~p
    Total ~ 4*n*p + O(n + p)  -> approximate as 4*n*p.
    Adaptive methods add a small constant factor per-param operations; we account for it.
    """
    base = 4 * n * p
    if method in ('adagrad', 'rmsprop', 'adam'):
        # A few extra vector ops per parameter: ~5p; negligible vs 4*n*p when n is large,
        # but we add a small premium to reflect overhead.
        return base + 10 * p
    else:
        return base

def lr_for_degree(deg):
    """A simple LR heuristic that works decently with your tests."""
    if deg <= 3: return 0.05
    if deg == 5: return 0.02
    if deg >= 7: return 0.01
    return 0.02

# ---------------------------
# Benchmark settings
# ---------------------------
DEGREES = [1, 3, 5, 7]
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1]
GD_METHODS = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
GD_MAX_ITER = 50_000
GD_TOL = 1e-10
USE_SGD = False  # use full-batch to compare cleanly to OLS closed-form

# Data: Runge clean (no noise) to focus on numeric/algorithmic differences
x = np.linspace(-1, 1, 200)
y = runge_function(x, noise=False)

# Same split/scale procedure you use in the project
X_train, X_test, y_train, y_test = split_scale(x, y)
# Flatten train features because your polynomial_features_scaled expects 1D x
xtr_1d = X_train.ravel()
xte_1d = X_test.ravel()
n_train = X_train.shape[0]

rows = []

# ==========================================================
# 1) Polynomial features vs sklearn + scaling consistency
#    (We only time/measure your path because sklearn path is trivial & optimized in C)
# ==========================================================
for deg in DEGREES:
    # Your poly + scaling (train)
    (Xtr, mu, sig), t_build, mem_kb = time_and_memory(
        polynomial_features_scaled, xtr_1d, deg, True, None, None, True
    )
    # Test matrix using same stats
    Xte, t_build_te, mem_kb_te = time_and_memory(
        polynomial_features_scaled, xte_1d, deg, True, mu, sig, False
    )
    p = Xtr.shape[1]

    rows.append(dict(
        section="poly_features",
        model="your_poly_scaled",
        degree=deg,
        alpha=np.nan,
        method=np.nan,
        n_train=n_train,
        p=p,
        fit_time_s=t_build,           # feature building time
        predict_time_s=t_build_te,    # test transform time
        memory_peak_kb=mem_kb + mem_kb_te,
        iterations=np.nan,
        flops_estimated=np.nan,
        r2_test=np.nan,
        mse_test=np.nan,
        coeff_l2_gap=np.nan,
        yhat_mse_gap=np.nan,
    ))

# ==========================================================
# 2) OLS closed-form vs sklearn LinearRegression (fit_intercept=False)
# ==========================================================
for deg in DEGREES:
    # Build features (already done above, but keep self-contained per loop)
    (Xtr, mu, sig), _, _ = time_and_memory(
        polynomial_features_scaled, xtr_1d, deg, True, None, None, True
    )
    Xte, _, _ = time_and_memory(
        polynomial_features_scaled, xte_1d, deg, True, mu, sig, False
    )
    n, p = Xtr.shape

    # Your OLS
    theta, t_fit_my, mem_my = time_and_memory(OLS_parameters, Xtr, y_train)
    # Predict
    y_hat_my, t_pred_my, mem_pred_my = time_and_memory(lambda A, t: A @ t, Xte, theta)
    mse_my = MSE(y_test, y_hat_my)
    r2_my = R2_score(y_test, y_hat_my)

    # sklearn OLS with the same design (bias is inside X), so fit_intercept=False
    lr = LinearRegression(fit_intercept=False)
    _, t_fit_skl, mem_skl = time_and_memory(lr.fit, Xtr, y_train)
    y_hat_skl, t_pred_skl, mem_pred_skl = time_and_memory(lr.predict, Xte)
    mse_skl = MSE(y_test, y_hat_skl)
    r2_skl = R2_score(y_test, y_hat_skl)

    coeff_gap = np.linalg.norm(theta - lr.coef_)
    yhat_gap = MSE(y_hat_skl, y_hat_my)
    flops = estimate_flops_ols_normal_eq(n, p)

    rows.append(dict(
        section="OLS",
        model="closed_form_vs_sklearn",
        degree=deg,
        alpha=np.nan,
        method="closed_form",
        n_train=n,
        p=p,
        fit_time_s=t_fit_my,
        predict_time_s=t_pred_my,
        memory_peak_kb=mem_my + mem_pred_my,
        iterations=np.nan,
        flops_estimated=flops,
        r2_test=r2_my,
        mse_test=mse_my,
        coeff_l2_gap=coeff_gap,
        yhat_mse_gap=yhat_gap,
    ))
    rows.append(dict(
        section="OLS",
        model="sklearn_linear_regression",
        degree=deg,
        alpha=np.nan,
        method="sklearn",
        n_train=n,
        p=p,
        fit_time_s=t_fit_skl,
        predict_time_s=t_pred_skl,
        memory_peak_kb=mem_skl + mem_pred_skl,
        iterations=np.nan,
        flops_estimated=np.nan,  # we do not estimate internal C/Fortran FLOPs
        r2_test=r2_skl,
        mse_test=mse_skl,
        coeff_l2_gap=0.0,
        yhat_mse_gap=0.0,
    ))

# ==========================================================
# 3) Ridge closed-form vs sklearn Ridge (fit_intercept=False)
#    (to match your tests where you removed intercept column or disabled intercept penalty)
# ==========================================================
for deg in DEGREES:
    # Here we follow the same protocol used in your tests for Ridge: intercept=False
    # so we build polynomial features WITHOUT the bias column.
    (Xtr, mu, sig), _, _ = time_and_memory(
        polynomial_features_scaled, xtr_1d, deg, False, None, None, True
    )
    Xte, _, _ = time_and_memory(
        polynomial_features_scaled, xte_1d, deg, False, mu, sig, False
    )
    n, p = Xtr.shape

    for alpha in RIDGE_ALPHAS:
        theta, t_fit_my, mem_my = time_and_memory(Ridge_parameters, Xtr, y_train, lam=alpha, intercept=False)
        y_hat_my, t_pred_my, mem_pred_my = time_and_memory(lambda A, t: A @ t, Xte, theta)
        mse_my = MSE(y_test, y_hat_my)
        r2_my = R2_score(y_test, y_hat_my)

        rr = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
        _, t_fit_skl, mem_skl = time_and_memory(rr.fit, Xtr, y_train)
        y_hat_skl, t_pred_skl, mem_pred_skl = time_and_memory(rr.predict, Xte)
        mse_skl = MSE(y_test, y_hat_skl)
        r2_skl = R2_score(y_test, y_hat_skl)

        coeff_gap = np.linalg.norm(theta - rr.coef_)
        yhat_gap = MSE(y_hat_skl, y_hat_my)
        flops = estimate_flops_ridge(n, p)

        rows.append(dict(
            section="Ridge",
            model="closed_form",
            degree=deg,
            alpha=alpha,
            method="closed_form",
            n_train=n,
            p=p,
            fit_time_s=t_fit_my,
            predict_time_s=t_pred_my,
            memory_peak_kb=mem_my + mem_pred_my,
            iterations=np.nan,
            flops_estimated=flops,
            r2_test=r2_my,
            mse_test=mse_my,
            coeff_l2_gap=coeff_gap,
            yhat_mse_gap=yhat_gap,
        ))
        rows.append(dict(
            section="Ridge",
            model="sklearn_ridge",
            degree=deg,
            alpha=alpha,
            method="sklearn",
            n_train=n,
            p=p,
            fit_time_s=t_fit_skl,
            predict_time_s=t_pred_skl,
            memory_peak_kb=mem_skl + mem_pred_skl,
            iterations=np.nan,
            flops_estimated=np.nan,
            r2_test=r2_skl,
            mse_test=mse_skl,
            coeff_l2_gap=0.0,
            yhat_mse_gap=0.0,
        ))

# ==========================================================
# 4) GD variants vs OLS optimum (Type=0, lam=0). We report:
#    time, iterations to convergence (via theta_history length),
#    parameter gap and yhat gap vs closed-form,
#    estimated total FLOPs.
# ==========================================================
for deg in DEGREES:
    (Xtr, mu, sig), _, _ = time_and_memory(
        polynomial_features_scaled, xtr_1d, deg, True, None, None, True
    )
    Xte, _, _ = time_and_memory(
        polynomial_features_scaled, xte_1d, deg, True, mu, sig, False
    )
    n, p = Xtr.shape

    # Closed-form optimum for reference
    theta_star = OLS_parameters(Xtr, y_train)
    y_star = Xte @ theta_star
    mse_star = MSE(y_test, y_star)
    r2_star = R2_score(y_test, y_star)

    for method in GD_METHODS:
        lr = lr_for_degree(deg)

        # Ask for theta history to count iterations to convergence
        history, t_gd, mem_gd = time_and_memory(
            Gradient_descent_advanced, Xtr, y_train,
            0, 0.0, lr, GD_MAX_ITER, GD_TOL, method, 0.9, 1e-8, 1, USE_SGD, True
        )
        # history has shape (n_iter_eff, p)
        iters = history.shape[0]
        theta_gd = history[-1] if iters > 0 else np.zeros(p)

        # Predictions and metrics
        y_gd, t_pred, mem_pred = time_and_memory(lambda A, t: A @ t, Xte, theta_gd)
        mse_gd = MSE(y_test, y_gd)
        r2_gd = R2_score(y_test, y_gd)

        coeff_gap = np.linalg.norm(theta_gd - theta_star)
        yhat_gap = MSE(y_star, y_gd)
        flops_per_iter = estimate_flops_gd_per_iter(n, p, method)
        flops_total = int(flops_per_iter * max(iters, 1))

        rows.append(dict(
            section="GD",
            model="your_GD_vs_OLS",
            degree=deg,
            alpha=np.nan,
            method=method,
            n_train=n,
            p=p,
            fit_time_s=t_gd,                 # "training" time
            predict_time_s=t_pred,           # prediction time
            memory_peak_kb=mem_gd + mem_pred,
            iterations=iters,
            flops_estimated=flops_total,
            r2_test=r2_gd,
            mse_test=mse_gd,
            coeff_l2_gap=coeff_gap,
            yhat_mse_gap=yhat_gap,
        ))

# ---------------------------
# Save & pretty-print
# ---------------------------
df = pd.DataFrame(rows)

# Sort for readability
df.sort_values(by=["section", "degree", "alpha", "method"], inplace=True, na_position="last")

out_path = "benchmark_results.csv"
df.to_csv(out_path, index=False)

# Console summary: best models by section/degree on R2 (ties broken by time)
def pick_best(g):
    g2 = g.sort_values(["r2_test", "fit_time_s"], ascending=[False, True])
    return g2.head(1)

summary = df.groupby(["section", "degree"]).apply(pick_best).reset_index(drop=True)

print("\n=== WROTE:", out_path, "===\n")
print("=== QUICK SUMMARY (best R2 per section/degree) ===")
with pd.option_context("display.max_columns", None, "display.width", 140):
    print(summary)
