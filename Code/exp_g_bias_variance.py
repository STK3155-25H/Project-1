# -----------------------------------------------------------------------------------------
# Part g, Bias-Variance analysis for polynomials to degree 15, different amount of data points
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, MSE_Bias_Variance, save_vector_with_degree, seed
)
from tqdm import trange

# -----------------------------
# Settings
# -----------------------------
n_points_list = [100]   # Different dataset sizes
max_degree = 15
noise = True
n_bootstrap = 50   # Number of bootstrap samples
N_RUNS = 30
# -----------------------------
# Storage arrays
# -----------------------------
mse_runs      = np.zeros((max_degree, len(n_points_list), N_RUNS))
bias2_runs    = np.zeros((max_degree, len(n_points_list), N_RUNS))
variance_runs = np.zeros((max_degree, len(n_points_list), N_RUNS))

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

print(f">>> Starting Part G: Bias-Variance | runs={N_RUNS} | n_points_list={n_points_list} "
      f"| max_degree={max_degree} | noise={noise} | n_bootstrap={n_bootstrap}")
# print(">>> Seed policy: for run r we set np.random.seed(seed+r) and use random_state=seed+r in the split;")
# print(">>> bootstrap resampling uses deterministic seeds per (run, n_points, degree, i) for reproducibility.")

# -----------------------------
# Experiment
# -----------------------------
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # Fix run seed -> same noise and same split choices for this run
    np.random.seed(seed + r)
    for j, n_points in enumerate(n_points_list):
        # Generate dataset once per run
        x = np.linspace(-1, 1, n_points)
        y = runge_function(x, noise=noise)
        # Consistent split for this run (train/test shared across degrees)
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)
        for degree in range(1, max_degree+1):
            # Design matrices (fit scaler on train, reuse for test)
            X_train, col_means, col_stds = polynomial_features_scaled(
                x_train.flatten(), degree, return_stats=True
            )
            X_test = polynomial_features_scaled(
                x_test.flatten(), degree, col_means=col_means, col_stds=col_stds
            )
            # Bootstrap predictions on test
            predictions = np.zeros((n_bootstrap, len(x_test)))
            # Deterministic bootstrap seeds for fairness and reproducibility
            bootstrap_seed_base = (seed + r) * 1_000_000 + j * 10_000 + degree * 100
            for i in range(n_bootstrap):
                X_re, y_re = resample(X_train, y_train, random_state=bootstrap_seed_base + i)
                theta = OLS_parameters(X_re, y_re)
                predictions[i] = X_test @ theta
            # Bias-variance decomposition against the true y_test
            mse, bias2, var = MSE_Bias_Variance(y_test, predictions)
            di = degree - 1
            mse_runs     [di, j, r] = mse
            bias2_runs   [di, j, r] = bias2
            variance_runs[di, j, r] = var


# Save per-n tables
for idx, n in enumerate(n_points_list):
    mse_mean = mse_runs[:, idx, :].mean(axis=1);     mse_std = mse_runs[:, idx, :].std(axis=1, ddof=1)
    b2_mean  = bias2_runs[:, idx, :].mean(axis=1);   b2_std  = bias2_runs[:, idx, :].std(axis=1, ddof=1)
    var_mean = variance_runs[:, idx, :].mean(axis=1);var_std = variance_runs[:, idx, :].std(axis=1, ddof=1)
    save_vector_with_degree(TAB / f"part_g_mse_n={n}.csv",      mse_mean, f"MSE_n={n}",      std=mse_std)
    save_vector_with_degree(TAB / f"part_g_bias2_n={n}.csv",    b2_mean,  f"Bias2_n={n}",    std=b2_std)
    save_vector_with_degree(TAB / f"part_g_variance_n={n}.csv", var_mean, f"Variance_n={n}", std=var_std)

print(f"Part G done. Aggregated over {N_RUNS} runs..")
