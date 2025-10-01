# -----------------------------------------------------------------------------------------
# Part c, The OLS and Ridge experiment for polynomials to degree 15 comparing analytical method to gradient descent
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, save_vector_with_degree, seed
)
from tqdm import trange

# -----------------------------
# Settings
# -----------------------------
n_points = [100]   # fix one dataset size (simpler comparison)
max_degree = 15
noise = True
lam = 0.01          # Ridge regularization parameter
learning_rates = [0.0001, 0.001, 0.01]  # try several η values
n_iter = 2000
N_RUNS = 20

# -----------------------------
# Storage
# -----------------------------
mse_analytical_ols_runs   = np.zeros((max_degree, len(n_points), N_RUNS))
mse_analytical_ridge_runs = np.zeros((max_degree, len(n_points), N_RUNS))
mse_gd_ols_runs   = {eta: np.zeros((max_degree, len(n_points), N_RUNS)) for eta in learning_rates}
mse_gd_ridge_runs = {eta: np.zeros((max_degree, len(n_points), N_RUNS)) for eta in learning_rates}

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

print(f">>> Starting Part C OLS vs Ridge (analytical vs GD) | runs={N_RUNS} "
      f"| n_points={n_points} | max_degree={max_degree} | noise={noise} | lam={lam}")
# print(">>> Seed policy: for each run r we set np.random.seed(seed+r) and use random_state=seed+r in the split,")
# print(">>> so within a run the dataset/split are IDENTICAL across models, methods and learning rates (fair comparison).")

# -----------------------------
# Experiment (single outer progress bar over runs)
# -----------------------------
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # Fix the run seed -> same noise & same train/test split for all models in this run
    np.random.seed(seed + r)
    for j, n in enumerate(n_points):
        # Generate one dataset per run (shared by all methods/LRs)
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)
        # Consistent split for all models in this run
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)
        for degree in range(1, max_degree+1):
            # Fit scaler on train once per degree; reuse for all methods/LRs
            X_train, col_means, col_stds = polynomial_features_scaled(
                x_train.flatten(), degree, return_stats=True
            )
            X_test = polynomial_features_scaled(
                x_test.flatten(), degree, col_means=col_means, col_stds=col_stds
            )
            di = degree - 1
            # ----- Analytical OLS -----
            theta_ols = OLS_parameters(X_train, y_train)
            y_test_pred = X_test @ theta_ols
            mse_analytical_ols_runs[di, j, r] = MSE(y_test, y_test_pred)
            # ----- Analytical Ridge -----
            theta_ridge = Ridge_parameters(X_train, y_train, lam=lam, intercept=True)
            y_test_pred = X_test @ theta_ridge
            mse_analytical_ridge_runs[di, j, r] = MSE(y_test, y_test_pred)
            # ----- Gradient Descent OLS -----
            for eta in learning_rates:
                theta_gd = Gradient_descent_advanced(
                    X_train, y_train, Type=0, lr=eta, n_iter=n_iter,
                    method='vanilla', lam=0.0, theta_history=False
                )
                y_test_pred_gd = X_test @ theta_gd
                mse_gd_ols_runs[eta][di, j, r] = MSE(y_test, y_test_pred_gd)
            # ----- Gradient Descent Ridge -----
            for eta in learning_rates:
                theta_gd = Gradient_descent_advanced(
                    X_train, y_train, Type=1, lam=lam, lr=eta, n_iter=n_iter,
                    method='vanilla', theta_history=False
                )
                y_test_pred_gd = X_test @ theta_gd
                mse_gd_ridge_runs[eta][di, j, r] = MSE(y_test, y_test_pred_gd)

# Save tables
idx0 = 0  # since n_points = [100], take the first (and only) column
# Aggregate mean and std across runs (append *_std column)
mse_analytical_ols   = mse_analytical_ols_runs.mean(axis=2)
mse_analytical_ols_s = mse_analytical_ols_runs.std(axis=2, ddof=1)
mse_analytical_ridge   = mse_analytical_ridge_runs.mean(axis=2)
mse_analytical_ridge_s = mse_analytical_ridge_runs.std(axis=2, ddof=1)

save_vector_with_degree(
    TAB / "part_c_mse_analytical_ols.csv",
    mse_analytical_ols[:, idx0], "MSE_analytical_OLS",
    std=mse_analytical_ols_s[:, idx0]
)
save_vector_with_degree(
    TAB / "part_c_mse_analytical_ridge.csv",
    mse_analytical_ridge[:, idx0], "MSE_analytical_Ridge",
    std=mse_analytical_ridge_s[:, idx0]
)
for eta in learning_rates:
    mse_gd_ols_mean = mse_gd_ols_runs[eta].mean(axis=2)
    mse_gd_ols_std  = mse_gd_ols_runs[eta].std(axis=2, ddof=1)
    save_vector_with_degree(
        TAB / f"part_c_mse_gd_ols_lr={eta}.csv",
        mse_gd_ols_mean[:, idx0], f"MSE_GD_OLS_lr={eta}",
        std=mse_gd_ols_std[:, idx0]
    )
    mse_gd_ridge_mean = mse_gd_ridge_runs[eta].mean(axis=2)
    mse_gd_ridge_std  = mse_gd_ridge_runs[eta].std(axis=2, ddof=1)
    save_vector_with_degree(
        TAB / f"part_c_mse_gd_ridge_lr={eta}.csv",
        mse_gd_ridge_mean[:, idx0], f"MSE_GD_Ridge_lr={eta}",
        std=mse_gd_ridge_std[:, idx0]
    )

print("Part C done.")
