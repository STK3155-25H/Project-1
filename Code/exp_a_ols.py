# -----------------------------------------------------------------------------------------
# Part a, The OLS experiment for polynomials to degree 15 and different amount of data points
# -----------------------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, MSE, R2_score, save_matrix_with_degree_cols_plus_std, seed
)
from tqdm import tqdm, trange


# -----------------------------
# Settings
# -----------------------------
n_points = [40, 50, 100, 500, 1000]
max_degree = 15
noise = True
N_RUNS = 30


# -----------------------------
# Storage
# -----------------------------
mse_train_runs = np.zeros((max_degree, len(n_points), N_RUNS))
mse_test_runs  = np.zeros((max_degree, len(n_points), N_RUNS))
R2_train_runs  = np.zeros((max_degree, len(n_points), N_RUNS))
R2_test_runs   = np.zeros((max_degree, len(n_points), N_RUNS))
theta_list     = [[None for _ in range(len(n_points))] for _ in range(max_degree)]

# Make output dirs
OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Experiment
# -----------------------------
print(f">>> Starting Part A OLS experiment | runs={N_RUNS} | noise={noise} | seed={seed}")
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # nuovo seed per ogni run (indipendenza delle realizzazioni)
    np.random.seed(seed + r)
    for j, n in enumerate(n_points):
        # Generate Runge data
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)
        # Split & scale (random_state diverso per ogni run)
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)
        for degree in range(1, max_degree+1):
            # features polinomiali (fit scaler su train e applichiamo a test)
            X_train, col_means, col_stds = polynomial_features_scaled(x_train.flatten(), degree, return_stats=True)
            X_test = polynomial_features_scaled(x_test.flatten(), degree, col_means=col_means, col_stds=col_stds)
            # OLS closed
            theta = OLS_parameters(X_train, y_train)
            # Predictions
            y_train_pred = X_train @ theta
            y_test_pred  = X_test  @ theta
            # Store per-run
            di = degree - 1
            mse_train_runs[di, j, r] = MSE(y_train, y_train_pred)
            mse_test_runs [di, j, r] = MSE(y_test,  y_test_pred)
            R2_train_runs [di, j, r] = R2_score(y_train, y_train_pred)
            R2_test_runs  [di, j, r] = R2_score(y_test,  y_test_pred)
            theta_list[di][j]        = theta  # ultimo θ della run

# calcoliamo mean e std lungo l'asse "run"
mse_train_list = mse_train_runs.mean(axis=2)
mse_test_list  = mse_test_runs.mean(axis=2)
R2_train_list  = R2_train_runs.mean(axis=2)
R2_test_list   = R2_test_runs.mean(axis=2)
mse_train_std  = mse_train_runs.std(axis=2, ddof=1)
mse_test_std   = mse_test_runs.std(axis=2, ddof=1)
R2_train_std   = R2_train_runs.std(axis=2, ddof=1)
R2_test_std    = R2_test_runs.std(axis=2, ddof=1)

# Save tables
col_names = [f"n={n}" for n in n_points]
save_matrix_with_degree_cols_plus_std(TAB / "part_a_mse_test.csv", mse_test_list, mse_test_std, col_names)
save_matrix_with_degree_cols_plus_std(TAB / "part_a_r2_test.csv",  R2_test_list,  R2_test_std,  col_names)

print(f"Part A done. Aggregated over {N_RUNS} runs.")
