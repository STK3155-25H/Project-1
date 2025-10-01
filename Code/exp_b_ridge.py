# -----------------------------------------------------------------------------------------
# Part b, The Ridge experiment for polynomials to degree 15, different amount of data points and different lambda values
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    Ridge_parameters, MSE, R2_score, save_matrix_with_degree_cols_plus_std, seed
)
from tqdm import trange
# -----------------------------
# Settings
# -----------------------------
n_points = [40, 50, 100, 500, 1000]
lam_list = [0, 0.001, 0.01, 0.1, 1, 10, 100]
max_degree = 15
noise = True
N_RUNS = 20

# -----------------------------
# Storage
# -----------------------------
# Per-run: (degree, n_points, lambdas, runs)
mse_train_runs = np.zeros((max_degree, len(n_points), len(lam_list), N_RUNS))
mse_test_runs  = np.zeros((max_degree, len(n_points), len(lam_list), N_RUNS))
R2_train_runs  = np.zeros((max_degree, len(n_points), len(lam_list), N_RUNS))
R2_test_runs   = np.zeros((max_degree, len(n_points), len(lam_list), N_RUNS))
theta_list     = [[[None for _ in range(len(lam_list))] for _ in range(len(n_points))] for _ in range(max_degree)]

# Output dirs
OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
print(f">>> Starting Part B Ridge experiment | runs={N_RUNS} | "
      f"n_points={n_points} | lambdas={lam_list} | max_degree={max_degree} | noise={noise}")
# Seed policy: for each run r I use np.random.seed(seed+r) and the same random_state=seed+r in the split,")
# So for each (run, n) the dataset and the lambdas are identical for each lambda (fair comparison).")

# Unica progress bar: una per le run
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # 1) Fisso il seed della run => stessa realizzazione del rumore e stesso split per tutti i lambda
    np.random.seed(seed + r)
    for j, n in enumerate(n_points):
        # Dataset unico per tutti i lambda in questa run (fair comparison)
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)
        # Split/scale coerente per tutti i lambda in questa run
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)
        for degree in range(1, max_degree+1):
            # Features: fit scaler sul train della run, e applico al test (coerente per tutti i lambda)
            X_train, col_means, col_stds = polynomial_features_scaled(
                x_train.flatten(), degree, return_stats=True
            )
            X_test = polynomial_features_scaled(
                x_test.flatten(), degree, col_means=col_means, col_stds=col_stds
            )
            for k, lam in enumerate(lam_list):
                # Ridge chiuso (intercept non penalizzato by default)
                theta = Ridge_parameters(X_train, y_train, lam)
                y_train_pred = X_train @ theta
                y_test_pred  = X_test  @ theta
                di = degree - 1
                mse_train_runs[di, j, k, r] = MSE(y_train, y_train_pred)
                mse_test_runs [di, j, k, r] = MSE(y_test,  y_test_pred)
                R2_train_runs [di, j, k, r] = R2_score(y_train, y_train_pred)
                R2_test_runs  [di, j, k, r] = R2_score(y_test,  y_test_pred)
                theta_list[di][j][k]        = theta  # salvo l'ultimo theta della run
# Medie e std lungo l'asse run
mse_train_list = mse_train_runs.mean(axis=3)
mse_test_list  = mse_test_runs.mean(axis=3)
R2_train_list  = R2_train_runs.mean(axis=3)
R2_test_list   = R2_test_runs.mean(axis=3)

mse_train_std  = mse_train_runs.std(axis=3, ddof=1)
mse_test_std   = mse_test_runs.std(axis=3, ddof=1)
R2_train_std   = R2_train_runs.std(axis=3, ddof=1)
R2_test_std    = R2_test_runs.std(axis=3, ddof=1)

# Salvataggi per-lambda: colonne = n_points (mean) + colonne *_std appese
col_names = [f"n={n}" for n in n_points]
for k, lam in enumerate(lam_list):
    save_matrix_with_degree_cols_plus_std(
        TAB / f"part_b_mse_test_lambda={lam}.csv",
        mse_test_list[:, :, k], mse_test_std[:, :, k], col_names
    )
    save_matrix_with_degree_cols_plus_std(
        TAB / f"part_b_r2_test_lambda={lam}.csv",
        R2_test_list[:, :, k], R2_test_std[:, :, k], col_names
    )

# -----------------------------
# Save MSE table for n=100 vs all lambdas (same style)
# -----------------------------
col_names = [f"lambda={lam}" for lam in lam_list]  # columns are lambdas
idx100 = n_points.index(100)
save_matrix_with_degree_cols_plus_std(
    TAB / "part_b_mse_n100_all_lambdas.csv",
    mse_test_list[:, idx100, :],
    mse_test_std[:, idx100, :],
    col_names
)


print("Part B done")
