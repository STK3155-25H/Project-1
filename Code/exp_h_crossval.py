# -----------------------------------------------------------------------------------------
# Part h: Cross-Validation with OLS, Ridge, and Lasso
# -----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, seed, save_vector_with_degree
)
from tqdm import trange

# -----------------------------
# Settings
# -----------------------------
n_points = 100
max_degree = 15
noise = True
k_folds = 5
lam_ridge = 0.01
lam_lasso = 0.01
N_RUNS = 20

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# Generate Runge data
x = np.linspace(-1, 1, n_points)
y = runge_function(x, noise=noise)

# Split and scale data
x_train, x_test, y_train, y_test = split_scale(x, y)

# Storage arrays for CV MSE
print(f">>> Starting Part H: Cross-Validation | runs={N_RUNS} | n_points={n_points} | "
      f"max_degree={max_degree} | k_folds={k_folds} | noise={noise}")
# print(">>> Seed policy: per run r we set np.random.seed(seed+r), split_state=seed+r, and KFold(..., random_state=seed+r)")
# print(">>> so folds/splits are identical across degrees and models within a run (fair comparisons).")

# Storage across runs: (degree, runs)
mse_cv_OLS_runs   = np.zeros((max_degree, N_RUNS))
mse_cv_Ridge_runs = np.zeros((max_degree, N_RUNS))
mse_cv_Lasso_runs = np.zeros((max_degree, N_RUNS))
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # Fix run seed: consistent noise and splits for this run
    np.random.seed(seed + r)

    # Generate data and create the train/test split (test isn't used by CV, but we mirror your pipeline)
    x = np.linspace(-1, 1, n_points)
    y = runge_function(x, noise=noise)
    x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)

    # One KFold per run so all degrees/models share the exact same folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed + r)

    # Loop over polynomial degrees
    for degree in range(1, max_degree+1):
        mse_list_OLS, mse_list_Ridge, mse_list_Lasso = [], [], []
        for train_idx, val_idx in kf.split(x_train):
            # Fold data
            x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            # Features + scaling (fit on train fold, apply to val fold)
            X_train_fold, col_means, col_stds = polynomial_features_scaled(
                x_train_fold.flatten(), degree, return_stats=True
            )
            X_val_fold = polynomial_features_scaled(
                x_val_fold.flatten(), degree, col_means=col_means, col_stds=col_stds
            )
            # --- OLS ---
            theta_OLS = OLS_parameters(X_train_fold, y_train_fold)
            y_pred_OLS = X_val_fold @ theta_OLS
            mse_list_OLS.append(MSE(y_val_fold, y_pred_OLS))
            # --- Ridge ---
            theta_Ridge = Ridge_parameters(X_train_fold, y_train_fold, lam=lam_ridge)
            y_pred_Ridge = X_val_fold @ theta_Ridge
            mse_list_Ridge.append(MSE(y_val_fold, y_pred_Ridge))
            # --- LASSO (GD) ---
            theta_Lasso = Gradient_descent_advanced(
                X_train_fold, y_train_fold, Type=2, lam=lam_lasso, n_iter=5000, method='vanilla', lr=0.01
            )
            y_pred_Lasso = X_val_fold @ theta_Lasso
            mse_list_Lasso.append(MSE(y_val_fold, y_pred_Lasso))
        # Average across folds for this run
        di = degree - 1
        mse_cv_OLS_runs  [di, r] = np.mean(mse_list_OLS)
        mse_cv_Ridge_runs[di, r] = np.mean(mse_list_Ridge)
        mse_cv_Lasso_runs[di, r] = np.mean(mse_list_Lasso)


# Aggregate across runs: mean + std (append *_std)
save_vector_with_degree(
    TAB / "part_h_cv_mse_ols.csv",
    mse_cv_OLS_runs.mean(axis=1),
    "CV_MSE_OLS",
    std=mse_cv_OLS_runs.std(axis=1, ddof=1)
)
save_vector_with_degree(
    TAB / "part_h_cv_mse_ridge.csv",
    mse_cv_Ridge_runs.mean(axis=1),
    "CV_MSE_Ridge(lam=0.01)",
    std=mse_cv_Ridge_runs.std(axis=1, ddof=1)
)
save_vector_with_degree(
    TAB / "part_h_cv_mse_lasso.csv",
    mse_cv_Lasso_runs.mean(axis=1),
    "CV_MSE_LASSO(lam=0.01)",
    std=mse_cv_Lasso_runs.std(axis=1, ddof=1)
)


print("Part H done.")
