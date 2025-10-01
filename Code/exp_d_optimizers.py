# -----------------------------------------------------------------------------------------
# Part d, OLS and Ridge: compare analytical vs advanced gradient descent methods
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, save_vector_with_degree, save_matrix_with_degree_cols_plus_std, seed
)
from tqdm import trange


# -----------------------------
# Settings
# -----------------------------
n_points = [100]   # fix one dataset size
max_degree = 15
noise = True
lam = 0.01
learning_rates = [0.01]      # usare sempre l'indice 0
methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
n_iter = 1000
N_RUNS = 20                  

# -----------------------------
# Storage per-run (run x degree)
# -----------------------------
mse_analytical_ols_runs   = np.zeros((N_RUNS, max_degree))
mse_analytical_ridge_runs = np.zeros((N_RUNS, max_degree))
mse_gd_ols_runs   = {method: np.zeros((N_RUNS, max_degree)) for method in methods}
mse_gd_ridge_runs = {method: np.zeros((N_RUNS, max_degree)) for method in methods}

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Experiment con N_RUNS
# -----------------------------
print(f">>> Starting Part D: analytical vs advanced GD | runs={N_RUNS} | n_points={n_points} | "
      f"max_degree={max_degree} | lam={lam} | lr={learning_rates[0]}")
# print(">>> Seed policy: for each run r we set np.random.seed(seed+r) and split_scale(..., random_state=seed+r)")
# print(">>> so within a run the dataset/split are IDENTICAL across OLS/Ridge and all GD methods (fair comparison).")
LR = learning_rates[0]
for run_idx in trange(N_RUNS, desc="Runs", unit="run"):
    # Fix run seed -> same noise & same split for all models in this run
    np.random.seed(seed + run_idx)
    for j, n in enumerate(n_points):
        # Generate Runge data (different per run; shared across models)
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)

        # Split & scale (consistent per run)
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + run_idx)

        for degree in range(1, max_degree+1):
            # Polynomial features
            X_train, col_means, col_stds = polynomial_features_scaled(
                x_train.flatten(), degree, return_stats=True
            )
            X_test = polynomial_features_scaled(
                x_test.flatten(), degree, col_means=col_means, col_stds=col_stds
            )

            # ----- Analytical OLS -----
            theta_ols = OLS_parameters(X_train, y_train)
            y_test_pred = X_test @ theta_ols
            mse_analytical_ols_runs[run_idx, degree-1] = MSE(y_test, y_test_pred)

            # ----- Analytical Ridge -----
            theta_ridge = Ridge_parameters(X_train, y_train, lam=lam, intercept=True)
            y_test_pred = X_test @ theta_ridge
            mse_analytical_ridge_runs[run_idx, degree-1] = MSE(y_test, y_test_pred)

            # ----- Gradient Descent OLS (all methods) -----
            for method in methods:
                theta_gd = Gradient_descent_advanced(
                    X_train, y_train, Type=0, lr=LR, n_iter=n_iter,
                    method=method, lam=0.0, theta_history=False
                )
                y_test_pred_gd = X_test @ theta_gd
                mse_gd_ols_runs[method][run_idx, degree-1] = MSE(y_test, y_test_pred_gd)

            # ----- Gradient Descent Ridge (all methods) -----
            for method in methods:
                theta_gd = Gradient_descent_advanced(
                    X_train, y_train, Type=1, lam=lam, lr=learning_rates[0], n_iter=n_iter,
                    method=method, theta_history=False
                )
                y_test_pred_gd = X_test @ theta_gd
                mse_gd_ridge_runs[method][run_idx, degree-1] = MSE(y_test, y_test_pred_gd)


# -----------------------------
# Saving: mean + std su N_RUNS per degree
# -----------------------------
m_ols   = mse_analytical_ols_runs.mean(axis=0)
s_ols   = mse_analytical_ols_runs.std(axis=0, ddof=1)
m_ridge = mse_analytical_ridge_runs.mean(axis=0)
s_ridge = mse_analytical_ridge_runs.std(axis=0, ddof=1)

# Saves: original column name = mean; appended column = *_std (no structure change)
save_vector_with_degree(
    TAB / "part_d_analytical_ols.csv",
    m_ols, "MSE_analytical_OLS", std=s_ols
)
save_vector_with_degree(
    TAB / "part_d_analytical_ridge.csv",
    m_ridge, "MSE_analytical_Ridge", std=s_ridge
)


for method in methods:
    m_gd_ols   = mse_gd_ols_runs[method].mean(axis=0)
    s_gd_ols   = mse_gd_ols_runs[method].std(axis=0, ddof=1)
    m_gd_ridge = mse_gd_ridge_runs[method].mean(axis=0)
    s_gd_ridge = mse_gd_ridge_runs[method].std(axis=0, ddof=1)

    save_vector_with_degree(
        TAB / f"part_d_ols_mse_{method}_lr={LR}.csv",
        m_gd_ols, f"MSE_GD_OLS_{method}", std=s_gd_ols
    )
    save_vector_with_degree(
        TAB / f"part_d_ridge_mse_{method}_lr={LR}.csv",
        m_gd_ridge, f"MSE_GD_Ridge_{method}", std=s_gd_ridge
    )

print(f"Part D done. Aggregated over {N_RUNS} runs.")
