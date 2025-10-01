# -----------------------------------------------------------------------------------------
# Part d, OLS and Ridge: compare analytical vs advanced gradient descent methods
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, save_vector_with_degree,save_matrix_with_degree_cols
)
from tqdm import tqdm


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
print("Progress on point D")
for run_idx in tqdm(range(N_RUNS)):
    for j, n in enumerate(n_points):
        # Generate Runge data (rumore diverso a ogni run)
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)

        # Split & scale
        x_train, x_test, y_train, y_test = split_scale(x, y)

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
                    X_train, y_train, Type=0, lr=learning_rates[0], n_iter=n_iter,
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
# Aggregazione: mean + std su N_RUNS per degree
# -----------------------------
m_ols   = mse_analytical_ols_runs.mean(axis=0)
s_ols   = mse_analytical_ols_runs.std(axis=0, ddof=1)
m_ridge = mse_analytical_ridge_runs.mean(axis=0)
s_ridge = mse_analytical_ridge_runs.std(axis=0, ddof=1)

# Salvataggi: stessa colonna per la media (nome invariato) + nuova colonna *_std
save_matrix_with_degree_cols(
    TAB / "part_d_analytical_ols.csv",
    np.column_stack([m_ols, s_ols]),
    col_names=("MSE_analytical_OLS", "MSE_analytical_OLS_std")
)
save_matrix_with_degree_cols(
    TAB / "part_d_analytical_ridge.csv",
    np.column_stack([m_ridge, s_ridge]),
    col_names=("MSE_analytical_Ridge", "MSE_analytical_Ridge_std")
)

for method in methods:
    m_gd_ols   = mse_gd_ols_runs[method].mean(axis=0)
    s_gd_ols   = mse_gd_ols_runs[method].std(axis=0, ddof=1)
    m_gd_ridge = mse_gd_ridge_runs[method].mean(axis=0)
    s_gd_ridge = mse_gd_ridge_runs[method].std(axis=0, ddof=1)

    save_matrix_with_degree_cols(
        TAB / f"part_d_ols_mse_{method}_lr={learning_rates[0]}.csv",
        np.column_stack([m_gd_ols, s_gd_ols]),
        col_names=(f"MSE_GD_OLS_{method}", f"MSE_GD_OLS_{method}_std")
    )
    save_matrix_with_degree_cols(
        TAB / f"part_d_ridge_mse_{method}_lr={learning_rates[0]}.csv",
        np.column_stack([m_gd_ridge, s_gd_ridge]),
        col_names=(f"MSE_GD_Ridge_{method}", f"MSE_GD_Ridge_{method}_std")
    )

print(f"Part D done. Aggregated over {N_RUNS} runs.")
