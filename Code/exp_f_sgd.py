# -----------------------------------------------------------------------------------------
# Part f: Stochastic Gradient Descent for OLS, Ridge, and LASSO
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    Gradient_descent_advanced, MSE, save_vector_with_degree
)

# -----------------------------
# Settings
# -----------------------------
n_points = [100]        # fixed dataset size
max_degree = 15
noise = True
lam_ridge = 0.01         # Ridge regularization
lam_lasso = 0.01        # LASSO regularization
learning_rates = [0.01]  # learning rates
n_iter = 5000           # iterations
batch_size = 10         # mini-batch size
methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']

# -----------------------------
# Storage
# -----------------------------
mse_sgd_results = {
    "OLS":   {method: {lr: np.zeros((max_degree, len(n_points))) for lr in learning_rates} for method in methods},
    "Ridge": {method: {lr: np.zeros((max_degree, len(n_points))) for lr in learning_rates} for method in methods},
    "LASSO": {method: {lr: np.zeros((max_degree, len(n_points))) for lr in learning_rates} for method in methods}
}

# Output dirs (root-level 'outputs' to match your tree)
OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
LOG = OUT / "logs"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Experiment
# -----------------------------
for j, n in enumerate(n_points):
    # Generate Runge data
    x = np.linspace(-1, 1, n)
    y = runge_function(x, noise=noise)

    # Split & scale
    x_train, x_test, y_train, y_test = split_scale(x, y)

    for degree in range(1, max_degree + 1):
        # Generate polynomial features and scale
        X_train, col_means, col_stds = polynomial_features_scaled(x_train.flatten(), degree, return_stats=True)
        X_test = polynomial_features_scaled(x_test.flatten(), degree, col_means=col_means, col_stds=col_stds)

        for method in methods:
            for lr in learning_rates:
                # ----- OLS SGD -----
                theta_ols = Gradient_descent_advanced(
                    X_train, y_train, Type=0, lr=lr, n_iter=n_iter, method=method, theta_history=False,
                    use_sgd=True, batch_size=batch_size
                )
                y_test_pred = X_test @ theta_ols
                mse_sgd_results["OLS"][method][lr][degree-1, j] = MSE(y_test, y_test_pred)

                # ----- Ridge SGD -----
                theta_ridge = Gradient_descent_advanced(
                    X_train, y_train, Type=1, lam=lam_ridge, lr=lr, n_iter=n_iter, method=method,
                    theta_history=False, use_sgd=True, batch_size=batch_size
                )
                y_test_pred = X_test @ theta_ridge
                mse_sgd_results["Ridge"][method][lr][degree-1, j] = MSE(y_test, y_test_pred)

                # ----- LASSO SGD -----
                theta_lasso = Gradient_descent_advanced(
                    X_train, y_train, Type=2, lam=lam_lasso, lr=lr, n_iter=n_iter, method=method,
                    theta_history=False, use_sgd=True, batch_size=batch_size
                )
                y_test_pred = X_test @ theta_lasso
                mse_sgd_results["LASSO"][method][lr][degree-1, j] = MSE(y_test, y_test_pred)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(18, 6))

# --- OLS ---
plt.subplot(1, 3, 1)
for method in methods:
    for lr in learning_rates:
        plt.plot(range(1, max_degree + 1), mse_sgd_results["OLS"][method][lr][:, 0], label=f"{method} lr={lr}")
plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title("OLS: SGD Methods")
plt.legend()

# --- Ridge ---
plt.subplot(1, 3, 2)
for method in methods:
    for lr in learning_rates:
        plt.plot(range(1, max_degree + 1), mse_sgd_results["Ridge"][method][lr][:, 0], label=f"{method} lr={lr}")
plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title(f"Ridge: SGD Methods (λ={lam_ridge})")
plt.legend()

# --- LASSO ---
plt.subplot(1, 3, 3)
for method in methods:
    for lr in learning_rates:
        plt.plot(range(1, max_degree + 1), mse_sgd_results["LASSO"][method][lr][:, 0], label=f"{method} lr={lr}")
plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title(f"LASSO: SGD Methods (λ={lam_lasso})")
plt.legend()

plt.tight_layout()
plt.savefig(FIG / "part_f_ols_ridge_lasso_sgd.png", dpi=150)

# Save tables (one CSV per (model, method, lr))
for model in ["OLS", "Ridge", "LASSO"]:
    for method in methods:
        for lr in learning_rates:
            save_vector_with_degree(
                TAB / f"part_f_{model.lower()}_mse_{method}_lr={lr}_sgd.csv",
                mse_sgd_results[model][method][lr][:, 0],
                f"MSE_{model}_{method}_lr={lr}_sgd"
            )
print("Part F done.")
