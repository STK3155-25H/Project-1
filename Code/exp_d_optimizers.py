# -----------------------------------------------------------------------------------------
# Part d, OLS and Ridge: compare analytical vs advanced gradient descent methods
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE
)

# -----------------------------
# Settings
# -----------------------------
n_points = [100]   # fix one dataset size
max_degree = 15
noise = True
lam = 0.1          # Ridge regularization
learning_rates = [0.001, 0.01]  # reasonable η for advanced methods
methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
n_iter = 2000

# -----------------------------
# Storage
# -----------------------------
mse_analytical_ols   = np.zeros((max_degree, len(n_points)))
mse_analytical_ridge = np.zeros((max_degree, len(n_points)))
mse_gd_ols   = {method: np.zeros((max_degree, len(n_points))) for method in methods}
mse_gd_ridge = {method: np.zeros((max_degree, len(n_points))) for method in methods}

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Experiment
# -----------------------------
for j, n in enumerate(n_points):
    # Generate Runge data
    x = np.linspace(-1, 1, n)
    y = runge_function(x, noise=noise)

    # Split & scale
    x_train, x_test, y_train, y_test = split_scale(x, y)

    for degree in range(1, max_degree+1):
        # Polynomial features
        X_train, col_means, col_stds = polynomial_features_scaled(x_train.flatten(), degree, return_stats=True)
        X_test = polynomial_features_scaled(x_test.flatten(), degree, col_means=col_means, col_stds=col_stds)

        # ----- Analytical OLS -----
        theta_ols = OLS_parameters(X_train, y_train)
        y_test_pred = X_test @ theta_ols
        mse_analytical_ols[degree-1, j] = MSE(y_test, y_test_pred)

        # ----- Analytical Ridge -----
        theta_ridge = Ridge_parameters(X_train, y_train, lam=lam, intercept=True)
        y_test_pred = X_test @ theta_ridge
        mse_analytical_ridge[degree-1, j] = MSE(y_test, y_test_pred)

        # ----- Gradient Descent OLS (all methods) -----
        for method in methods:
            theta_gd = Gradient_descent_advanced(
                X_train, y_train, Type=0, lr=learning_rates[1], n_iter=n_iter,
                method=method, lam=0.0, theta_history=False
            )
            y_test_pred_gd = X_test @ theta_gd
            mse_gd_ols[method][degree-1, j] = MSE(y_test, y_test_pred_gd)

        # ----- Gradient Descent Ridge (all methods) -----
        for method in methods:
            theta_gd = Gradient_descent_advanced(
                X_train, y_train, Type=1, lam=lam, lr=learning_rates[1], n_iter=n_iter,
                method=method, theta_history=False
            )
            y_test_pred_gd = X_test @ theta_gd
            mse_gd_ridge[method][degree-1, j] = MSE(y_test, y_test_pred_gd)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(range(1, max_degree+1), mse_analytical_ols[:,0], 'k--', label="Analytical OLS")
for method in methods:
    plt.plot(range(1, max_degree+1), mse_gd_ols[method][:,0], label=f"GD OLS ({method})")
plt.xlabel("Polynomial degree"); plt.ylabel("Test MSE")
plt.title("OLS: Analytical vs Gradient Descent"); plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, max_degree+1), mse_analytical_ridge[:,0], 'k--', label=f"Analytical Ridge (λ={lam})")
for method in methods:
    plt.plot(range(1, max_degree+1), mse_gd_ridge[method][:,0], label=f"GD Ridge ({method})")
plt.xlabel("Polynomial degree"); plt.ylabel("Test MSE")
plt.title(f"Ridge: Analytical vs Gradient Descent (λ={lam})"); plt.legend()
plt.tight_layout()
plt.savefig(FIG / "part_d_optimizers.png", dpi=150)

# Save tables per method (OLS & Ridge) at lr=learning_rates[1]
for method in methods:
    np.savetxt(TAB / f"part_d_ols_mse_{method}_lr={learning_rates[1]}.csv",   mse_gd_ols[method][:,0],   delimiter=",")
    np.savetxt(TAB / f"part_d_ridge_mse_{method}_lr={learning_rates[1]}.csv", mse_gd_ridge[method][:,0], delimiter=",")
np.savetxt(TAB / "part_d_analytical_ols.csv",   mse_analytical_ols[:,0],   delimiter=",")
np.savetxt(TAB / "part_d_analytical_ridge.csv", mse_analytical_ridge[:,0], delimiter=",")
print("Part D done.")
