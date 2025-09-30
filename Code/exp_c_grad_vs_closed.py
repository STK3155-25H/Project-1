# -----------------------------------------------------------------------------------------
# Part c, The OLS and Ridge experiment for polynomials to degree 15 comparing analytical method to gradient descent
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, save_vector_with_degree
)

# -----------------------------
# Settings
# -----------------------------
n_points = [100]   # fix one dataset size (simpler comparison)
max_degree = 15
noise = True
lam = 0.01          # Ridge regularization parameter
learning_rates = [0.0001, 0.001, 0.01]  # try several η values
n_iter = 2000

# -----------------------------
# Storage
# -----------------------------
mse_analytical_ols   = np.zeros((max_degree, len(n_points)))
mse_analytical_ridge = np.zeros((max_degree, len(n_points)))
mse_gd_ols   = {eta: np.zeros((max_degree, len(n_points))) for eta in learning_rates}
mse_gd_ridge = {eta: np.zeros((max_degree, len(n_points))) for eta in learning_rates}

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
        # Generate polynomial features and scale
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
    
        # ----- Gradient Descent OLS -----
        for eta in learning_rates:
            theta_gd = Gradient_descent_advanced(X_train, y_train, Type=0, lr=eta, n_iter=n_iter, method='vanilla', lam=0.0, theta_history=False)
            y_test_pred_gd = X_test @ theta_gd
            mse_gd_ols[eta][degree-1, j] = MSE(y_test, y_test_pred_gd)

        # ----- Gradient Descent Ridge -----
        for eta in learning_rates:
            theta_gd = Gradient_descent_advanced(X_train, y_train, Type=1, lam=lam, lr=eta, n_iter=n_iter, method='vanilla', theta_history=False)
            y_test_pred_gd = X_test @ theta_gd
            mse_gd_ridge[eta][degree-1, j] = MSE(y_test, y_test_pred_gd)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(range(1, max_degree+1), mse_analytical_ols[:, 0], 'k--', label="Analytical OLS")
for eta in learning_rates:
    plt.plot(range(1, max_degree+1), mse_gd_ols[eta][:, 0], label=f"GD OLS (lr={eta})")
plt.xlabel("Polynomial degree"); plt.ylabel("Test MSE")
plt.title("OLS: Analytical vs Gradient Descent"); plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, max_degree+1), mse_analytical_ridge[:, 0], 'k--', label=f"Analytical Ridge (λ={lam})")
for eta in learning_rates:
    plt.plot(range(1, max_degree+1), mse_gd_ridge[eta][:, 0], label=f"GD Ridge (lr={eta})")
plt.xlabel("Polynomial degree"); plt.ylabel("Test MSE")
plt.title(f"Ridge: Analytical vs Gradient Descent (λ={lam})"); plt.legend()
plt.tight_layout()
plt.savefig(FIG / "part_c_mse_analytical_vs_gd.png", dpi=150)

# Save tables
save_vector_with_degree(TAB / "part_c_mse_analytical_ols.csv",
                        mse_analytical_ols[:, 0], "MSE_analytical_OLS")
save_vector_with_degree(TAB / "part_c_mse_analytical_ridge.csv",
                        mse_analytical_ridge[:, 0], "MSE_analytical_Ridge")

for eta in learning_rates:
    save_vector_with_degree(TAB / f"part_c_mse_gd_ols_lr={eta}.csv",
                            mse_gd_ols[eta][:, 0], f"MSE_GD_OLS_lr={eta}")
    save_vector_with_degree(TAB / f"part_c_mse_gd_ridge_lr={eta}.csv",
                            mse_gd_ridge[eta][:, 0], f"MSE_GD_Ridge_lr={eta}")

print("Part C done.")
