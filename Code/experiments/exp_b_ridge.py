# -----------------------------------------------------------------------------------------
# Part b, The Ridge experiment for polynomials to degree 15, different amount of data points and different lambda values
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Code.src.ml_core import (
    runge_function, split_scale, polynomial_features_scaled,
    Ridge_parameters, MSE, R2_score
)

# -----------------------------
# Settings
# -----------------------------
n_points = [40, 50, 100, 500, 1000]
lam_list = [0, 0.001, 0.01, 0.1, 1, 10, 100]
max_degree = 15
noise = True

# -----------------------------
# Storage
# -----------------------------
mse_train_list = np.zeros((max_degree, len(n_points), len(lam_list)))
mse_test_list  = np.zeros((max_degree, len(n_points), len(lam_list)))
R2_train_list  = np.zeros((max_degree, len(n_points), len(lam_list)))
R2_test_list   = np.zeros((max_degree, len(n_points), len(lam_list)))
theta_list     = [[[None for _ in range(len(lam_list))] for _ in range(len(n_points))] for _ in range(max_degree)]

# Output dirs
OUT = Path("Code/outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Experiment
# -----------------------------
for k, lam in enumerate(lam_list):
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
    
            # Analytical Ridge
            theta = Ridge_parameters(X_train, y_train, lam)
    
            # Predictions
            y_train_pred = X_train @ theta
            y_test_pred  = X_test  @ theta
    
            # Store metrics
            mse_train_list[degree-1, j, k] = MSE(y_train, y_train_pred)
            mse_test_list [degree-1, j, k] = MSE(y_test,  y_test_pred)
            R2_train_list [degree-1, j, k] = R2_score(y_train, y_train_pred)
            R2_test_list  [degree-1, j, k] = R2_score(y_test,  y_test_pred)
            theta_list[degree-1][j][k]     = theta

    # -----------------------------
    # Plot results (per lambda)
    # -----------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for j, n in enumerate(n_points):
        plt.plot(range(1, max_degree+1), mse_test_list[:, j, k], label=f'n={n}')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE (Test)')
    plt.title('Test MSE vs Polynomial degree, with lambda = ' + str(lam))
    plt.legend()

    plt.subplot(1,2,2)
    for j, n in enumerate(n_points):
        plt.plot(range(1, max_degree+1), R2_test_list[:, j, k], label=f'n={n}')
    plt.xlabel('Polynomial degree')
    plt.ylabel('R2 score (Test)')
    plt.title('Test R2 vs Polynomial degree, with lambda = ' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"part_b_mse_r2_lambda={lam}.png", dpi=150)

    plt.figure(figsize=(12,6))
    for degree in range(max_degree):
        plt.plot(theta_list[degree][4][k], label=f'degree {degree+1}')
    plt.xlabel('Theta index')
    plt.ylabel('Theta value')
    plt.title('Ridge Parameters vs Polynomial degree, with lambda = ' + str(lam))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"part_b_thetas_lambda={lam}.png", dpi=150)

    # Save tables per lambda (test metrics)
    np.savetxt(TAB / f"part_b_mse_test_lambda={lam}.csv", mse_test_list[:, :, k], delimiter=",")
    np.savetxt(TAB / f"part_b_r2_test_lambda={lam}.csv",  R2_test_list[:,  :, k], delimiter=",")

print("Part B done.")
