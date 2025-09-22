# -----------------------------------------------------------------------------------------
# Part a, The OLS experiment for polynomials to degree 15 and different amount of data points
# -----------------------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, MSE, R2_score
)

# -----------------------------
# Settings
# -----------------------------
n_points = [40, 50, 100, 500, 1000]
max_degree = 15
noise = True

# -----------------------------
# Storage
# -----------------------------
mse_train_list = np.zeros((max_degree, len(n_points)))
mse_test_list  = np.zeros((max_degree, len(n_points)))
R2_train_list  = np.zeros((max_degree, len(n_points)))
R2_test_list   = np.zeros((max_degree, len(n_points)))
theta_list     = [[None for _ in range(len(n_points))] for _ in range(max_degree)]

# Make output dirs
OUT = Path("Code/outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

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
    
        # Analytical OLS
        theta = OLS_parameters(X_train, y_train)
    
        # Predictions
        y_train_pred = X_train @ theta
        y_test_pred  = X_test  @ theta
    
        # Store metrics
        mse_train_list[degree-1, j] = MSE(y_train, y_train_pred)
        mse_test_list [degree-1, j] = MSE(y_test,  y_test_pred)
        R2_train_list [degree-1, j] = R2_score(y_train, y_train_pred)
        R2_test_list  [degree-1, j] = R2_score(y_test,  y_test_pred)
        theta_list[degree-1][j]     = theta

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for j, n in enumerate(n_points):
    plt.plot(range(1, max_degree+1), mse_test_list[:, j], label=f'n={n}')
plt.xlabel('Polynomial degree')
plt.ylabel('MSE (Test)')
plt.title('Test MSE vs Polynomial degree')
plt.legend()

plt.subplot(1,2,2)
for j, n in enumerate(n_points):
    plt.plot(range(1, max_degree+1), R2_test_list[:, j], label=f'n={n}')
plt.xlabel('Polynomial degree')
plt.ylabel('R2 score (Test)')
plt.title('Test R2 vs Polynomial degree')
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "part_a_mse_r2.png", dpi=150)

# Plot the theta value vs degree (for largest n: index 4)
plt.figure(figsize=(12,6))
for degree in range(max_degree):
    plt.plot(theta_list[degree][4], label=f'degree {degree+1}')
plt.xlabel('Theta index')
plt.ylabel('Theta value')
plt.title('OLS Parameters vs Polynomial degree')
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "part_a_thetas.png", dpi=150)

# Save tables
np.savetxt(TAB / "part_a_mse_test.csv", mse_test_list, delimiter=",")
np.savetxt(TAB / "part_a_r2_test.csv",  R2_test_list,  delimiter=",")
print("Part A done.")
