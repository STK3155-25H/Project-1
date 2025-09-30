# -----------------------------------------------------------------------------------------
# Part g, Bias-Variance analysis for polynomials to degree 15, different amount of data points
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, MSE_Bias_Variance, save_vector_with_degree
)

# -----------------------------
# Settings
# -----------------------------
n_points_list = [40, 50, 100, 500, 1000]   # Different dataset sizes
max_degree = 15
noise = True
n_bootstrap = 50   # Number of bootstrap samples

# -----------------------------
# Storage arrays
# -----------------------------
mse_list     = np.zeros((max_degree, len(n_points_list)))
bias2_list   = np.zeros((max_degree, len(n_points_list)))
variance_list= np.zeros((max_degree, len(n_points_list)))

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Experiment
# -----------------------------
for j, n_points in enumerate(n_points_list):
    # Generate Runge data
    x = np.linspace(-1, 1, n_points)
    y = runge_function(x, noise=noise)

    # Split & scale
    x_train, x_test, y_train, y_test = split_scale(x, y)
    
    for degree in range(1, max_degree+1):
        # Make the predictions and targets lists
        predictions = np.zeros((n_bootstrap, len(x_test)))
        targets     = np.zeros((n_bootstrap, len(x_test)))

        X_train, col_means, col_stds = polynomial_features_scaled(x_train.flatten(), degree, return_stats=True)
        X_test = polynomial_features_scaled(x_test.flatten(), degree, col_means=col_means, col_stds=col_stds)
        
        for i in range(n_bootstrap):
            # Resample data with replacement
            X_train_re, y_train_re = resample(X_train, y_train)
            
            # Fit OLS on resampled data
            theta = OLS_parameters(X_train_re, y_train_re)
            
            # Make predictions
            predictions[i] = X_test @ theta

        # Compute MSE, bias^2, and variance using the helper function
        mse, bias2, var = MSE_Bias_Variance(y_test, predictions)

        # Store metrics
        mse_list    [degree-1, j] = mse
        bias2_list  [degree-1, j] = bias2
        variance_list[degree-1, j]= var

# -----------------------------
# Plot results
# -----------------------------
for j, n_points in enumerate(n_points_list):
    plt.figure(figsize=(12,6))
    plt.plot(range(1, max_degree+1), mse_list[:, j],     label='MSE')
    plt.plot(range(1, max_degree+1), bias2_list[:, j],   label='Bias^2')
    plt.plot(range(1, max_degree+1), variance_list[:, j],label='Variance')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.title(f'Bias-Variance Trade-Off, n={n_points}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG / f"part_g_bias_variance_n={n_points}.png", dpi=150)

# Save per-n tables
for idx, n in enumerate(n_points_list):
    save_vector_with_degree(TAB / f"part_g_mse_n={n}.csv",
                            mse_list[:, idx],      f"MSE_n={n}")
    save_vector_with_degree(TAB / f"part_g_bias2_n={n}.csv",
                            bias2_list[:, idx],    f"Bias2_n={n}")
    save_vector_with_degree(TAB / f"part_g_variance_n={n}.csv",
                            variance_list[:, idx], f"Variance_n={n}")
print("Part G done.")
