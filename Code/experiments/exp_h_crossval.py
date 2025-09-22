# -----------------------------------------------------------------------------------------
# Part h: Cross-Validation with OLS, Ridge, and Lasso
# -----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from Code.src.ml_core import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, Ridge_parameters, Gradient_descent_advanced, MSE, seed
)

# -----------------------------
# Settings
# -----------------------------
n_points = 100
max_degree = 15
noise = True
k_folds = 5
lam_ridge = 0.01
lam_lasso = 0.01

# Output dirs
OUT = Path("Code/outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# Generate Runge data
x = np.linspace(-1, 1, n_points)
y = runge_function(x, noise=noise)

# Split and scale data
x_train, x_test, y_train, y_test = split_scale(x, y)

# Storage arrays for CV MSE
mse_cv_OLS   = np.zeros(max_degree)
mse_cv_Ridge = np.zeros(max_degree)
mse_cv_Lasso = np.zeros(max_degree)

# KFold object
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

# -----------------------------
# Loop over polynomial degrees
# -----------------------------
for degree in range(1, max_degree+1):
    mse_list_OLS = []
    mse_list_Ridge = []
    mse_list_Lasso = []

    for train_idx, val_idx in kf.split(x_train):
        # Get fold data
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Polynomial features and scaling
        X_train_fold, col_means, col_stds = polynomial_features_scaled(x_train_fold.flatten(), degree, return_stats=True)
        X_val_fold = polynomial_features_scaled(x_val_fold.flatten(), degree, col_means=col_means, col_stds=col_stds)

        # --- OLS ---
        theta_OLS = OLS_parameters(X_train_fold, y_train_fold)
        y_pred_OLS = X_val_fold @ theta_OLS
        mse_list_OLS.append(MSE(y_val_fold, y_pred_OLS))

        # --- Ridge ---
        theta_Ridge = Ridge_parameters(X_train_fold, y_train_fold, lam=lam_ridge)
        y_pred_Ridge = X_val_fold @ theta_Ridge
        mse_list_Ridge.append(MSE(y_val_fold, y_pred_Ridge))

        # --- Lasso (using gradient descent) ---
        theta_Lasso = Gradient_descent_advanced(X_train_fold, y_train_fold, Type=2, lam=lam_lasso, n_iter=5000)
        y_pred_Lasso = X_val_fold @ theta_Lasso
        mse_list_Lasso.append(MSE(y_val_fold, y_pred_Lasso))

    # Average MSE over folds
    mse_cv_OLS  [degree-1] = np.mean(mse_list_OLS)
    mse_cv_Ridge[degree-1] = np.mean(mse_list_Ridge)
    mse_cv_Lasso[degree-1] = np.mean(mse_list_Lasso)

# -----------------------------
# Plot CV results
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, max_degree+1), mse_cv_OLS,   label='OLS')
plt.plot(range(1, max_degree+1), mse_cv_Ridge, label=f'Ridge (λ={lam_ridge})')
plt.plot(range(1, max_degree+1), mse_cv_Lasso, label=f'Lasso (λ={lam_lasso})')
plt.xlabel('Polynomial Degree')
plt.ylabel(f'{k_folds}-Fold CV MSE')
plt.title('Cross-Validation MSE vs Polynomial Degree')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG / "part_h_cv_mse.png", dpi=150)

# Save tables (both separate and combined)
np.savetxt(TAB / "part_h_cv_mse_ols.csv",   mse_cv_OLS,   delimiter=",")
np.savetxt(TAB / "part_h_cv_mse_ridge.csv", mse_cv_Ridge, delimiter=",")
np.savetxt(TAB / "part_h_cv_mse_lasso.csv", mse_cv_Lasso, delimiter=",")

# Combined (with a header row written manually)
with open(TAB / "part_h_cv_mse.csv", "w") as f:
    f.write("degree,OLS,Ridge(lam=0.01),LASSO(lam=0.01)\n")
    for d in range(1, max_degree+1):
        f.write(f"{d},{mse_cv_OLS[d-1]},{mse_cv_Ridge[d-1]},{mse_cv_Lasso[d-1]}\n")

print("Part H done.")
