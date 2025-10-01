# -----------------------------------------------------------------------------------------
# Part e: Gradient descent comparison for OLS, Ridge, and LASSO
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    Gradient_descent_advanced, MSE, save_vector_with_degree, seed
)
from tqdm import trange
# -----------------------------
# Settings
# -----------------------------
n_points = [100]        # fixed dataset size
max_degree = 15
noise = True
lam_ridge = 0.01         # Ridge regularization
lam_lasso = 0.01        # LASSO regularization
learning_rates = [0.0001, 0.001, 0.01]  # learning rates for gradient descent
n_iter = 1000           # more iterations for convergence
methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
# 1 run with vanilla method with lr = [0.0001, 0.001, 0.01]
# 1 run with lr = 0.01 and methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
N_RUNS = 20
LR_FOR_METHODS = 0.01    

# -----------------------------
# Storage
# -----------------------------
models = {
    "OLS":   {"Type": 0, "lam": 0.0},
    "Ridge": {"Type": 1, "lam": lam_ridge},
    "LASSO": {"Type": 2, "lam": lam_lasso},
}
# Build only the combos requested in the comments (deduplicated)
combos = set()
for model in models.keys():
    for lr in learning_rates:
        combos.add((model, "vanilla", float(lr)))
for model in models.keys():
    for m in methods:
        combos.add((model, m, float(LR_FOR_METHODS)))
# Allocate storage for each combo: (degree, n_points, runs)
results_runs = {cmb: np.zeros((max_degree, len(n_points), N_RUNS)) for cmb in combos}

# Output dirs
OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

print(f">>> Starting Part E: GD comparison (OLS / Ridge / LASSO) | runs={N_RUNS} | "
      f"n_points={n_points} | max_degree={max_degree} | noise={noise}")
# print(">>> Seed policy: for each run r we set np.random.seed(seed+r) and use random_state=seed+r in the split,")
# print(">>> so within a run the dataset/split are IDENTICAL across all models/methods/lrs (fair comparisons).")

# -----------------------------
# Experiment
# -----------------------------
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # Fix the run seed -> same noise realization and split for all combos in this run
    np.random.seed(seed + r)
    for j, n in enumerate(n_points):
        # One dataset per run, shared across all combos
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)
        # Consistent split for all combos in this run
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)
        for degree in range(1, max_degree + 1):
            # Fit scaler on train once per degree; reuse for all combos
            X_train, col_means, col_stds = polynomial_features_scaled(
                x_train.flatten(), degree, return_stats=True
            )
            X_test = polynomial_features_scaled(
                x_test.flatten(), degree, col_means=col_means, col_stds=col_stds
            )
            di = degree - 1
            # Evaluate only the requested combos
            for (model, method, lr) in combos:
                cfg = models[model]
                theta = Gradient_descent_advanced(
                    X_train, y_train,
                    Type=cfg["Type"],
                    lam=cfg["lam"],
                    lr=lr,
                    n_iter=n_iter,
                    method=method,
                    theta_history=False
                )
                y_test_pred = X_test @ theta
                results_runs[(model, method, lr)][di, j, r] = MSE(y_test, y_test_pred)



# Save tables (one CSV per (model,method,lr))

idx0 = 0  # since n_points = [100], pick its column
for (model, method, lr), arr in results_runs.items():
    mean_vec = arr.mean(axis=2)[:, idx0]
    std_vec  = arr.std(axis=2, ddof=1)[:, idx0]
    save_vector_with_degree(
        TAB / f"part_e_{model.lower()}_mse_{method}_lr={lr}.csv",
        mean_vec,
        f"MSE_{model}_{method}_lr={lr}",
        std=std_vec
    )

print(f"Part E done. Aggregated over {N_RUNS} runs.")
