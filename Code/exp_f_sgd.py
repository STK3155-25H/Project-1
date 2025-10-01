# -----------------------------------------------------------------------------------------
# Part f: Stochastic Gradient Descent for OLS, Ridge, and LASSO
# -----------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import (
    runge_function, split_scale, polynomial_features_scaled,
    Gradient_descent_advanced, MSE, save_vector_with_degree,seed
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
learning_rates = [0.01]  # learning rates
n_iter = 1000           # iterations
batch_size = 10         # mini-batch size
methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
N_RUNS = 30

# -----------------------------
# Storage
# -----------------------------
models = {
    "OLS":   {"Type": 0, "lam": 0.0},
    "Ridge": {"Type": 1, "lam": lam_ridge},
    "LASSO": {"Type": 2, "lam": lam_lasso},
}
combos = {(model, method, float(lr)) for model in models for method in methods for lr in learning_rates}
# Each entry: (degree, n_points, runs)
results_runs = {cmb: np.zeros((max_degree, len(n_points), N_RUNS)) for cmb in combos}


# Output dirs (root-level 'outputs' to match your tree)
OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
LOG = OUT / "logs"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)


print(f">>> Starting Part F: SGD (OLS / Ridge / LASSO) | runs={N_RUNS} | "
      f"n_points={n_points} | max_degree={max_degree} | noise={noise} | batch_size={batch_size}")
# print(">>> Seed policy: per run r we set np.random.seed(seed+r) and use random_state=seed+r in the split;")
# print(">>> for SGD fairness we also reset a per-degree sgd_seed before each optimizer call so minibatch sequences match.")


# -----------------------------
# Experiment
# -----------------------------
for r in trange(N_RUNS, desc="Runs", unit="run"):
    # Same noise and same split for all combos in this run
    np.random.seed(seed + r)
    for j, n in enumerate(n_points):
        x = np.linspace(-1, 1, n)
        y = runge_function(x, noise=noise)
        x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)
        for degree in range(1, max_degree + 1):
            # Feature scaling fitted on train and reused for test
            X_train, col_means, col_stds = polynomial_features_scaled(
                x_train.flatten(), degree, return_stats=True
            )
            X_test = polynomial_features_scaled(
                x_test.flatten(), degree, col_means=col_means, col_stds=col_stds
            )
            di = degree - 1
            # Fix minibatch RNG per degree to ensure identical SGD batches across models/methods/lrs
            sgd_seed_base = (seed + r) * 1_000_000 + j * 1_000 + degree
            for (model, method, lr) in combos:
                np.random.seed(sgd_seed_base)  # identical minibatch sequence for every combo
                cfg = models[model]
                theta = Gradient_descent_advanced(
                    X_train, y_train,
                    Type=cfg["Type"],
                    lam=cfg["lam"],
                    lr=lr,
                    n_iter=n_iter,
                    method=method,
                    theta_history=False,
                    use_sgd=True,
                    batch_size=batch_size
                )
                y_test_pred = X_test @ theta
                results_runs[(model, method, lr)][di, j, r] = MSE(y_test, y_test_pred)


# Save tables 
idx0 = 0  # since n_points = [100]
for (model, method, lr), arr in results_runs.items():
    mean_vec = arr.mean(axis=2)[:, idx0]
    std_vec  = arr.std(axis=2, ddof=1)[:, idx0]
    save_vector_with_degree(
        TAB / f"part_f_{model.lower()}_mse_{method}_lr={lr}_sgd.csv",
        mean_vec,
        f"MSE_{model}_{method}_lr={lr}_sgd",
        std=std_vec
    )
print(f"Part F done. Aggregated over {N_RUNS} runs.")
