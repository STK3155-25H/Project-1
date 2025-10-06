[![Assertion Tests](https://github.com/STK3155-25H/Project-1/actions/workflows/assertion_tests.yml/badge.svg)](https://github.com/STK3155-25H/Project-1/actions/workflows/assertion_tests.yml)
# Runge Regression — OLS / Ridge / LASSO (Closed‑Form, GD, SGD)

Midterm project for benchmarking linear regression methods on the **Runge function**:

- OLS and Ridge (**closed form**)
- **Gradient Descent** (vanilla, Momentum, AdaGrad, RMSProp, Adam)
- **Stochastic Gradient Descent (SGD)** with mini-batches for OLS, Ridge, and LASSO
- **Bias–Variance** analysis
- **K‑Fold Cross‑Validation**

All experiments save **figures** and **CSV tables** ready to drop into the report.

---

## Group members

- **Francesco Giuseppe Minisini** — francegm@uio.no
- **Stan Daniels**
- **Carolina Ceccacci**
- **Teresa Ghirlandi**

## Short project description

We evaluate and compare OLS, Ridge, and LASSO on synthetic data generated from the Runge function. We investigate:

1. closed‑form solutions vs gradient descent,
2. different optimizers and learning rates,
3. full‑batch GD vs **SGD** with mini‑batches,
4. the **bias–variance** trade‑off and **cross‑validation**.

The goal is to produce reproducible figures and metrics tables (MSE, R²) for a paper‑style report.

---

## Repository structure

```bash
├── .gitignore
├── Code
│   ├── exp_a_ols.py
│   ├── exp_b_ridge.py
│   ├── exp_c_grad_vs_closed.py
│   ├── exp_d_optimizers.py
│   ├── exp_e_lasso.py
│   ├── exp_f_sgd.py
│   ├── exp_g_bias_variance.py
│   ├── exp_h_crossval.py
│   └── src
│       ├── __init__.py
│       ├── exp_benchmarks_metrics.py
│       ├── ml_core.py
│       └── tests
│           ├── __init__.py
│           └── tests.py
├── Exercises_week_39.pdf
├── Makefile
├── README.md
├── Report
│   ├── Graphs
│   │   ├── Data_Lists
│   │   │   ├── OLS_MSE.csv
│   │   │   └── OLS_R2.csv
│   │   ├── Explanation.tex
│   │   └── OLS_MSE.tex
│   ├── main.pdf
│   └── main.tex
└── requirements.txt
```

> **Important note**: `Code/` is a Python package. Run experiments as **modules** from the repo root (see below) to avoid import issues.

---

## Requirements

- Python **3.9+** (tested with 3.10/3.11)
- `pip`
- (Optional) **GNU Make** for convenient commands `make a`, `make all`, etc.

Install Python dependencies:

```bash
# Linux/macOS
python3 -m pip install -r Code/requirements.txt
# Windows
python -m pip install -r Code/requirements.txt
```

---

## Quick Start

**On Linux**If `make` is already installed, run:

```bash
make setup
make all
```

**On Windows**Use **WSL** and install `make`. If that’s not possible, install `make` on Windows (less optimal). Then change the Makefile global variable **PY** to:

```makefile
PY?=python
```

and run:

```bash
make setup
make all
```

If none of the previous options work and you can’t install `make`, install the required packages with:

```bash
pip install -r Code/requirements.txt
```

and run each experiment manually (see below).

---

## How to run the experiments

### Recommended (Python module mode — avoids import issues)

From the **repo root**:

```bash
# Linux/macOS
python3 -m Code.experiments.exp_a_ols
python3 -m Code.experiments.exp_b_ridge
python3 -m Code.experiments.exp_c_grad_vs_closed
python3 -m Code.experiments.exp_d_optimizers
python3 -m Code.experiments.exp_e_lasso
python3 -m Code.experiments.exp_f_sgd
python3 -m Code.experiments.exp_g_bias_variance
python3 -m Code.experiments.exp_h_crossval
```

```powershell
# Windows (PowerShell)
python -m Code.experiments.exp_a_ols
python -m Code.experiments.exp_b_ridge
python -m Code.experiments.exp_c_grad_vs_closed
python -m Code.experiments.exp_d_optimizers
python -m Code.experiments.exp_e_lasso
python -m Code.experiments.exp_f_sgd
python -m Code.experiments.exp_g_bias_variance
python -m Code.experiments.exp_h_crossval
```

Generic placeholder:

```bash
python Code/exp_letter_description.py
```

For this project, real paths are e.g.:

```bash
python Code/experiments/exp_a_ols.py
python Code/experiments/exp_f_sgd.py
# etc.
```

---

## How to run the benchmarks

To run the benchmarks, use the `make bench` command, which executes the `exp_benchmarks_metrics.py` script with predefined parameters and saves results to the `outputs/benchmarks/` directory. From the **repo root**:

```bash
make bench
```

This command:

- Runs a comprehensive benchmark with the parameters defined in the Makefile (e.g., `BENCH_NPOINTS`, `BENCH_MAX_DEGREE`, `METHODS`, etc.).
- Saves results to `outputs/benchmarks/tables/` (CSV files), `outputs/benchmarks/figures/` (PNG files), and `outputs/benchmarks/logs/` (log files with a JSON summary).

Alternatively, you can run the benchmark script manually:

```bash
# Linux/macOS
python3 -m Code.src.exp_benchmarks_metrics --n-points 100 --max-degree 14 --n-runs 30 --lam 0.01 --lr 0.01 --n-iter 10000 --tol 1e-8 --beta 0.9 --epsilon 1e-8 --batch-size 32 --bootstrap 30 --test-size 0.33 --methods ols ridge gd-vanilla gd-momentum gd-adam
```

```powershell
# Windows (PowerShell)
python -m Code.src.exp_benchmarks_metrics --n-points 100 --max-degree 14 --n-runs 30 --lam 0.01 --lr 0.01 --n-iter 10000 --tol 1e-8 --beta 0.9 --epsilon 1e-8 --batch-size 32 --bootstrap 30 --test-size 0.33 --methods ols ridge gd-vanilla gd-momentum gd-adam
```

To enable SGD with mini-batches, add the `--use-sgd` flag:

```bash
python3 -m Code.src.exp_benchmarks_metrics --use-sgd ...
```

### Modifying benchmark parameters

Benchmark parameters are defined in the Makefile under the "Config" section. To customize them, edit the following variables in the `Makefile`:

```makefile
# Example configuration in Makefile
BENCH_NPOINTS?=100
BENCH_MAX_DEGREE?=14
METHODS?=ols ridge gd-vanilla gd-momentum gd-adam
LAM?=0.01
LR?=0.01
N_ITER?=10000
TOL?=1e-8
BETA?=0.9
EPSILON?=1e-8
USE_SGD?=false
BATCH_SIZE?=32
BOOTSTRAP?=30
TEST_SIZE?=0.33
N_RUNS?=30
```

To override these without editing the Makefile, pass them as arguments to `make bench`:

make bench BENCH_NPOINTS=200 BENCH_MAX_DEGREE=10 METHODS="ols ridge" LAM=0.1

Alternatively, modify the parameters directly in the manual command:

```bash
python3 -m Code.src.exp_benchmarks_metrics --n-points 200 --max-degree 10 --methods ols ridge --lam 0.1
```

---

## Where results are saved

All scripts write to `outputs/`:

- **Figures**: `outputs/figures/*.png`
- **Tables**: `outputs/tables/*.csv`
- **Logs**: `outputs/logs/part_<letter>_YYYYMMDD-HHMMSS.log` (with a final JSON summary)
- **Benchmarks**: `outputs/benchmarks/tables/*.csv`, `outputs/benchmarks/figures/*.png`, `outputs/benchmarks/logs/*.log`

---

## Makefile usage (optional)

- `make setup` — install Python dependencies
- `make a|b|c|d|e|f|g|h` — run a single experiment part
- `make all` — run **all** parts a…h
- `make tables` — list generated tables
- `make tests` — runs all the tests
- `make bench` — runs all the benchmarks and places the results in `outputs/benchmarks` directory
- `make clean` — remove the `outputs/` directory

---

## Configuring the SEED

You can either:

1. **Edit** the hard‑coded value in `Code/src/ml_core.py` (variable `seed`), or
2. **Set an environment variable** `SEED` (if you choose to read it in code).

**On Linux/macOS**

```bash
export SEED=42
```

**On Windows (PowerShell)**

```powershell
$env:SEED = "42"
```

> To let the environment variable override the default seed, you may (optionally) tweak `ml_core.py`:
>
> ```python
> import os
> seed = int(os.getenv("SEED", 314))
> np.random.seed(seed)
> ```
