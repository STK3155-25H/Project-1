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
- **Francesco Giuseppe Minisini** — [francegm@uio.no](mailto:francegm@uio.no)
- **Stan Daniels**
- **Carolina Ceccacci**
- **Teresa Ghirlandi**

## Short project description
We evaluate and compare OLS, Ridge, and LASSO on synthetic data generated from the Runge function. We investigate:
1) closed‑form solutions vs gradient descent,
2) different optimizers and learning rates,
3) full‑batch GD vs **SGD** with mini‑batches,
4) the **bias–variance** trade‑off and **cross‑validation**.

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
│       └── ml_core.py
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

> **Import note**: `Code/` is a Python package. Run experiments as **modules** from the repo root (see below) to avoid import issues.

---

## Requirements

- Python **3.9+** (tested with 3.10/3.11)
- `pip`
- (Optional) **GNU Make** for convenient commands `make a`, `make all`, etc.

Install Python deps:
```bash
# Linux/macOS
python3 -m pip install -r Code/requirements.txt

# Windows
python -m pip install -r Code/requirements.txt
```

## Quick Start

**On Linux**  
If `make` is already installed, run:
```bash
make setup
make all
```

**On Windows**  
Use **WSL** and install `make`. If that’s not possible, install `make` on Windows (less optimal).  
Then change the Makefile global variable **PY** to:
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
If needed, set `PYTHONPATH` to the repo root before running:
```bash
# Linux/macOS
export PYTHONPATH=$(pwd)

# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location).Path
```

---

## Where results are saved

All scripts write to `outputs/`:
- **Figures**: `outputs/figures/*.png`
- **Tables**:  `outputs/tables/*.csv`
- **Logs**:    `outputs/logs/part_<letter>_YYYYMMDD-HHMMSS.log` (with a final JSON summary)

---

## Makefile usage (optional)

- `make setup` — install Python dependencies  
- `make a|b|c|d|e|f|g|h` — run a single experiment part  
- `make all` — run **all** parts a…h  
- `make figures` — list generated figures  
- `make tables` — list generated tables  
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
> ```python
> import os
> seed = int(os.getenv("SEED", 314))
> np.random.seed(seed)
> ```

