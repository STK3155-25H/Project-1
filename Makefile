# ---------- Config ----------
PY?=python3
CODE=Code
EXP=$(CODE)
OUT=outputs
# OUT=$(CODE)/outputs
FIGS=$(OUT)/figures
TABLES=$(OUT)/tables
LOGS=$(OUT)/logs
TESTS=$(OUT)/tests
PYTEST?=pytest
PYTEST_FLAGS?=-q
SEED?=123
BENCHMARKS = $(CODE)/src

# Common knobs (override like: make a MAX_DEGREE=12)
MAX_DEGREE?=15
NPOINTS?=40,50,100,500,1000

BENCHMARKS_OUT_DIR = $(OUT)/benchmarks
BENCHMARKS_OUT_TABLES_DIR = $(BENCHMARKS_OUT_DIR)/tables

BENCH_NPOINTS?=100
BENCH_MAX_DEGREE?=14
METHODS?=ols ridge gd-vanilla gd-momentum gd-adam
LAM?=0.01
LR?=0.01
N_ITER?=10000
TOL?=1e-8
BETA?=0.9
EPSILON?=1e-8
USE_SGD?=false          # set to true to use mini-batches
BATCH_SIZE?=32
BOOTSTRAP?=30
TEST_SIZE?=0.33
N_RUNS?=30

# ---------- Phony ----------
.PHONY: help setup dirs all a b c d e g h figures tables clean bench benchmark benchmarkrun tests

help:
	@echo "Targets:"
	@echo "  setup        Install Python deps from Code/requirements.txt"
	@echo "  dirs         Create outputs folders"
	@echo "  a            Run Part A (OLS)"
	@echo "  b            Run Part B (Ridge grid)"
	@echo "  c            Run Part C (Closed-form vs GD)"
	@echo "  d            Run Part D (optimizers)"
	@echo "  e            Run Part E (OLS/Ridge/LASSO via GD)"
	@echo "  f            Run Part E (sgd)"
	@echo "  g            Run Part G (Bias-Variance)"
	@echo "  h            Run Part H (Cross-Validation)"
	@echo "  all          Run all parts A…H"
	@echo "  figures      List saved figures"
	@echo "  tables       List saved tables"
	@echo "  clean        Remove generated outputs"
	@echo "  tests        Run unit tests (pytest) su Code/tests"
	@echo "  bench        Run complete benchmark (single, metric-rich) -> outputs/tables + logs"
	@echo "  benchmark    Alias for 'bench'"
	@echo "  benchmarkrun Alias for 'bench'"


setup:
	$(PY) -m pip install -r requirements.txt

dirs:
	mkdir -p $(FIGS) $(TABLES) $(LOGS) $(TESTS)

# ---------- Individual parts ----------
a: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_a_ols.py --n-points $(NPOINTS) --max-degree $(MAX_DEGREE) --noise 2>&1 | tee "$(LOGS)/part_a_$$STAMP.log"

b: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_b_ridge.py 2>&1 | tee "$(LOGS)/part_b_$$STAMP.log"

c: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_c_grad_vs_closed.py 2>&1 | tee "$(LOGS)/part_c_$$STAMP.log"

d: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_d_optimizers.py 2>&1 | tee "$(LOGS)/part_d_$$STAMP.log"

e: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_e_lasso.py 2>&1 | tee "$(LOGS)/part_e_$$STAMP.log"

f: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_f_sgd.py 2>&1 | tee "$(LOGS)/part_f_$$STAMP.log"

g: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_g_bias_variance.py 2>&1 | tee "$(LOGS)/part_g_$$STAMP.log"

h: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_h_crossval.py 2>&1 | tee "$(LOGS)/part_h_$$STAMP.log"

# ---------- Meta ----------
all: a b c d e f g h

tests:
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	PYTHONPATH=$(CODE) SEED=$(SEED) \
	$(PYTEST) $(PYTEST_FLAGS) \
	-o python_files="tests.py test_*.py" \
	-o testpaths="$(CODE)/tests" 2>&1 | tee "$(TESTS)/tests_$${STAMP}.log"


figures:
	@echo "Figures:"; \
	if [ -d "$(FIGS)" ]; then find "$(FIGS)" -type f; else echo "(none yet)"; fi

tables:
	@echo "Tables:"; \
	if [ -d "$(TABLES)" ]; then find "$(TABLES)" -type f; else echo "(none yet)"; fi

# ---------- Complete benchmark run ----------
bench: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	FLAGS="--n-points $(BENCH_NPOINTS) --max-degree $(BENCH_MAX_DEGREE) \
	       --n-runs $(N_RUNS) \
		   --lam $(LAM) --lr $(LR) --n-iter $(N_ITER) --tol $(TOL) \
	       --beta $(BETA) --epsilon $(EPSILON) --batch-size $(BATCH_SIZE) \
	       --bootstrap $(BOOTSTRAP) --test-size $(TEST_SIZE)"; \
	if [ "$(USE_SGD)" = "true" ]; then FLAGS="$$FLAGS --use-sgd"; fi; \
	if [ -n "$(NOISE)" ]; then FLAGS="$$FLAGS --noise"; fi; \
	if [ -n "$(METHODS)" ]; then FLAGS="$$FLAGS --methods $(METHODS)"; fi; \
	echo "Running benchmarks with: $$FLAGS" | tee "$(LOGS)/bench_$$STAMP.log"; \
	$(PY) $(BENCHMARKS)/exp_benchmarks_metrics.py $$FLAGS 2>&1 | tee -a "$(LOGS)/bench_$$STAMP.log"; \
	echo "Done. See CSVs in $(TABLES) and log $(LOGS)/bench_$$STAMP.log"

benchmark: bench
benchmarkrun: bench

clean:
	rm -rf $(OUT)
