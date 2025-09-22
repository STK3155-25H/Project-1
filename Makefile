# ---------- Config ----------
PY?=python3
CODE=Code
EXP=$(CODE)/experiments
OUT=outputs
# OUT=$(CODE)/outputs
FIGS=$(OUT)/figures
TABLES=$(OUT)/tables
LOGS=$(OUT)/logs

# Common knobs (override like: make a MAX_DEGREE=12)
MAX_DEGREE?=15
NPOINTS?=40,50,100,500,1000

# ---------- Phony ----------
.PHONY: help setup dirs all a b c d e g h figures tables clean

help:
	@echo "Targets:"
	@echo "  setup        Install Python deps from Code/requirements.txt"
	@echo "  dirs         Create outputs folders"
	@echo "  a            Run Part A (OLS)"
	@echo "  b            Run Part B (Ridge grid)"
	@echo "  c            Run Part C (Closed-form vs GD)"
	@echo "  d            Run Part D (optimizers)"
	@echo "  e            Run Part E (OLS/Ridge/LASSO via GD)"
	@echo "  g            Run Part G (Bias-Variance)"
	@echo "  h            Run Part H (Cross-Validation)"
	@echo "  all          Run all parts A…H"
	@echo "  figures      List saved figures"
	@echo "  tables       List saved tables"
	@echo "  clean        Remove generated outputs"

setup:
	$(PY) -m pip install -r requirements.txt

dirs:
	mkdir -p $(FIGS) $(TABLES) $(LOGS)

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

g: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_g_bias_variance.py 2>&1 | tee "$(LOGS)/part_g_$$STAMP.log"

h: dirs
	@STAMP=`date +%Y%m%d-%H%M%S`; \
	$(PY) $(EXP)/exp_h_crossval.py 2>&1 | tee "$(LOGS)/part_h_$$STAMP.log"

# ---------- Meta ----------
all: a b c d e g h

figures:
	@echo "Figures:"; \
	if [ -d "$(FIGS)" ]; then find "$(FIGS)" -type f; else echo "(none yet)"; fi

tables:
	@echo "Tables:"; \
	if [ -d "$(TABLES)" ]; then find "$(TABLES)" -type f; else echo "(none yet)"; fi

clean:
	rm -rf $(OUT)
