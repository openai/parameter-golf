# GOLDEN SUNFLOWERS · local verification harness.
# Anchor: phi^2 + phi^-2 = 3 · TRINITY · 🌻
#
# `make verify`      runs the full chain: parse + 5/5 smoke + 3/3 equivalence
#                    + cffconvert + coqc on the 2 Qed theorems.
# `make smoke`       module smoke (5/5) only.
# `make equivalence` baseline byte-equivalence (3/3) only.
# `make coq`         compile theorems/GoldenSunflowers.v (requires coqc 8.18+).
# `make clean`       remove Coq build artefacts and compiled __pycache__.

HERE         := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
PY           ?= python3
COQC         ?= coqc

.PHONY: verify smoke equivalence cff coq clean help

help:
	@echo "GOLDEN SUNFLOWERS — local verification"
	@echo "  make verify        — parse + smoke + equivalence + cff + coq (all)"
	@echo "  make smoke         — module smoke 5/5"
	@echo "  make equivalence   — baseline byte-equivalence 3/3"
	@echo "  make cff           — cffconvert --validate CITATION.cff"
	@echo "  make coq           — coqc theorems/GoldenSunflowers.v"
	@echo "  make clean         — remove build artefacts"
	@echo
	@echo "Anchor: phi^2 + phi^-2 = 3"

verify: parse smoke equivalence cff coq
	@echo
	@echo "🌻 GOLDEN SUNFLOWERS · local verify complete"
	@echo "   phi^2 + phi^-2 = 3"

parse:
	@$(PY) -c "import ast; ast.parse(open('$(HERE)train_gpt.py').read()); print('train_gpt.py: parse OK')"

smoke:
	@$(PY) $(HERE)smoke_modules.py

equivalence:
	@$(PY) $(HERE)baseline_equivalence.py

cff:
	@$(PY) -m pip install --quiet cffconvert 2>/dev/null || true
	@cffconvert --validate -i $(HERE)CITATION.cff

coq:
	@cd $(HERE)theorems && $(COQC) -q GoldenSunflowers.v \
		&& echo "theorems/GoldenSunflowers.v: coqc OK (2 Qed)"

clean:
	@rm -f $(HERE)theorems/*.vo $(HERE)theorems/*.vos $(HERE)theorems/*.vok $(HERE)theorems/*.glob
	@find $(HERE) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
