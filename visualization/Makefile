# Browser visualization build.
#
# Two halves, glued by `npm run build`:
#
#   1. Artifact pipeline (this Makefile): turn the .ptz artifacts into the
#      static files the demo needs at /models/ar and /models/mdlm, plus
#      copy onnxruntime-web's WASM into public/onnx-wasm/.
#   2. Vite bundle (`vite build`): roll up index.html / parity.html and the
#      ESM modules into dist/, copying public/ along the way.
#
# Targets:
#   make             # = make all
#   make all         # build everything: artifacts + ort-wasm
#   make build       # build artifacts only (no ort-wasm copy)
#   make ort-wasm    # copy onnxruntime-web's WASM/mjs into public/onnx-wasm/
#                    # (needed in both dev and prod so onnx_runner.js can
#                    #  find them at /onnx-wasm/...)
#   make verify      # SHA-256 + parity assertions; safe to run repeatedly
#   make clean       # remove generated files in public/models/* and public/onnx-wasm
#
# Compatible with GNU Make 3.81 (Apple's default) and BSD cp.

PYTHON          ?= ../.venv/bin/python
NODE            ?= node
ARTIFACTS_DIR   := ../artifacts
PUBLIC          := public
SCRIPTS         := scripts

AR_PTZ          := $(ARTIFACTS_DIR)/ar-baseline-9L-512d.ptz
MDLM_PTZ        := $(ARTIFACTS_DIR)/mdlm-submission-seed42-19M.ptz

AR_DIR          := $(PUBLIC)/models/ar
MDLM_DIR        := $(PUBLIC)/models/mdlm

PACK            := $(SCRIPTS)/pack_safetensors.py
EXPORT          := $(SCRIPTS)/export_to_onnx.py
MANIFEST        := $(SCRIPTS)/build_onnx_data_manifest.py
PYREBUILD       := $(SCRIPTS)/reconstruct_onnx_data.py
JSREBUILD       := $(SCRIPTS)/node_parity_safetensors.js
JS_AR_PARITY    := $(SCRIPTS)/node_parity_ar.js
JS_MDLM_PARITY  := $(SCRIPTS)/node_parity.js

# Per-model "primary" output. Make sees one rule for each primary; secondary
# outputs (model.json, model.onnx.data) are declared as depending on the
# primary so make knows they're produced by the same recipe — no recipe of
# their own. (3.81-portable workaround for missing `&:` grouped targets.)

AR_PRIMARY      := $(AR_DIR)/model.safetensors.gz
AR_SECONDARY    := $(AR_DIR)/model.json
MDLM_PRIMARY    := $(MDLM_DIR)/model.safetensors.gz
MDLM_SECONDARY  := $(MDLM_DIR)/model.json

AR_OUTPUTS      := $(AR_PRIMARY) $(AR_SECONDARY) \
                   $(AR_DIR)/model.onnx $(AR_DIR)/model.onnx.data \
                   $(AR_DIR)/onnx_data_manifest.json
MDLM_OUTPUTS    := $(MDLM_PRIMARY) $(MDLM_SECONDARY) \
                   $(MDLM_DIR)/model.onnx $(MDLM_DIR)/model.onnx.data \
                   $(MDLM_DIR)/onnx_data_manifest.json

.PHONY: all build verify clean ort-wasm publish smollm2 help

all: build ort-wasm smollm2

build: $(AR_OUTPUTS) $(MDLM_OUTPUTS)

# ---- SmolLM2-135M-Instruct: int4 ONNX built from source ----
# Pipeline (per scripts/quantize_smollm2.py, adapted from sidechat's 360M
# recipe):
#   export_smollm2_with_kv.py → fp32 model.onnx (with cumulative KV cache)
#                              → MatMulNBits int4 weight-only quantize
#                              → byte-alias Gather (embedding) with lm_head
#                                MatMulNBits → on-disk embedding stored once
#                              → split external data across 4 chunk files
# The byte-aliasing trades ~0.08 BPB (1.03 vs 0.95 on the canonical
# Transformer abstract) for ~40 MB on disk (77 MB vs HF's 117 MB).
# Activations stay fp32; SmolLM2's layernorm Cast(fp32) pattern defeats
# onnxconverter_common.float16, and the small extra fp32 footprint isn't
# worth the rabbit hole. No fp64 anywhere.
#
# Step 1 uses scripts/export_smollm2_with_kv.py (NOT raw optimum-cli) because
# optimum 2.1.0 + transformers ≥4.56's `patched_dynamic_layer_update` only
# traces the `if self.keys is None: self.keys = key_states` branch, so the
# Concat that turns delta `present.*` outputs into cumulative ones is missing
# from the exported graph — and a stock `--task text-generation-with-past`
# export gives delta-only `present.{i}.{key,value}` outputs, which don't
# compose into a per-step decode cache. The wrapper monkey-patches the
# function so the Concat is always traced; output present is then cumulative
# and our manual KV-cache loop in src/generate.js works.

SMOLLM2_DIR        := $(PUBLIC)/models/smollm2-135m
SMOLLM2_HF         := https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main
SMOLLM2_BUILD      := build-cache/smollm2-135m
SMOLLM2_FP32       := $(SMOLLM2_BUILD)/model.onnx
SMOLLM2_ONNX       := $(SMOLLM2_DIR)/onnx/model_q4.onnx
SMOLLM2_FILES      := $(SMOLLM2_DIR)/tokenizer.json \
                      $(SMOLLM2_DIR)/tokenizer_config.json \
                      $(SMOLLM2_DIR)/special_tokens_map.json \
                      $(SMOLLM2_DIR)/config.json \
                      $(SMOLLM2_DIR)/generation_config.json
SMOLLM2_EXPORT     := $(SCRIPTS)/export_smollm2_with_kv.py
SMOLLM2_QUANTIZE   := $(SCRIPTS)/quantize_smollm2.py

smollm2: $(SMOLLM2_ONNX) $(SMOLLM2_FILES)

# Step 1: HF → fp32 model.onnx via the patched optimum-cli wrapper
# (~2 min, ~622 MB). Cached in build-cache/ so re-runs of `make smollm2` skip
# it. Same "build once, cache" pattern as the rest of the artifacts: rm the
# fp32 file (or the whole build-cache/) to force a re-export.

$(SMOLLM2_FP32):
	mkdir -p $(SMOLLM2_BUILD)
	cd $(SMOLLM2_BUILD) && $(CURDIR)/$(PYTHON) $(CURDIR)/$(SMOLLM2_EXPORT)

# Step 2: int4 quantize → 4 chunk files in $(SMOLLM2_DIR)/onnx/
# `$(SMOLLM2_FP32)` is an order-only prerequisite (|): it has to exist when
# we run, but we don't trigger a re-quantize just because its mtime is
# newer than the q4 output. That keeps `make smollm2` a no-op once the
# int4 artifact is committed, even if the (gitignored) build-cache is gone.
$(SMOLLM2_ONNX): $(SMOLLM2_QUANTIZE) | $(SMOLLM2_FP32)
	mkdir -p $(SMOLLM2_DIR)/onnx
	$(PYTHON) $(SMOLLM2_QUANTIZE) \
	  --src $(SMOLLM2_FP32) --out-dir $(SMOLLM2_DIR)/onnx --hidden 576

# Step 3: tokenizer + config files (small, fetch from HF directly).
$(SMOLLM2_DIR)/%.json:
	mkdir -p $(SMOLLM2_DIR)
	curl -sSL '$(SMOLLM2_HF)/$(notdir $@)' -o $@

# ---- pack: .ptz → model.safetensors.gz + model.json ----

$(AR_PRIMARY): $(AR_PTZ) $(PACK)
	$(PYTHON) $(PACK) --artifact $(AR_PTZ) --out-dir $(AR_DIR)

$(AR_SECONDARY): $(AR_PRIMARY)

$(MDLM_PRIMARY): $(MDLM_PTZ) $(PACK)
	$(PYTHON) $(PACK) --artifact $(MDLM_PTZ) --out-dir $(MDLM_DIR)

$(MDLM_SECONDARY): $(MDLM_PRIMARY)

# ---- export: .ptz → model.onnx + model.onnx.data ----

$(AR_DIR)/model.onnx: $(AR_PTZ) $(EXPORT)
	$(PYTHON) $(EXPORT) --kind ar --artifact $(AR_PTZ) --out $(AR_DIR)/model.onnx

$(AR_DIR)/model.onnx.data: $(AR_DIR)/model.onnx

$(MDLM_DIR)/model.onnx: $(MDLM_PTZ) $(EXPORT)
	$(PYTHON) $(EXPORT) --kind mdlm --artifact $(MDLM_PTZ) --out $(MDLM_DIR)/model.onnx

$(MDLM_DIR)/model.onnx.data: $(MDLM_DIR)/model.onnx

# ---- manifest: model.onnx + safetensors.gz → onnx_data_manifest.json ----
# Reads the .gz directly (decompresses to a temp file internally).

$(AR_DIR)/onnx_data_manifest.json: $(AR_DIR)/model.onnx $(AR_DIR)/model.onnx.data $(AR_PRIMARY) $(MANIFEST)
	$(PYTHON) $(MANIFEST) --model-dir $(AR_DIR)

$(MDLM_DIR)/onnx_data_manifest.json: $(MDLM_DIR)/model.onnx $(MDLM_DIR)/model.onnx.data $(MDLM_PRIMARY) $(MANIFEST)
	$(PYTHON) $(MANIFEST) --model-dir $(MDLM_DIR)

# ---- ort-wasm: copy onnxruntime-web's runtime files into public/ ----
# Vite serves public/ at site root, so /onnx-wasm/ works in both `vite dev`
# and the production `dist/` build. ~75 MB of WASM in dist/ is the cost of
# shipping ORT-Web — it's the largest single thing in the build.

ORT_DIST     := node_modules/onnxruntime-web/dist
ORT_PUBLIC   := $(PUBLIC)/onnx-wasm
ORT_STAMP    := $(ORT_PUBLIC)/.stamp

ort-wasm: $(ORT_STAMP)

$(ORT_STAMP): $(wildcard $(ORT_DIST)/*.wasm) $(wildcard $(ORT_DIST)/*.mjs) | $(ORT_PUBLIC)
	cp $(ORT_DIST)/*.wasm $(ORT_PUBLIC)/
	cp $(ORT_DIST)/*.mjs $(ORT_PUBLIC)/
	touch $@

$(ORT_PUBLIC):
	mkdir -p $@

# ---- verify: byte-exactness + JS↔Python parity assertions ----

verify:
	$(PYTHON) $(PYREBUILD) --model-dir $(AR_DIR)
	$(PYTHON) $(PYREBUILD) --model-dir $(MDLM_DIR)
	$(NODE) $(JSREBUILD) all
	$(NODE) $(JS_AR_PARITY)
	$(NODE) $(JS_MDLM_PARITY)

# ---- publish: produce dist/ stripped of build-only fp32 sidecars ----
# `model.onnx.data` is huge (65–75 MB) and not needed at runtime — the JS
# loader reconstructs it byte-for-byte from model.safetensors.gz +
# onnx_data_manifest.json. We keep it in public/ during dev so `make verify`
# can use it as a third byte-for-byte witness; we strip it from dist/ here.

publish: all
	npx vite build
	@echo
	@echo "Trimming build-only files from dist/..."
	# fp32 ONNX sidecars: reconstructed in-memory at load time from
	# safetensors.gz + onnx_data_manifest.json.
	rm -f dist/models/ar/model.onnx.data
	rm -f dist/models/mdlm/model.onnx.data
	# goldens/ is ~667 MB of debugging fixtures (logits captures, top-k
	# tables, mask sequences) used only by node parity scripts and
	# parity.html — never by the public demo. Strip from production.
	rm -rf dist/goldens
	# Vite hoists onnxruntime-web's WASM into dist/assets/<hash>.wasm via
	# import.meta.url heuristics. Our wasmPaths setting points to
	# /onnx-wasm/, so the hoisted copies are dead weight (~47 MB).
	rm -f dist/assets/*.wasm
	@echo
	@echo "dist/ contents:"
	@du -sh dist/* | sort -h
	@echo
	@echo "dist/models/* (per-artifact ≤16 MiB target):"
	@du -ah dist/models/ar dist/models/mdlm | sort -k1
	@echo
	@echo "dist/ total:"
	@du -sh dist/

# ---- clean ----

clean:
	rm -rf $(AR_DIR) $(MDLM_DIR) $(SMOLLM2_DIR) $(ORT_PUBLIC) build-cache dist

help:
	@echo "make            build everything (artifacts + ort-wasm + smollm2)"
	@echo "make build      build AR + MDLM artifacts only"
	@echo "make ort-wasm   copy onnxruntime-web's WASM into public/onnx-wasm/"
	@echo "make smollm2    fetch HF SmolLM2-135M-Instruct q4f16 + tokenizer locally"
	@echo "make verify     run SHA-256 + JS↔Python parity assertions"
	@echo "make publish    artifacts + ort-wasm + smollm2 + vite build → dist/"
	@echo "make clean      remove all generated files"
