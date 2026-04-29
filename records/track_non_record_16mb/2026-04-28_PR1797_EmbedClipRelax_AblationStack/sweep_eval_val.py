"""Standalone eval_val sweep over NGramMixer (alpha, beta) configs.

Reuses train_gpt.py machinery. Loads a pre-trained quantized artifact once,
compiles forward_logits once, then loops over configs calling eval_val for
each. This amortizes compile cost across the sweep — each eval after the
first is ~1-2 minutes on 8xH100.

Run:
    torchrun --standalone --nproc_per_node=8 sweep_eval_val.py

Env (pass same values as the training run):
    DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, CASEOPS_ENABLED,
    QUANTIZED_MODEL_PATH  (path to the baseline model.bin)
Optional:
    SWEEP_OUTPUT  — where to append per-config JSON lines (default /workspace/runs/sweep.jsonl)
    SWEEP_CONFIGS — either "default" (use the 10-point grid below) or a
                    JSON array string of {alpha,beta,scale,use_uni_prior} dicts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Import train_gpt from the same folder.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Silence __main__ side effects and training. Import only what we need.
import importlib.util
spec = importlib.util.spec_from_file_location("train_gpt_mod", HERE / "train_gpt.py")
tg = importlib.util.module_from_spec(spec)
# Guard against train_gpt.main() running at import time: train_gpt.py has an
# `if __name__ == "__main__": main()` guard, so importing it does NOT trigger
# main. Good.
spec.loader.exec_module(tg)


DEFAULT_GRID = [
    # Sanity anchors
    {"name": "off_sanity", "kind": "ngram", "enabled": False},
    {"name": "temp_id",    "kind": "temp",  "enabled": True, "t_base": 1.0, "beta": 0.0},
    # Pure temperature scaling: fixed T (beta=0)
    {"name": "T_096_b00",  "kind": "temp", "enabled": True, "t_base": 0.96, "beta": 0.0},
    {"name": "T_098_b00",  "kind": "temp", "enabled": True, "t_base": 0.98, "beta": 0.0},
    {"name": "T_099_b00",  "kind": "temp", "enabled": True, "t_base": 0.99, "beta": 0.0},
    {"name": "T_100_b00",  "kind": "temp", "enabled": True, "t_base": 1.00, "beta": 0.0},  # exact dup of identity
    {"name": "T_101_b00",  "kind": "temp", "enabled": True, "t_base": 1.01, "beta": 0.0},
    {"name": "T_102_b00",  "kind": "temp", "enabled": True, "t_base": 1.02, "beta": 0.0},
    {"name": "T_104_b00",  "kind": "temp", "enabled": True, "t_base": 1.04, "beta": 0.0},
    # Adaptive: vary beta (running-NLL gain)
    {"name": "T_100_b+10", "kind": "temp", "enabled": True, "t_base": 1.00, "beta": +0.10},
    {"name": "T_100_b+20", "kind": "temp", "enabled": True, "t_base": 1.00, "beta": +0.20},
    {"name": "T_100_b-10", "kind": "temp", "enabled": True, "t_base": 1.00, "beta": -0.10},
    {"name": "T_100_b-20", "kind": "temp", "enabled": True, "t_base": 1.00, "beta": -0.20},
    {"name": "T_098_b+10", "kind": "temp", "enabled": True, "t_base": 0.98, "beta": +0.10},
    {"name": "T_102_b-10", "kind": "temp", "enabled": True, "t_base": 1.02, "beta": -0.10},
]


def main():
    # DDP init identical to train_gpt main()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")

    h = tg.Hyperparameters()
    # Hyperparameters reads env lazily in __init__ — it picked up our DATA_PATH
    # TOKENIZER_PATH etc. already. We only need to override mixer knobs per config.

    tg.set_logging_hparams(h)
    tg.BOS_ID = 1

    val_data = tg.ValidationData(h, device)
    tg.log(f"vocab: {h.vocab_size}  val_tokens: {val_data.val_tokens.numel()-1}")

    eval_model = tg.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    tg.log("deserialized artifact; compiling forward_logits...")

    compiled_forward_logits = torch.compile(
        eval_model.forward_logits, dynamic=False, fullgraph=True
    )
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    # Trigger compile by running one short eval with mixer OFF.
    tg.log("priming compile with one eval_val pass (mixer off)...")
    t0 = time.perf_counter()
    h.ngram_mix_enabled = False
    v_loss, v_bpb = tg.eval_val(h, device, val_data, compiled_model, compiled_forward_logits)
    compile_elapsed = time.perf_counter() - t0
    tg.log(f"prime-pass eval_val: val_loss={v_loss:.6f} val_bpb={v_bpb:.6f} ({compile_elapsed:.1f}s incl. compile)")

    # Select grid
    raw = os.environ.get("SWEEP_CONFIGS", "default")
    grid = DEFAULT_GRID if raw == "default" else json.loads(raw)

    out_path = Path(os.environ.get("SWEEP_OUTPUT", "/workspace/runs/sweep.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_main = local_rank == 0

    # Also record the prime-pass result as the canonical "mix off" datum.
    if is_main:
        with out_path.open("a") as fh:
            fh.write(json.dumps({
                "config": "prime_mix_off",
                "val_loss": float(v_loss),
                "val_bpb": float(v_bpb),
                "elapsed_s": round(compile_elapsed, 2),
            }) + "\n")

    for cfg in grid:
        name = cfg.get("name", "?")
        kind = cfg.get("kind", "ngram")
        # reset both knobs to OFF, then enable the one this cfg specifies.
        h.ngram_mix_enabled = False
        h.temp_scale_enabled = False
        if kind == "ngram" and cfg.get("enabled", False):
            h.ngram_mix_enabled = True
            h.ngram_mix_alpha = float(cfg.get("alpha", 2.0))
            h.ngram_mix_beta = float(cfg.get("beta", -0.25))
            h.ngram_mix_scale = float(cfg.get("scale", 8.0))
            h.ngram_mix_use_uni_prior = bool(cfg.get("use_uni_prior", True))
        elif kind == "temp" and cfg.get("enabled", False):
            h.temp_scale_enabled = True
            h.temp_base = float(cfg.get("t_base", 1.0))
            h.temp_beta = float(cfg.get("beta", 0.0))
            h.temp_ref_nll = float(cfg.get("ref_nll", 2.4))
            h.temp_warmup_tokens = int(cfg.get("warmup", 64))
        tg.log(f"=== sweep '{name}' kind={kind} ngram={h.ngram_mix_enabled} temp={h.temp_scale_enabled} ===")
        t0 = time.perf_counter()
        v_loss, v_bpb = tg.eval_val(h, device, val_data, compiled_model, compiled_forward_logits)
        elapsed = time.perf_counter() - t0
        tg.log(f"[{name}] val_loss={v_loss:.6f} val_bpb={v_bpb:.6f} elapsed={elapsed:.1f}s")
        if is_main:
            with out_path.open("a") as fh:
                fh.write(json.dumps({
                    "config": name,
                    "kind": kind,
                    "ngram_enabled": h.ngram_mix_enabled,
                    "temp_enabled": h.temp_scale_enabled,
                    "ngram_alpha": getattr(h, "ngram_mix_alpha", None) if h.ngram_mix_enabled else None,
                    "ngram_beta": getattr(h, "ngram_mix_beta", None) if h.ngram_mix_enabled else None,
                    "t_base": getattr(h, "temp_base", None) if h.temp_scale_enabled else None,
                    "temp_beta": getattr(h, "temp_beta", None) if h.temp_scale_enabled else None,
                    "val_loss": float(v_loss),
                    "val_bpb": float(v_bpb),
                    "elapsed_s": round(elapsed, 2),
                }) + "\n")

    tg.log("sweep complete — results in " + str(out_path))
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
