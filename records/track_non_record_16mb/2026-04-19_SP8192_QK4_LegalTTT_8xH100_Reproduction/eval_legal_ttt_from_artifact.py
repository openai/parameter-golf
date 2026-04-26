#!/usr/bin/env python3
"""Evaluate a quantized SP8192 artifact with legal score-first TTT.

This wrapper imports the April 6 QK/TTT record implementation without requiring
the minified submission script to parse under Python 3.11. It patches the two
Python 3.12-only f-string constructs produced by minification, then runs the
record's `eval_val_sliding_ttt` against a supplied `final_model.int6.ptz`.
"""

from __future__ import annotations

import ast
import base64 as B
import importlib.util
import lzma as L
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "records/track_10min_16mb").is_dir():
            return parent
    raise RuntimeError(f"could not locate parameter-golf repo root from {path}")


def _load_qk_ttt_module():
    repo = _repo_root()
    record = repo / "records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/train_gpt.py"
    tree = ast.parse(record.read_text(encoding="utf-8"))
    compressed_expr = tree.body[1].value.args[0]
    source = eval(
        compile(ast.Expression(compressed_expr), str(record), "eval"),
        {"L": L, "B": B},
    ).decode("utf-8")
    source = source.replace(
        'log(f"  {cat}: {", ".join(sorted(categories[cat]))}")',
        'log(f"  {cat}: {\', \'.join(sorted(categories[cat]))}")',
    )
    source = source.replace('glob("fineweb_train_*.bin")', "glob('fineweb_train_*.bin')")

    tmp = Path(tempfile.gettempdir()) / "parameter_golf_qk_ttt_py311.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("parameter_golf_qk_ttt_py311", tmp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {tmp}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, record


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        dist.barrier()
    return distributed, world_size, local_rank


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    module, record_script = _load_qk_ttt_module()
    distributed, world_size, local_rank = _init_distributed()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    source_artifact = Path(os.environ["SOURCE_ARTIFACT"]).resolve()
    if not source_artifact.exists():
        raise FileNotFoundError(source_artifact)

    h = module.Hyperparameters()
    h.quantized_model_path = str(source_artifact)
    h.model_path = os.environ.get("MODEL_PATH", "unused_final_model.pt")
    h.logfile = f"logs/{h.run_id}.txt"
    h.distributed = distributed
    h.world_size = world_size
    h.local_rank = local_rank
    h.rank = int(os.environ.get("RANK", "0"))
    h.is_main_process = h.rank == 0
    h.grad_accum_steps = 8 // world_size
    h.sliding_window_enabled = False
    h.ttt_enabled = True
    module.set_logging_hparams(h)

    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    code_size = len(record_script.read_text(encoding="utf-8").encode("utf-8"))
    artifact_size = source_artifact.stat().st_size
    module.log(f"legal_ttt_eval:source_artifact {source_artifact}")
    module.log(f"legal_ttt_eval:artifact_bytes {artifact_size}")
    module.log(f"legal_ttt_eval:ttt_code_bytes {code_size}")
    module.log(f"legal_ttt_eval:ttt_submission_total_bytes {artifact_size + code_size}")
    module.log(f"legal_ttt_eval:qk_gain_init {h.qk_gain_init}")
    module.log(f"legal_ttt_eval:ttt_lr {h.ttt_lr}")
    module.log(f"legal_ttt_eval:ttt_epochs {h.ttt_epochs}")
    module.log(f"legal_ttt_eval:ttt_chunk_tokens {h.ttt_chunk_tokens}")
    module.log(f"legal_ttt_eval:eval_stride {h.eval_stride}")
    module.log("legal_ttt_eval:loading validation")

    val_data = module.ValidationData(h, device)
    module.log(f"legal_ttt_eval:val_tokens {val_data.val_tokens.numel() - 1}")

    ttt_model = module.deserialize(h, device)
    if h.num_loops > 0:
        ttt_model.looping_active = True

    module.timed_eval(
        "legal_ttt_exact",
        module.eval_val_sliding_ttt,
        h,
        ttt_model,
        h.rank,
        h.world_size,
        device,
        val_data,
        h.eval_stride,
    )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
