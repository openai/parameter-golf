import modal
import sys
import subprocess
import os
import glob
import shutil
import struct
import json
from collections import deque

app = modal.App("pg-train-detached")

image = (
    modal.Image.from_dockerfile("deploy/Dockerfile", context_dir=".", add_python="3.12")
    .pip_install("huggingface_hub")
)

data_volume = modal.Volume.from_name("pg-data", create_if_missing=True)
output_volume = modal.Volume.from_name("pg-output", create_if_missing=True)


def _pop_result_json(args: list[str]):
    forwarded = list(args)
    result_json = os.environ.get("PG_RESULT_JSON")
    if "--result-json" in forwarded:
        idx = forwarded.index("--result-json")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--result-json requires a path")
        result_json = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    return forwarded, result_json


def _write_result_json(path: str | None, result: dict):
    if not path:
        return
    if not path.startswith("/output/"):
        raise RuntimeError("--result-json must write under /output")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
    output_volume.commit()


def _write_running_result_json(path: str | None, label: str, cmd: list[str]):
    _write_result_json(
        path,
        {
            "label": label,
            "command": cmd,
            "returncode": None,
            "status": "running",
            "tail": "",
        },
    )


def _coerce_metric_value(raw: str):
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_key_value_metrics(text: str) -> dict:
    metrics: dict[str, object] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("["):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or " " in key or key.endswith("_json"):
            continue
        metrics[key] = _coerce_metric_value(value)
    _add_per_step_timing_metrics(metrics)
    return metrics


def _add_per_step_timing_metrics(metrics: dict[str, object]) -> None:
    steps = metrics.get("timing_steps")
    if not isinstance(steps, int) or steps <= 0:
        return
    for key, value in list(metrics.items()):
        if not key.startswith("timing_") or key.endswith("_per_step"):
            continue
        if key in {"timing_steps", "timing_measured_ms_per_step"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[f"{key}_per_step"] = float(value) / float(steps)


def _apply_frontier_fast_record_env(stage_timing: bool, poison_prepacked_qkv: bool):
    # Matches the best measured record-shaped profile line. Keep this as an
    # explicit opt-in so slow correctness baselines remain easy to run.
    os.environ["PG_CUDA_EVENT_TIMING"] = "1"
    os.environ["PG_GPU_BACKWARD_STAGE_TIMING"] = "1" if stage_timing else "0"
    os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "all"
    os.environ["PG_GPU_DIRECT_SAVED_ACTS"] = "1"
    os.environ["PG_GPU_BF16_PRIMARY_FORWARD_GEMM"] = "1"
    os.environ["PG_GPU_BF16_LOGITS"] = "1"
    os.environ["PG_GPU_QKV_DX_BETA_ACCUM"] = "1"
    os.environ["PG_GPU_FUSED_QKV_PROJ"] = "1"
    os.environ["PG_GPU_FUSED_QKV_PROJ_RECORD_OK"] = "1"
    os.environ["PG_GPU_BF16_MLP_UP_OUTPUT"] = "1"
    os.environ["PG_GPU_BF16_NORM_SIDE_OUTPUTS"] = "1"
    os.environ["PG_GPU_BF16_NORM_GRAD_PATH"] = "1"
    os.environ["PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT"] = "1"
    os.environ["PG_GPU_BF16_ATTN_PROJ_OUTPUT"] = "1"
    os.environ["PG_GPU_FINAL_NORM_BF16_OUTPUT"] = "1"
    os.environ["PG_GPU_CUDNN_PREPACKED_BF16_ATTN"] = "1"
    os.environ["PG_GPU_CUDNN_PREPACKED_BF16_POISON"] = "1" if poison_prepacked_qkv else "0"
    os.environ["PG_GPU_BF16_SPARSE_XSA_FWD"] = "1"
    os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "1"
    os.environ["PG_GPU_HOST_SCALAR_UPDATES"] = "0"
    os.environ["PG_GPU_MUON_NS_PROFILE"] = "polar_express"
    os.environ.setdefault("PG_NCCL_BF16_BANK_GRAD_WIRE", "1")
    os.environ.setdefault("PG_NCCL_GROUP_SHARDED_GRAD_COLLECTIVES", "1")
    os.environ.setdefault("PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER", "1")
    os.environ["PG_GPU_RESIDUAL_SCALE_REDUCE"] = "1"
    os.environ["PG_GPU_CHUNKED_Q_GAIN_BWD"] = "1"
    # v82 H100 A/B regressed this path (278.9 ms/step vs v76 251.7 ms).
    # Keep it as an explicit A/B flag until the downstream BF16 QKV gradient
    # pack path is complete enough to offset the cuDNN BF16-gradient overhead.
    os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "0"
    os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "0"
    os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "1"
    os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "1"
    os.environ.setdefault("PG_GPU_OUTPUT_CE_CHUNK_TOKENS", "8192")
    # v87 H100 A/B regressed this path (267.3 ms/step vs v86 256.8 ms).
    # Keep compact U16 upload as an explicit A/B flag, but do not ship it in
    # the fastest record-shaped profile until the sampler is fully GPU-resident.
    os.environ["PG_GPU_SHIFTED_U16_BATCH_UPLOAD"] = "0"
    os.environ["PG_GPU_SPLIT_RESIDUAL_MIX_GRAD"] = "0"
    os.environ.setdefault("PG_RECORD_TIMING_SKIP_STEPS", "2")


def _apply_gpu_env_flags(forwarded: list[str]):
    if "--frontier-fast-record-profile" in forwarded:
        forwarded.remove("--frontier-fast-record-profile")
        _apply_frontier_fast_record_env(stage_timing=True, poison_prepacked_qkv=True)
    if "--frontier-throughput-record-profile" in forwarded:
        forwarded.remove("--frontier-throughput-record-profile")
        _apply_frontier_fast_record_env(stage_timing=False, poison_prepacked_qkv=False)
    if "--frontier-graph-record-profile" in forwarded:
        forwarded.remove("--frontier-graph-record-profile")
        _apply_frontier_fast_record_env(stage_timing=False, poison_prepacked_qkv=False)
        os.environ["PG_CUDA_BACKWARD_GRAPH"] = "1"
        os.environ["PG_CUDA_BACKWARD_GRAPH_STRICT"] = "1"
    if "--chunked-residual-mix-bwd" in forwarded:
        forwarded.remove("--chunked-residual-mix-bwd")
        os.environ["PG_GPU_CHUNKED_RESIDUAL_MIX_BWD"] = "1"
    if "--recompute-residual-mix-norm-inputs" in forwarded:
        forwarded.remove("--recompute-residual-mix-norm-inputs")
        os.environ["PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS"] = "1"
    if "--bf16-qkv-dx-output" in forwarded:
        forwarded.remove("--bf16-qkv-dx-output")
        os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "1"
    if "--cuda-event-timing" in forwarded:
        forwarded.remove("--cuda-event-timing")
        os.environ["PG_CUDA_EVENT_TIMING"] = "1"
    if "--backward-stage-timing" in forwarded:
        forwarded.remove("--backward-stage-timing")
        os.environ["PG_GPU_BACKWARD_STAGE_TIMING"] = "1"
    if "--cuda-stage-timing" in forwarded:
        forwarded.remove("--cuda-stage-timing")
        os.environ["PG_GPU_BACKWARD_STAGE_TIMING"] = "1"
    if "--cuda-backward-graph" in forwarded:
        forwarded.remove("--cuda-backward-graph")
        os.environ["PG_CUDA_BACKWARD_GRAPH"] = "1"
    if "--cuda-backward-graph-strict" in forwarded:
        forwarded.remove("--cuda-backward-graph-strict")
        os.environ["PG_CUDA_BACKWARD_GRAPH"] = "1"
        os.environ["PG_CUDA_BACKWARD_GRAPH_STRICT"] = "1"
    if "--save-layer-acts" in forwarded:
        forwarded.remove("--save-layer-acts")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "1"
    if "--save-recurrent-layer-acts" in forwarded:
        forwarded.remove("--save-recurrent-layer-acts")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "recurrent"
    if "--save-inner-layer-acts" in forwarded:
        forwarded.remove("--save-inner-layer-acts")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "inner"
    if "--save-all-layer-acts" in forwarded:
        forwarded.remove("--save-all-layer-acts")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "all"
    if "--direct-saved-layer-acts" in forwarded:
        forwarded.remove("--direct-saved-layer-acts")
        os.environ["PG_GPU_DIRECT_SAVED_ACTS"] = "1"
    if "--ttt-audit" in forwarded:
        forwarded.remove("--ttt-audit")
        os.environ["PG_TTT_AUDIT"] = "1"
    if "--assert-ttt-score-no-mutation" in forwarded:
        forwarded.remove("--assert-ttt-score-no-mutation")
        os.environ["PG_TTT_ASSERT_SCORE_NO_MUTATION"] = "1"
    if "--eval-gpu-world-size" in forwarded:
        idx = forwarded.index("--eval-gpu-world-size")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--eval-gpu-world-size requires a positive integer")
        os.environ["PG_EVAL_GPU_WORLD_SIZE"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--submission-code-bytes" in forwarded:
        idx = forwarded.index("--submission-code-bytes")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--submission-code-bytes requires a byte count")
        os.environ["PG_SUBMISSION_CODE_BYTES"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--submission-code-dir" in forwarded:
        idx = forwarded.index("--submission-code-dir")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--submission-code-dir requires a path")
        os.environ["PG_SUBMISSION_CODE_DIR"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--skip-first-step-timing" in forwarded:
        forwarded.remove("--skip-first-step-timing")
        os.environ["PG_RECORD_TIMING_SKIP_STEPS"] = "1"
    if "--record-max-ms-per-step" in forwarded:
        idx = forwarded.index("--record-max-ms-per-step")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--record-max-ms-per-step requires a numeric ceiling")
        os.environ["PG_RECORD_MAX_MS_PER_STEP"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--fast-tf32" in forwarded:
        forwarded.remove("--fast-tf32")
        os.environ["PG_CUBLAS_FAST_TF32"] = "1"
        os.environ.setdefault("PG_CUBLAS_FORCE_TENSOR_OP_ALGO", "1")
    if "--bf16-gemm-algo" in forwarded:
        idx = forwarded.index("--bf16-gemm-algo")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--bf16-gemm-algo requires an algorithm id")
        os.environ["PG_CUBLAS_BF16_ALGO"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--disable-bf16-forward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-forward-gemm")
        os.environ["PG_GPU_BF16_FORWARD_GEMM"] = "0"
    if "--enable-bf16-primary-forward-gemm" in forwarded:
        forwarded.remove("--enable-bf16-primary-forward-gemm")
        os.environ["PG_GPU_BF16_PRIMARY_FORWARD_GEMM"] = "1"
    if "--disable-bf16-primary-forward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-primary-forward-gemm")
        os.environ["PG_GPU_BF16_PRIMARY_FORWARD_GEMM"] = "0"
    if "--disable-bf16-backward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-backward-gemm")
        os.environ["PG_GPU_BF16_BACKWARD_GEMM"] = "0"
    if "--disable-bf16-output-gemm" in forwarded:
        forwarded.remove("--disable-bf16-output-gemm")
        os.environ["PG_GPU_BF16_OUTPUT_GEMM"] = "0"
    if "--disable-bf16-output-backward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-output-backward-gemm")
        os.environ["PG_GPU_BF16_OUTPUT_BACKWARD_GEMM"] = "0"
    if "--enable-bf16-logits" in forwarded:
        forwarded.remove("--enable-bf16-logits")
        os.environ["PG_GPU_BF16_LOGITS"] = "1"
    if "--disable-bf16-logits" in forwarded:
        forwarded.remove("--disable-bf16-logits")
        os.environ["PG_GPU_BF16_LOGITS"] = "0"
    if "--disable-fused-ce-loss-bwd" in forwarded:
        forwarded.remove("--disable-fused-ce-loss-bwd")
        os.environ["PG_GPU_FUSED_CE_LOSS_BWD"] = "0"
    if "--enable-tiled-output-ce" in forwarded:
        forwarded.remove("--enable-tiled-output-ce")
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "1"
    if "--disable-tiled-output-ce" in forwarded:
        forwarded.remove("--disable-tiled-output-ce")
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    if "--enable-chunked-output-ce-cache" in forwarded:
        forwarded.remove("--enable-chunked-output-ce-cache")
        os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "1"
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    if "--disable-chunked-output-ce-cache" in forwarded:
        forwarded.remove("--disable-chunked-output-ce-cache")
        os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "0"
    if "--output-ce-chunk-tokens" in forwarded:
        idx = forwarded.index("--output-ce-chunk-tokens")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--output-ce-chunk-tokens requires a token count")
        os.environ["PG_GPU_OUTPUT_CE_CHUNK_TOKENS"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--output-ce-tile-vocab" in forwarded:
        idx = forwarded.index("--output-ce-tile-vocab")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--output-ce-tile-vocab requires a tile size")
        os.environ["PG_GPU_OUTPUT_CE_TILE_VOCAB"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--enable-qkv-dx-beta-accum" in forwarded:
        forwarded.remove("--enable-qkv-dx-beta-accum")
        os.environ["PG_GPU_QKV_DX_BETA_ACCUM"] = "1"
    if "--experimental-fused-qkv-proj" in forwarded:
        forwarded.remove("--experimental-fused-qkv-proj")
        os.environ["PG_GPU_FUSED_QKV_PROJ"] = "1"
    if "--fused-qkv-proj-record-ok" in forwarded:
        forwarded.remove("--fused-qkv-proj-record-ok")
        os.environ["PG_GPU_FUSED_QKV_PROJ_RECORD_OK"] = "1"
    if "--qkv-dx-beta-accum" in forwarded:
        forwarded.remove("--qkv-dx-beta-accum")
        os.environ["PG_GPU_QKV_DX_BETA_ACCUM"] = "1"
    if "--enable-bf16-qkv-dx-output" in forwarded:
        forwarded.remove("--enable-bf16-qkv-dx-output")
        os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "1"
    if "--disable-bf16-qkv-dx-output" in forwarded:
        forwarded.remove("--disable-bf16-qkv-dx-output")
        os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "0"
    if "--disable-fused-qk-rope-gain-bwd" in forwarded:
        forwarded.remove("--disable-fused-qk-rope-gain-bwd")
        os.environ["PG_GPU_FUSED_QK_ROPE_GAIN_BWD"] = "0"
    if "--disable-fused-qk-rope-gain-fwd" in forwarded:
        forwarded.remove("--disable-fused-qk-rope-gain-fwd")
        os.environ["PG_GPU_FUSED_QK_ROPE_GAIN_FWD"] = "0"
    if "--disable-fused-residual-mix-norm" in forwarded:
        forwarded.remove("--disable-fused-residual-mix-norm")
        os.environ["PG_GPU_FUSED_RESIDUAL_MIX_NORM"] = "0"
    if "--disable-fused-mlp-act-bf16" in forwarded:
        forwarded.remove("--disable-fused-mlp-act-bf16")
        os.environ["PG_GPU_FUSED_MLP_ACT_BF16"] = "0"
    if "--bf16-mlp-up-output" in forwarded:
        forwarded.remove("--bf16-mlp-up-output")
        os.environ["PG_GPU_BF16_MLP_UP_OUTPUT"] = "1"
    if "--enable-bf16-norm-side-outputs" in forwarded:
        forwarded.remove("--enable-bf16-norm-side-outputs")
        os.environ["PG_GPU_BF16_NORM_SIDE_OUTPUTS"] = "1"
    if "--disable-bf16-norm-side-outputs" in forwarded:
        forwarded.remove("--disable-bf16-norm-side-outputs")
        os.environ["PG_GPU_BF16_NORM_SIDE_OUTPUTS"] = "0"
    if "--enable-bf16-norm-grad-path" in forwarded:
        forwarded.remove("--enable-bf16-norm-grad-path")
        os.environ["PG_GPU_BF16_NORM_GRAD_PATH"] = "1"
    if "--disable-bf16-norm-grad-path" in forwarded:
        forwarded.remove("--disable-bf16-norm-grad-path")
        os.environ["PG_GPU_BF16_NORM_GRAD_PATH"] = "0"
    if "--enable-bf16-residual-proj-output" in forwarded:
        forwarded.remove("--enable-bf16-residual-proj-output")
        os.environ["PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT"] = "1"
    if "--disable-bf16-residual-proj-output" in forwarded:
        forwarded.remove("--disable-bf16-residual-proj-output")
        os.environ["PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT"] = "0"
    if "--enable-bf16-attn-proj-output" in forwarded:
        forwarded.remove("--enable-bf16-attn-proj-output")
        os.environ["PG_GPU_BF16_ATTN_PROJ_OUTPUT"] = "1"
    if "--disable-bf16-attn-proj-output" in forwarded:
        forwarded.remove("--disable-bf16-attn-proj-output")
        os.environ["PG_GPU_BF16_ATTN_PROJ_OUTPUT"] = "0"
    if "--enable-final-norm-bf16-output" in forwarded:
        forwarded.remove("--enable-final-norm-bf16-output")
        os.environ["PG_GPU_FINAL_NORM_BF16_OUTPUT"] = "1"
    if "--disable-final-norm-bf16-output" in forwarded:
        forwarded.remove("--disable-final-norm-bf16-output")
        os.environ["PG_GPU_FINAL_NORM_BF16_OUTPUT"] = "0"
    if "--enable-prepacked-bf16-attention" in forwarded:
        forwarded.remove("--enable-prepacked-bf16-attention")
        os.environ["PG_GPU_CUDNN_PREPACKED_BF16_ATTN"] = "1"
    if "--disable-prepacked-bf16-attention" in forwarded:
        forwarded.remove("--disable-prepacked-bf16-attention")
        os.environ["PG_GPU_CUDNN_PREPACKED_BF16_ATTN"] = "0"
    if "--poison-prepacked-bf16-attention" in forwarded:
        forwarded.remove("--poison-prepacked-bf16-attention")
        os.environ["PG_GPU_CUDNN_PREPACKED_BF16_POISON"] = "1"
    if "--enable-bf16-sparse-xsa-forward" in forwarded:
        forwarded.remove("--enable-bf16-sparse-xsa-forward")
        os.environ["PG_GPU_BF16_SPARSE_XSA_FWD"] = "1"
    if "--disable-bf16-sparse-xsa-forward" in forwarded:
        forwarded.remove("--disable-bf16-sparse-xsa-forward")
        os.environ["PG_GPU_BF16_SPARSE_XSA_FWD"] = "0"
    if "--enable-sparse-xsa-warphead-bwd" in forwarded:
        forwarded.remove("--enable-sparse-xsa-warphead-bwd")
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "1"
    if "--disable-sparse-xsa-warphead-bwd" in forwarded:
        forwarded.remove("--disable-sparse-xsa-warphead-bwd")
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "0"
    if "--disable-host-scalar-updates" in forwarded:
        forwarded.remove("--disable-host-scalar-updates")
        os.environ["PG_GPU_HOST_SCALAR_UPDATES"] = "0"
    if "--enable-host-scalar-updates" in forwarded:
        forwarded.remove("--enable-host-scalar-updates")
        os.environ["PG_GPU_HOST_SCALAR_UPDATES"] = "1"
    if "--enable-bf16-bank-grad-wire" in forwarded:
        forwarded.remove("--enable-bf16-bank-grad-wire")
        os.environ["PG_NCCL_BF16_BANK_GRAD_WIRE"] = "1"
    if "--disable-bf16-bank-grad-wire" in forwarded:
        forwarded.remove("--disable-bf16-bank-grad-wire")
        os.environ["PG_NCCL_BF16_BANK_GRAD_WIRE"] = "0"
    if "--enable-grouped-sharded-grad-collectives" in forwarded:
        forwarded.remove("--enable-grouped-sharded-grad-collectives")
        os.environ["PG_NCCL_GROUP_SHARDED_GRAD_COLLECTIVES"] = "1"
    if "--disable-grouped-sharded-grad-collectives" in forwarded:
        forwarded.remove("--disable-grouped-sharded-grad-collectives")
        os.environ["PG_NCCL_GROUP_SHARDED_GRAD_COLLECTIVES"] = "0"
    if "--enable-nccl-bucket-overlap" in forwarded:
        forwarded.remove("--enable-nccl-bucket-overlap")
        os.environ["PG_NCCL_BUCKET_OVERLAP"] = "1"
    if "--disable-nccl-bucket-overlap" in forwarded:
        forwarded.remove("--disable-nccl-bucket-overlap")
        os.environ["PG_NCCL_BUCKET_OVERLAP"] = "0"
    if "--enable-nccl-side-stream-collectives" in forwarded:
        forwarded.remove("--enable-nccl-side-stream-collectives")
        os.environ["PG_NCCL_SIDE_STREAM_COLLECTIVES"] = "1"
    if "--disable-nccl-side-stream-collectives" in forwarded:
        forwarded.remove("--disable-nccl-side-stream-collectives")
        os.environ["PG_NCCL_SIDE_STREAM_COLLECTIVES"] = "0"
    if "--enable-backward-nccl-bucket-overlap" in forwarded:
        forwarded.remove("--enable-backward-nccl-bucket-overlap")
        os.environ["PG_NCCL_BACKWARD_BUCKET_OVERLAP"] = "1"
        os.environ["PG_NCCL_SIDE_STREAM_COLLECTIVES"] = "1"
    if "--disable-backward-nccl-bucket-overlap" in forwarded:
        forwarded.remove("--disable-backward-nccl-bucket-overlap")
        os.environ["PG_NCCL_BACKWARD_BUCKET_OVERLAP"] = "0"
    if "--disable-fused-attn-residual-from-base" in forwarded:
        forwarded.remove("--disable-fused-attn-residual-from-base")
        os.environ["PG_GPU_FUSED_ATTN_RESIDUAL_FROM_BASE"] = "0"
    if "--disable-fused-parallel-attn-resid-rms" in forwarded:
        forwarded.remove("--disable-fused-parallel-attn-resid-rms")
        os.environ["PG_GPU_FUSED_PARALLEL_ATTN_RESID_RMS"] = "0"
    if "--disable-batched-muon-ns" in forwarded:
        forwarded.remove("--disable-batched-muon-ns")
        os.environ["PG_GPU_MUON_BATCHED_NS"] = "0"
    if "--muon-ns-profile" in forwarded:
        idx = forwarded.index("--muon-ns-profile")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--muon-ns-profile requires simple|quintic|polar_express")
        os.environ["PG_GPU_MUON_NS_PROFILE"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--legacy-muon-ns" in forwarded:
        forwarded.remove("--legacy-muon-ns")
        os.environ["PG_GPU_MUON_NS_PROFILE"] = "simple"
    if "--polar-express-muon-ns" in forwarded:
        forwarded.remove("--polar-express-muon-ns")
        os.environ["PG_GPU_MUON_NS_PROFILE"] = "polar_express"
    if "--disable-cudnn-saved-bf16-attn" in forwarded:
        forwarded.remove("--disable-cudnn-saved-bf16-attn")
        os.environ["PG_GPU_CUDNN_SAVED_BF16_ATTN"] = "0"
    if "--disable-skip-f32-attn-saved-acts" in forwarded:
        forwarded.remove("--disable-skip-f32-attn-saved-acts")
        os.environ["PG_GPU_SKIP_F32_ATTN_SAVED_ACTS"] = "0"
    if "--disable-lean-bf16-saved-acts" in forwarded:
        forwarded.remove("--disable-lean-bf16-saved-acts")
        os.environ["PG_GPU_LEAN_BF16_SAVED_ACTS"] = "0"
    if "--enable-recompute-residual-mix-norm-inputs" in forwarded:
        forwarded.remove("--enable-recompute-residual-mix-norm-inputs")
        os.environ["PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS"] = "1"
    if "--disable-recompute-residual-mix-norm-inputs" in forwarded:
        forwarded.remove("--disable-recompute-residual-mix-norm-inputs")
        os.environ["PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS"] = "0"
    if "--enable-residual-scale-reduce" in forwarded:
        forwarded.remove("--enable-residual-scale-reduce")
        os.environ["PG_GPU_RESIDUAL_SCALE_REDUCE"] = "1"
    if "--disable-residual-scale-reduce" in forwarded:
        forwarded.remove("--disable-residual-scale-reduce")
        os.environ["PG_GPU_RESIDUAL_SCALE_REDUCE"] = "0"
    if "--bf16-shadow-all-gather" in forwarded:
        forwarded.remove("--bf16-shadow-all-gather")
        os.environ["PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER"] = "1"
    if "--disable-bf16-shadow-all-gather" in forwarded:
        forwarded.remove("--disable-bf16-shadow-all-gather")
        os.environ["PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER"] = "0"
    if "--enable-chunked-q-gain-bwd" in forwarded:
        forwarded.remove("--enable-chunked-q-gain-bwd")
        os.environ["PG_GPU_CHUNKED_Q_GAIN_BWD"] = "1"
    if "--disable-chunked-q-gain-bwd" in forwarded:
        forwarded.remove("--disable-chunked-q-gain-bwd")
        os.environ["PG_GPU_CHUNKED_Q_GAIN_BWD"] = "0"
    if "--enable-bf16-attn-backward-tail" in forwarded:
        forwarded.remove("--enable-bf16-attn-backward-tail")
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "1"
    if "--disable-bf16-attn-backward-tail" in forwarded:
        forwarded.remove("--disable-bf16-attn-backward-tail")
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "0"
    if "--enable-bf16-attn-tail-qkv-pack" in forwarded:
        forwarded.remove("--enable-bf16-attn-tail-qkv-pack")
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "1"
    if "--disable-bf16-attn-tail-qkv-pack" in forwarded:
        forwarded.remove("--disable-bf16-attn-tail-qkv-pack")
        os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "0"
    if "--enable-shifted-u16-batch-upload" in forwarded:
        forwarded.remove("--enable-shifted-u16-batch-upload")
        os.environ["PG_GPU_SHIFTED_U16_BATCH_UPLOAD"] = "1"
    if "--disable-shifted-u16-batch-upload" in forwarded:
        forwarded.remove("--disable-shifted-u16-batch-upload")
        os.environ["PG_GPU_SHIFTED_U16_BATCH_UPLOAD"] = "0"
    if "--export-record-shaped-artifact" in forwarded:
        forwarded.remove("--export-record-shaped-artifact")
        os.environ["PG_RECORD_SHAPED_EXPORT_ARTIFACT"] = "1"
    if "--enable-chunked-residual-mix-bwd" in forwarded:
        forwarded.remove("--enable-chunked-residual-mix-bwd")
        os.environ["PG_GPU_CHUNKED_RESIDUAL_MIX_BWD"] = "1"
    if "--disable-chunked-residual-mix-bwd" in forwarded:
        forwarded.remove("--disable-chunked-residual-mix-bwd")
        os.environ["PG_GPU_CHUNKED_RESIDUAL_MIX_BWD"] = "0"
    if "--enable-split-residual-mix-grad" in forwarded:
        forwarded.remove("--enable-split-residual-mix-grad")
        os.environ["PG_GPU_SPLIT_RESIDUAL_MIX_GRAD"] = "1"
    if "--disable-split-residual-mix-grad" in forwarded:
        forwarded.remove("--disable-split-residual-mix-grad")
        os.environ["PG_GPU_SPLIT_RESIDUAL_MIX_GRAD"] = "0"


def _maybe_seed_data_env():
    if os.environ.get("PG_TRAIN_GLOB") and os.environ.get("PG_VAL_GLOB"):
        return

    candidates = [
        os.environ.get("DATA_DIR"),
        "/data/datasets/fineweb10B_sp8192",
        "/data/datasets/fineweb10B_sp1024",
    ]
    for root in candidates:
        if not root:
            continue
        train_glob = os.path.join(root, "fineweb_train_*.bin")
        val_glob = os.path.join(root, "fineweb_val_[0-9]*.bin")
        if not os.environ.get("PG_TRAIN_GLOB") and glob.glob(train_glob):
            os.environ["PG_TRAIN_GLOB"] = train_glob
        if not os.environ.get("PG_VAL_GLOB") and glob.glob(val_glob):
            os.environ["PG_VAL_GLOB"] = val_glob
        if "sp8192" in root and not os.environ.get("PG_TOKENIZER_VOCAB"):
            for vocab_path in (
                os.path.join(root, "tokenizer.vocab"),
                "/data/tokenizers/fineweb_8192_bpe.vocab",
            ):
                if os.path.exists(vocab_path):
                    os.environ["PG_TOKENIZER_VOCAB"] = vocab_path
                    break
        if os.environ.get("PG_TRAIN_GLOB") and os.environ.get("PG_VAL_GLOB"):
            break

    if not os.environ.get("PG_CASEOPS_BYTE_SIDECAR"):
        sidecar_candidates = [
            os.path.join(root, "fineweb_val_bytes_*.bin")
            for root in candidates
            if root
        ]
        for pattern in sidecar_candidates:
            if glob.glob(pattern):
                os.environ["PG_CASEOPS_BYTE_SIDECAR"] = pattern
                break


def _run_pg_train(args: list[str], label: str):
    os.environ["RUST_LOG"] = "info"
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    mode = "smoke"
    if "--mode" in forwarded:
        mode_idx = forwarded.index("--mode")
        if mode_idx + 1 < len(forwarded):
            mode = forwarded[mode_idx + 1]
    if mode == "record-shaped-proxy" and "--allow-unsupported-variants" not in forwarded:
        forwarded.append("--allow-unsupported-variants")
    if os.environ.get("PG_TRAIN_GLOB") and "--train-data" not in forwarded:
        forwarded.extend(["--train-data", os.environ["PG_TRAIN_GLOB"]])
    include_val_data = mode == "record" or os.environ.get("PG_INCLUDE_VAL_DATA") == "1"
    if include_val_data and os.environ.get("PG_VAL_GLOB") and "--val-data" not in forwarded:
        forwarded.extend(["--val-data", os.environ["PG_VAL_GLOB"]])
    if os.environ.get("PG_TOKENIZER_VOCAB") and "--tokenizer-vocab" not in forwarded:
        forwarded.extend(["--tokenizer-vocab", os.environ["PG_TOKENIZER_VOCAB"]])
    if (
        os.environ.get("PG_CASEOPS_BYTE_SIDECAR")
        and "--caseops-byte-sidecar" not in forwarded
    ):
        forwarded.extend(["--caseops-byte-sidecar", os.environ["PG_CASEOPS_BYTE_SIDECAR"]])
    if mode != "record" and "--eval-max-tokens" not in forwarded:
        forwarded.extend(["--eval-max-tokens", os.environ.get("PG_EVAL_MAX_TOKENS", "16384")])
    if not forwarded or forwarded[0] not in {"run", "sweep"}:
        forwarded.insert(0, "run")
    cmd = ["pg-train"] + forwarded
    print(f"Running {label} command:", " ".join(cmd), flush=True)
    print(
        "Data environment:",
        {
            "DATA_DIR": os.environ.get("DATA_DIR"),
            "PG_TRAIN_GLOB": os.environ.get("PG_TRAIN_GLOB"),
            "PG_VAL_GLOB": os.environ.get("PG_VAL_GLOB"),
            "PG_TOKENIZER_VOCAB": os.environ.get("PG_TOKENIZER_VOCAB"),
            "PG_CASEOPS_BYTE_SIDECAR": os.environ.get("PG_CASEOPS_BYTE_SIDECAR"),
        },
        flush=True,
    )

    _write_running_result_json(result_json, label, cmd)
    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)

    proc.wait()
    result = {
        "label": label,
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    result["metrics"] = _parse_key_value_metrics(result["tail"])
    _write_result_json(result_json, result)
    output_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result


def _run_pg_eval(args: list[str]):
    os.environ["RUST_LOG"] = "info"
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    if os.environ.get("PG_VAL_GLOB") and "--val-data" not in forwarded:
        forwarded.extend(["--val-data", os.environ["PG_VAL_GLOB"]])
    if os.environ.get("PG_TOKENIZER_VOCAB") and "--tokenizer-vocab" not in forwarded:
        forwarded.extend(["--tokenizer-vocab", os.environ["PG_TOKENIZER_VOCAB"]])
    if (
        os.environ.get("PG_CASEOPS_BYTE_SIDECAR")
        and "--caseops-byte-sidecar" not in forwarded
    ):
        forwarded.extend(["--caseops-byte-sidecar", os.environ["PG_CASEOPS_BYTE_SIDECAR"]])
    leaderboard_eval = "--leaderboard" in forwarded
    if leaderboard_eval:
        os.environ.setdefault("PG_TTT_AUDIT", "1")
        os.environ.setdefault("PG_TTT_ASSERT_SCORE_NO_MUTATION", "1")
    if leaderboard_eval and "PG_EVAL_GPU_WORLD_SIZE" not in os.environ:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            os.environ["PG_EVAL_GPU_WORLD_SIZE"] = str(
                len([part for part in visible.split(",") if part.strip()])
            )
        else:
            os.environ["PG_EVAL_GPU_WORLD_SIZE"] = "8"
    if (
        not leaderboard_eval
        and os.environ.get("PG_EVAL_MAX_TOKENS")
        and "--max-tokens" not in forwarded
    ):
        forwarded.extend(["--max-tokens", os.environ["PG_EVAL_MAX_TOKENS"]])
    cmd = ["pg-eval"] + forwarded
    print("Running eval command:", " ".join(cmd), flush=True)

    _write_running_result_json(result_json, "eval", cmd)
    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = {
        "label": "eval",
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    result["metrics"] = _parse_key_value_metrics(result["tail"])
    _write_result_json(result_json, result)
    output_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result


def _is_control_piece(piece: str) -> bool:
    return piece in {"<pad>", "<s>", "</s>", "<unk>"} or piece.startswith("<unused")


def _is_byte_piece(piece: str) -> bool:
    return (
        len(piece) == 6
        and piece.startswith("<0x")
        and piece.endswith(">")
        and all(ch in "0123456789abcdefABCDEF" for ch in piece[3:5])
    )


def _build_bpb_luts(vocab_path: str):
    base_bytes: list[int] = []
    has_leading_space: list[bool] = []
    is_boundary: list[bool] = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            piece = line.split("\t", 1)[0]
            boundary = _is_control_piece(piece)
            leading = piece.startswith("▁")
            if boundary:
                nbytes = 0
            elif _is_byte_piece(piece):
                nbytes = 1
            else:
                nbytes = len(piece.lstrip("▁").encode("utf-8"))
            base_bytes.append(min(nbytes, 65535))
            has_leading_space.append(leading)
            is_boundary.append(boundary)
    if not base_bytes:
        raise RuntimeError(f"tokenizer vocab {vocab_path} contained no pieces")
    return base_bytes, has_leading_space, is_boundary


def _token_byte_count(prev: int, target: int, base_bytes, has_leading_space, is_boundary) -> int:
    if target >= len(base_bytes):
        return 1
    nbytes = max(0, base_bytes[target])
    prev_is_boundary = prev >= len(is_boundary) or is_boundary[prev]
    if has_leading_space[target] and not prev_is_boundary:
        nbytes += 1
    return min(nbytes, 65535)


def _read_u16_shard(path: str):
    with open(path, "rb") as f:
        header = f.read(256 * 4)
        if len(header) != 256 * 4:
            raise RuntimeError(f"token shard {path} is missing the 256-int32 header")
        raw = f.read()
    if len(raw) % 2 != 0:
        raise RuntimeError(f"token shard {path} payload has odd byte length")
    tokens = list(struct.unpack(f"<{len(raw) // 2}H", raw))
    return header, tokens


def _write_u16_shard(path: str, header: bytes, values: list[int]):
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack(f"<{len(values)}H", *values))


def _ensure_caseops_byte_sidecars(dataset_dir: str, vocab_path: str) -> str:
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_[0-9]*.bin")))
    if not val_files:
        raise RuntimeError(f"no validation shards found in {dataset_dir}")
    sidecar_pattern = os.path.join(dataset_dir, "fineweb_val_bytes_*.bin")
    sidecars = sorted(glob.glob(sidecar_pattern))
    if len(sidecars) == len(val_files):
        return sidecar_pattern

    print("Generating CaseOps validation byte sidecars", flush=True)
    base_bytes, has_leading_space, is_boundary = _build_bpb_luts(vocab_path)
    prev = 0
    for val_path in val_files:
        header, tokens = _read_u16_shard(val_path)
        byte_counts: list[int] = []
        for tok in tokens:
            byte_counts.append(_token_byte_count(prev, tok, base_bytes, has_leading_space, is_boundary))
            prev = tok
        name = os.path.basename(val_path).replace("fineweb_val_", "fineweb_val_bytes_")
        _write_u16_shard(os.path.join(dataset_dir, name), header, byte_counts)
    return sidecar_pattern

def _run_pg_bench(args: list[str]):
    os.environ["RUST_LOG"] = "info"
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    if not forwarded:
        raise RuntimeError("bench requires a binary name")
    allowed = {
        "parity-kernels": "pg-parity-kernels",
        "parity-forward": "pg-parity-forward",
        "parity-step": "pg-parity-step",
        "gemm-bench": "pg-gemm-bench",
        "attention-bench": "pg-attention-bench",
        "nccl-bench": "pg-nccl-bench",
        "preliminary": "pg-preliminary",
        "smoke": "pg-smoke",
    }
    binary = allowed.get(forwarded[0])
    if binary is None:
        raise RuntimeError(f"unsupported bench binary {forwarded[0]!r}; allowed={sorted(allowed)}")
    cmd = [binary] + list(forwarded[1:])
    print("Running bench command:", " ".join(cmd), flush=True)

    _write_running_result_json(result_json, "bench", cmd)
    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = {
        "label": "bench",
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    result["metrics"] = _parse_key_value_metrics(result["tail"])
    _write_result_json(result_json, result)
    output_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result


def _forwarded_requests_multi_gpu(forwarded: list[str]) -> bool:
    if "--backend" in forwarded:
        idx = forwarded.index("--backend")
        if idx + 1 < len(forwarded) and forwarded[idx + 1] == "cuda-distributed":
            return True
    if "--world-size" in forwarded:
        idx = forwarded.index("--world-size")
        if idx + 1 < len(forwarded):
            try:
                return int(forwarded[idx + 1]) > 1
            except ValueError:
                return False
    return False


@app.function(
    image=image,
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
    },
)
def seed_data():
    from huggingface_hub import snapshot_download

    dataset_dir = "/data/datasets/fineweb10B_sp8192"
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_[0-9]*.bin")))
    vocab_path = "/data/tokenizers/fineweb_8192_bpe.vocab"
    if train_files and val_files and os.path.exists(vocab_path):
        print(
            "SP8192 data already present:",
            {
                "train_files": len(train_files),
                "val_files": len(val_files),
                "vocab_path": vocab_path,
            },
            flush=True,
        )
    else:
        print("Downloading SP8192 shards/tokenizer into pg-data volume", flush=True)
        snapshot_download(
            repo_id="sproos/parameter-golf-tokenizers",
            local_dir="/data",
            allow_patterns=[
                "datasets/fineweb10B_sp8192/*",
                "tokenizers/fineweb_8192_bpe.model",
                "tokenizers/fineweb_8192_bpe.vocab",
            ],
        )

    os.makedirs(dataset_dir, exist_ok=True)
    if os.path.exists(vocab_path):
        shutil.copyfile(vocab_path, os.path.join(dataset_dir, "tokenizer.vocab"))
    sidecar_pattern = None
    if os.path.exists(vocab_path):
        sidecar_pattern = _ensure_caseops_byte_sidecars(dataset_dir, vocab_path)
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_[0-9]*.bin")))
    sidecar_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_bytes_*.bin")))
    result = {
        "dataset_dir": dataset_dir,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "tokenizer_vocab": vocab_path if os.path.exists(vocab_path) else None,
        "caseops_byte_sidecar": sidecar_pattern,
        "caseops_byte_sidecar_files": len(sidecar_files),
    }
    print("Seed-data result:", result, flush=True)
    if not train_files or not val_files or not result["tokenizer_vocab"]:
        raise RuntimeError(f"SP8192 seed incomplete: {result}")
    if len(sidecar_files) != len(val_files):
        raise RuntimeError(f"CaseOps sidecar generation incomplete: {result}")
    data_volume.commit()
    return result

@app.function(
    image=image,
    gpu="H100:1",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_command(args: list[str]):
    return _run_pg_train(args, "single-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_command_multi(args: list[str]):
    return _run_pg_train(args, "multi-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_eval_command(args: list[str]):
    return _run_pg_eval(args)

@app.function(
    image=image,
    gpu="H100:1",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_bench_command(args: list[str]):
    return _run_pg_bench(args)

@app.local_entrypoint()
def main(*args: str):
    use_multi = False
    forwarded = list(args)
    wait_for_result = os.environ.get("PG_WAIT") == "1"
    if forwarded and forwarded[0] == "--modal-wait":
        wait_for_result = True
        forwarded.pop(0)
    if forwarded and forwarded[0] == "seed-data":
        if wait_for_result:
            result = seed_data.remote()
            print("Seed-data result:", result, flush=True)
            return
        call = seed_data.spawn()
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned seed-data Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "eval":
        if wait_for_result:
            result = run_eval_command.remote(forwarded[1:])
            print("Eval result:", result, flush=True)
            return
        call = run_eval_command.spawn(forwarded[1:])
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned eval Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "bench":
        if wait_for_result:
            result = run_bench_command.remote(forwarded[1:])
            print("Bench result:", result, flush=True)
            return
        call = run_bench_command.spawn(forwarded[1:])
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned bench Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "--multi":
        use_multi = True
        forwarded = forwarded[1:]
    if os.environ.get("PG_MULTI_GPU") == "1":
        use_multi = True
    if _forwarded_requests_multi_gpu(forwarded):
        use_multi = True
    print(
        "Dispatching command to detached runner:",
        {"multi_gpu": use_multi, "args": forwarded},
        flush=True,
    )
    if use_multi:
        if wait_for_result:
            result = run_command_multi.remote(forwarded)
            print("Multi-GPU result:", result, flush=True)
            return
        call = run_command_multi.spawn(forwarded)
    else:
        if wait_for_result:
            result = run_command.remote(forwarded)
            print("Single-GPU result:", result, flush=True)
            return
        call = run_command.spawn(forwarded)
    call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
    print("Spawned Modal call:", call_id or call, flush=True)
