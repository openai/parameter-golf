import modal
import sys
import subprocess
import os
import glob
import shutil
import shlex
import struct
import json
import tomllib
from collections import deque

app = modal.App("pg-train-detached")

image = (
    modal.Image.from_dockerfile("deploy/Dockerfile", context_dir=".", add_python="3.12")
    .add_local_dir(
        ".",
        remote_path="/build",
        copy=False,
        ignore=[
            ".git",
            "target",
            ".venv",
            "__pycache__",
            "output",
        ],
    )
)

data_volume = modal.Volume.from_name("pg-data", create_if_missing=True)
output_volume = modal.Volume.from_name("pg-output", create_if_missing=True)
build_cache_volume = modal.Volume.from_name("pg-build-cache", create_if_missing=True)


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


def _pg_train_command() -> list[str]:
    explicit = os.environ.get("PG_TRAIN_BIN")
    if explicit:
        return [explicit]
    source_dir = os.environ.get("PG_SOURCE_DIR", "/build")
    if not os.path.exists(os.path.join(source_dir, "Cargo.toml")):
        for candidate in (
            "/root",
            "/root/parameter-golf-rs",
            "/root/parameter-golf/parameter-golf-rs",
            os.getcwd(),
        ):
            if os.path.exists(os.path.join(candidate, "Cargo.toml")):
                source_dir = candidate
                break
    if not os.path.exists(os.path.join(source_dir, "Cargo.toml")):
        raise RuntimeError(
            "could not locate parameter-golf-rs Cargo.toml; set PG_SOURCE_DIR to the mounted repo"
        )
    target_dir = os.environ.get("CARGO_TARGET_DIR", "/build/target")
    binary = os.path.join(target_dir, "release", "pg-train")
    if os.environ.get("PG_TRAIN_INCREMENTAL_BUILD", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        build_env = os.environ.copy()
        build_env["CARGO_TARGET_DIR"] = target_dir
        if os.environ.get("PG_FORCE_CARGO_CLEAN", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(f"forcing cargo clean for target cache {target_dir}", flush=True)
            subprocess.run(
                ["cargo", "clean", "--target-dir", target_dir],
                cwd=source_dir,
                env=build_env,
                check=True,
            )
        print("compiling pg-train inside Modal function with persistent target cache", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "--features", "cuda", "-p", "pg-train"],
            cwd=source_dir,
            env=build_env,
            check=True,
        )
        build_cache_volume.commit()
    elif not os.path.exists(binary):
        existing = shutil.which("pg-train")
        if existing:
            return [existing]
        raise RuntimeError("pg-train binary missing and PG_TRAIN_INCREMENTAL_BUILD=0")
    if os.environ.get("PG_STRIP_TRAIN_BIN", "1").lower() not in {"0", "false", "no", "off"}:
        strip = shutil.which("strip")
        if strip:
            subprocess.run([strip, binary], check=True)
    return [binary]


def _pg_eval_command() -> list[str]:
    explicit = os.environ.get("PG_EVAL_BIN")
    if explicit:
        return [explicit]
    source_dir = os.environ.get("PG_SOURCE_DIR", "/build")
    if not os.path.exists(os.path.join(source_dir, "Cargo.toml")):
        for candidate in (
            "/root",
            "/root/parameter-golf-rs",
            "/root/parameter-golf/parameter-golf-rs",
            os.getcwd(),
        ):
            if os.path.exists(os.path.join(candidate, "Cargo.toml")):
                source_dir = candidate
                break
    if not os.path.exists(os.path.join(source_dir, "Cargo.toml")):
        raise RuntimeError(
            "could not locate parameter-golf-rs Cargo.toml; set PG_SOURCE_DIR to the mounted repo"
        )
    target_dir = os.environ.get("CARGO_TARGET_DIR", "/build/target")
    binary = os.path.join(target_dir, "release", "pg-eval")
    if os.environ.get("PG_EVAL_INCREMENTAL_BUILD", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        build_env = os.environ.copy()
        build_env["CARGO_TARGET_DIR"] = target_dir
        if os.environ.get("PG_FORCE_CARGO_CLEAN", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(f"forcing cargo clean for target cache {target_dir}", flush=True)
            subprocess.run(
                ["cargo", "clean", "--target-dir", target_dir],
                cwd=source_dir,
                env=build_env,
                check=True,
            )
        print("compiling pg-eval inside Modal function with persistent target cache", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "--features", "cuda", "-p", "pg-eval"],
            cwd=source_dir,
            env=build_env,
            check=True,
        )
        build_cache_volume.commit()
    elif not os.path.exists(binary):
        existing = shutil.which("pg-eval")
        if existing:
            return [existing]
        raise RuntimeError("pg-eval binary missing and PG_EVAL_INCREMENTAL_BUILD=0")
    if os.environ.get("PG_STRIP_EVAL_BIN", "1").lower() not in {"0", "false", "no", "off"}:
        strip = shutil.which("strip")
        if strip:
            subprocess.run([strip, binary], check=True)
    return [binary]


def _prepare_submission_code_dir() -> str:
    """Stage the source files counted by the submission byte budget.

    The Modal runtime mounts the persistent Rust build cache at /build/target.
    Counting /build directly would therefore count build artifacts, not the
    submitted source. Stage an explicit source-only bundle and let pg-train hash
    and count that directory.
    """

    source_dir = os.environ.get("PG_SOURCE_DIR", "/build")
    out_dir = "/tmp/pg_submission_code"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    include_paths = [
        "Cargo.toml",
        "Cargo.lock",
        "crates",
        "specs",
        "deploy/Dockerfile",
        "deploy/run_detached.py",
        "deploy/build_submission.py",
    ]
    ignore = shutil.ignore_patterns(
        "target",
        ".git",
        ".venv",
        ".venv-*",
        "__pycache__",
        "*.pyc",
        "output",
        ".pytest_cache",
    )
    for rel in include_paths:
        src = os.path.join(source_dir, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=ignore)
        else:
            shutil.copy2(src, dst)
    return out_dir


def _forwarded_option(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    idx = args.index(name)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _spec_total_iterations(args: list[str]) -> int | None:
    spec_path = _forwarded_option(args, "--spec")
    if not spec_path:
        return None
    try:
        with open(spec_path, "rb") as f:
            spec = tomllib.load(f)
    except OSError:
        return None
    value = spec.get("train", {}).get("total_iterations")
    if isinstance(value, int) and value > 0:
        return value
    return None


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


def _parse_json_events(text: str) -> dict:
    events: dict[str, list[object]] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.endswith("_json"):
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            events.setdefault(key, []).append({"parse_error": True, "raw": value})
            continue
        events.setdefault(key, []).append(parsed)
    return events


def _merge_json_event_metrics(metrics: dict, json_events: dict) -> dict:
    """Promote scalar JSON event fields into the flat result metrics map.

    The Rust binary emits authoritative record/audit data as single-line JSON
    events. Keep those events intact, but also surface the latest scalar values
    in `metrics` so A/B tooling and CI checks do not have to scrape stdout.
    """

    promoted_prefixes = {
        "record_artifact_audit_json": "artifact",
        "submission_budget_json": "submission",
        "record_audit_json": "audit",
        "run_timing_json": "timing",
    }
    for event_key, prefix in promoted_prefixes.items():
        for event in json_events.get(event_key, []):
            if not isinstance(event, dict):
                continue
            for key, value in event.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metrics[key] = value
                    metrics[f"{prefix}_{key}"] = value
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
    os.environ["PG_CUDA_BACKWARD_GRAPH"] = "0"
    os.environ["PG_CUDA_BACKWARD_GRAPH_STRICT"] = "0"
    os.environ["PG_GPU_BACKWARD_STAGE_TIMING"] = "1" if stage_timing else "0"
    os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "all"
    os.environ["PG_GPU_DIRECT_SAVED_ACTS"] = "1"
    os.environ["PG_GPU_LEAN_FORWARD_CACHE"] = "1"
    os.environ["PG_GPU_BF16_PRIMARY_FORWARD_GEMM"] = "1"
    os.environ["PG_CUBLAS_FAST_TF32"] = "1"
    os.environ.setdefault("PG_CUBLAS_FORCE_TENSOR_OP_ALGO", "1")
    os.environ.setdefault("PG_CUBLAS_BF16_ALGO", "1")
    os.environ.setdefault("PG_CUBLASLT_BF16_GEMM", "1")
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
    os.environ["PG_GPU_FUSED_QKV_ROPE_PREPACK_FWD"] = "1"
    os.environ["PG_GPU_BF16_SPARSE_XSA_FWD"] = "1"
    os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_FWD"] = "1"
    os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "1"
    os.environ["PG_GPU_HOST_SCALAR_UPDATES"] = "0"
    os.environ["PG_GPU_MUON_NS_PROFILE"] = "polar_express"
    os.environ.setdefault("PG_NCCL_BF16_BANK_GRAD_WIRE", "1")
    os.environ.setdefault("PG_NCCL_GROUP_SHARDED_GRAD_COLLECTIVES", "1")
    os.environ.setdefault("PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER", "1")
    os.environ["PG_GPU_TOKEN_RING_SAMPLER"] = "1"
    os.environ["PG_GPU_TOKEN_RING_FULL_SCHEDULE"] = "1"
    os.environ["PG_RECORD_REQUIRE_DEVICE_BATCH"] = "1"
    os.environ["PG_GPU_RESIDUAL_SCALE_REDUCE"] = "1"
    os.environ["PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD"] = "1"
    os.environ["PG_GPU_TILED_RESIDUAL_SCALE_BWD"] = "1"
    os.environ.setdefault("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK", "1024")
    # The graph path zeros non-bank gradients explicitly and lets each layer's
    # first dW contribution overwrite its bank slot. Recurrent layers still
    # accumulate their second tied-weight contribution with beta=1.
    os.environ["PG_GPU_OVERWRITE_BANK_GRADS"] = "1"
    os.environ["PG_GPU_CHUNKED_Q_GAIN_BWD"] = "1"
    os.environ.setdefault("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS", "1024")
    os.environ["PG_GPU_BF16_MLP_DOWN_DX"] = "1"
    os.environ["PG_GPU_VEC4_MLP_ACT_BWD"] = "1"
    os.environ["PG_GPU_FAST_MLP_ACT_BWD"] = "1"
    os.environ["PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS"] = "1"
    os.environ["PG_GPU_OVERLAP_MLP_UP_BWD_GEMMS"] = "1"
    os.environ["PG_GPU_OVERLAP_QKV_BWD_GEMMS"] = "1"
    os.environ["PG_GPU_OVERLAP_ATTN_OUT_BWD_GEMMS"] = "1"
    os.environ["PG_GPU_BF16_BACKWARD_CHAIN"] = "1"
    os.environ.setdefault("PG_GPU_BF16_BACKWARD_CHAIN_STRICT", "1")
    os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "1"
    os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "1"
    os.environ["PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK"] = "1"
    os.environ["PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO"] = "1"
    os.environ.setdefault("PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER", "direct_compact")
    os.environ["PG_GPU_SPLIT_QKV_NORM_RESID_BWD"] = "0"
    os.environ["PG_GPU_CHUNKED_QKV_NORM_RESID_BWD"] = "0"
    os.environ["PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT"] = "1"
    os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "1"
    os.environ["PG_GPU_OUTPUT_CE_BACKEND"] = "chunked_bf16_cache"
    os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "1"
    os.environ.setdefault("PG_GPU_OUTPUT_CE_CHUNK_TOKENS", "8192")
    # v87 H100 A/B regressed this path (267.3 ms/step vs v86 256.8 ms).
    # Keep compact U16 upload as an explicit A/B flag, but do not ship it in
    # the fastest record-shaped profile until the sampler is fully GPU-resident.
    os.environ["PG_GPU_SHIFTED_U16_BATCH_UPLOAD"] = "0"
    os.environ["PG_GPU_SPLIT_RESIDUAL_MIX_GRAD"] = "0"
    os.environ.setdefault("PG_RECORD_TIMING_SKIP_STEPS", "64")


def _apply_gpu_env_flags(forwarded: list[str]):
    explicit_bf16_bank_grad_wire = (
        "--enable-bf16-bank-grad-wire" in forwarded
        or "--disable-bf16-bank-grad-wire" in forwarded
    )
    if "--frontier-fast-record-profile" in forwarded:
        forwarded.remove("--frontier-fast-record-profile")
        _apply_frontier_fast_record_env(stage_timing=True, poison_prepacked_qkv=True)
    if "--force-cargo-clean" in forwarded:
        forwarded.remove("--force-cargo-clean")
        os.environ["PG_FORCE_CARGO_CLEAN"] = "1"
    if "--ttt-audit" in forwarded:
        forwarded.remove("--ttt-audit")
        os.environ["PG_TTT_AUDIT"] = "1"
    if "--assert-score-no-mutation" in forwarded:
        forwarded.remove("--assert-score-no-mutation")
        os.environ["PG_TTT_ASSERT_SCORE_NO_MUTATION"] = "1"
    if "--frontier-throughput-stage-profile" in forwarded:
        forwarded.remove("--frontier-throughput-stage-profile")
        _apply_frontier_fast_record_env(stage_timing=True, poison_prepacked_qkv=False)
    if "--frontier-throughput-record-profile" in forwarded:
        forwarded.remove("--frontier-throughput-record-profile")
        _apply_frontier_fast_record_env(stage_timing=False, poison_prepacked_qkv=False)
    if "--frontier-graph-record-profile" in forwarded:
        forwarded.remove("--frontier-graph-record-profile")
        _apply_frontier_fast_record_env(stage_timing=False, poison_prepacked_qkv=False)
        os.environ["PG_CUDA_BACKWARD_GRAPH"] = "1"
        os.environ["PG_CUDA_BACKWARD_GRAPH_STRICT"] = "1"
        os.environ.setdefault("PG_CUDA_GRAPH_CAPTURE_MODE", "relaxed")
        # Measured H100 active-recurrence profile: graphing the sharded Parallel
        # Muon pre-norm and local-update slices cuts exposed bank-update wall
        # time from ~13 ms/step to ~9 ms/step. Keep this owned by the throughput
        # profile so the best-known graph path is reproducible from one flag.
        os.environ["PG_GPU_SHARDED_MUON_LOCAL_GRAPH"] = "1"
        os.environ["PG_GPU_SHARDED_MUON_PRE_NORM_GRAPH"] = "1"
    if "--chunked-residual-mix-bwd" in forwarded:
        forwarded.remove("--chunked-residual-mix-bwd")
        os.environ["PG_GPU_CHUNKED_RESIDUAL_MIX_BWD"] = "1"
    if "--enable-chunked-qkv-norm-resid-bwd" in forwarded:
        forwarded.remove("--enable-chunked-qkv-norm-resid-bwd")
        os.environ["PG_GPU_CHUNKED_QKV_NORM_RESID_BWD"] = "1"
    if "--disable-chunked-qkv-norm-resid-bwd" in forwarded:
        forwarded.remove("--disable-chunked-qkv-norm-resid-bwd")
        os.environ["PG_GPU_CHUNKED_QKV_NORM_RESID_BWD"] = "0"
    if "--enable-overwrite-bank-grads" in forwarded:
        forwarded.remove("--enable-overwrite-bank-grads")
        os.environ["PG_GPU_OVERWRITE_BANK_GRADS"] = "1"
    if "--disable-overwrite-bank-grads" in forwarded:
        forwarded.remove("--disable-overwrite-bank-grads")
        os.environ["PG_GPU_OVERWRITE_BANK_GRADS"] = "0"
    if "--enable-split-qkv-norm-resid-bwd" in forwarded:
        forwarded.remove("--enable-split-qkv-norm-resid-bwd")
        os.environ["PG_GPU_SPLIT_QKV_NORM_RESID_BWD"] = "1"
    if "--disable-split-qkv-norm-resid-bwd" in forwarded:
        forwarded.remove("--disable-split-qkv-norm-resid-bwd")
        os.environ["PG_GPU_SPLIT_QKV_NORM_RESID_BWD"] = "0"
    if "--enable-split-compact-qkv-norm-resid-bwd" in forwarded:
        forwarded.remove("--enable-split-compact-qkv-norm-resid-bwd")
        os.environ["PG_GPU_SPLIT_QKV_NORM_RESID_BWD"] = "1"
        os.environ["PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT"] = "1"
    if "--qkv-norm-resid-bwd-rows-per-chunk" in forwarded:
        idx = forwarded.index("--qkv-norm-resid-bwd-rows-per-chunk")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--qkv-norm-resid-bwd-rows-per-chunk requires a row count")
        rows = int(forwarded[idx + 1])
        if rows < 256:
            raise RuntimeError("--qkv-norm-resid-bwd-rows-per-chunk must be >=256")
        os.environ["PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK"] = str(rows)
        del forwarded[idx : idx + 2]
    if "--bf16-backward-chain-qkv-norm-resid-reducer" in forwarded:
        idx = forwarded.index("--bf16-backward-chain-qkv-norm-resid-reducer")
        if idx + 1 >= len(forwarded):
            raise RuntimeError(
                "--bf16-backward-chain-qkv-norm-resid-reducer requires direct_compact|split_compact|chunked_compact"
            )
        reducer = forwarded[idx + 1]
        allowed = {"direct_compact", "split_compact", "chunked_compact"}
        if reducer not in allowed:
            raise RuntimeError(
                f"--bf16-backward-chain-qkv-norm-resid-reducer must be one of {sorted(allowed)}, got {reducer!r}"
            )
        os.environ["PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER"] = reducer
        os.environ["PG_GPU_SPLIT_QKV_NORM_RESID_BWD"] = "1" if reducer == "split_compact" else "0"
        os.environ["PG_GPU_CHUNKED_QKV_NORM_RESID_BWD"] = "1" if reducer == "chunked_compact" else "0"
        del forwarded[idx : idx + 2]
    if "--recompute-residual-mix-norm-inputs" in forwarded:
        forwarded.remove("--recompute-residual-mix-norm-inputs")
        os.environ["PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS"] = "1"
    if "--bf16-qkv-dx-output" in forwarded:
        forwarded.remove("--bf16-qkv-dx-output")
        os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "1"
    if "--cuda-event-timing" in forwarded:
        forwarded.remove("--cuda-event-timing")
        os.environ["PG_CUDA_EVENT_TIMING"] = "1"
    if "--disable-cuda-event-timing" in forwarded:
        forwarded.remove("--disable-cuda-event-timing")
        os.environ["PG_CUDA_EVENT_TIMING"] = "0"
    if "--enable-cublaslt-bf16-gemm" in forwarded:
        forwarded.remove("--enable-cublaslt-bf16-gemm")
        os.environ["PG_CUBLASLT_BF16_GEMM"] = "1"
    if "--cublaslt-workspace-mb" in forwarded:
        idx = forwarded.index("--cublaslt-workspace-mb")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--cublaslt-workspace-mb requires an integer MiB value")
        workspace_mb = forwarded[idx + 1]
        try:
            parsed_workspace_mb = int(workspace_mb)
        except ValueError as exc:
            raise RuntimeError(
                f"--cublaslt-workspace-mb requires an integer MiB value, got {workspace_mb!r}"
            ) from exc
        if parsed_workspace_mb < 0:
            raise RuntimeError("--cublaslt-workspace-mb must be non-negative")
        os.environ["PG_CUBLASLT_WORKSPACE_MB"] = str(parsed_workspace_mb)
        del forwarded[idx : idx + 2]
    if "--enable-cublaslt-bf16-strict" in forwarded:
        forwarded.remove("--enable-cublaslt-bf16-strict")
        os.environ["PG_CUBLASLT_BF16_GEMM"] = "1"
        os.environ["PG_CUBLASLT_BF16_STRICT"] = "1"
    if "--disable-cublaslt-bf16-gemm" in forwarded:
        forwarded.remove("--disable-cublaslt-bf16-gemm")
        os.environ["PG_CUBLASLT_BF16_GEMM"] = "0"
        os.environ["PG_CUBLASLT_BF16_STRICT"] = "0"
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
    if "--cuda-graph-capture-debug" in forwarded:
        forwarded.remove("--cuda-graph-capture-debug")
        os.environ["PG_CUDA_GRAPH_CAPTURE_DEBUG"] = "1"
    if "--cuda-graph-disable-cudnn-sdpa" in forwarded:
        forwarded.remove("--cuda-graph-disable-cudnn-sdpa")
        os.environ["PG_CUDA_GRAPH_DISABLE_CUDNN_SDPA"] = "1"
    if "--enable-lean-forward-cache" in forwarded:
        forwarded.remove("--enable-lean-forward-cache")
        os.environ["PG_GPU_LEAN_FORWARD_CACHE"] = "1"
    if "--disable-lean-forward-cache" in forwarded:
        forwarded.remove("--disable-lean-forward-cache")
        os.environ["PG_GPU_LEAN_FORWARD_CACHE"] = "0"
    if "--cuda-graph-capture-mode" in forwarded:
        idx = forwarded.index("--cuda-graph-capture-mode")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--cuda-graph-capture-mode requires thread_local|global|relaxed")
        mode = forwarded[idx + 1]
        allowed = {"thread_local", "global", "relaxed"}
        if mode not in allowed:
            raise RuntimeError(
                f"--cuda-graph-capture-mode must be one of {sorted(allowed)}, got {mode!r}"
            )
        os.environ["PG_CUDA_GRAPH_CAPTURE_MODE"] = mode
        del forwarded[idx : idx + 2]
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
    if "--record-timing-skip-steps" in forwarded:
        idx = forwarded.index("--record-timing-skip-steps")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--record-timing-skip-steps requires a non-negative integer")
        value = int(forwarded[idx + 1])
        if value < 0:
            raise RuntimeError("--record-timing-skip-steps must be >= 0")
        os.environ["PG_RECORD_TIMING_SKIP_STEPS"] = str(value)
        del forwarded[idx : idx + 2]
    if "--record-shaped-proxy-max-steps" in forwarded:
        idx = forwarded.index("--record-shaped-proxy-max-steps")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--record-shaped-proxy-max-steps requires a positive integer")
        value = int(forwarded[idx + 1])
        if value <= 0:
            raise RuntimeError("--record-shaped-proxy-max-steps must be > 0")
        os.environ["PG_RECORD_SHAPED_PROXY_MAX_STEPS"] = str(value)
        del forwarded[idx : idx + 2]
    if "--record-max-ms-per-step" in forwarded:
        idx = forwarded.index("--record-max-ms-per-step")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--record-max-ms-per-step requires a numeric ceiling")
        os.environ["PG_RECORD_MAX_MS_PER_STEP"] = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    if "--debug-eval-in-memory-model" in forwarded:
        forwarded.remove("--debug-eval-in-memory-model")
        os.environ["PG_DEBUG_EVAL_IN_MEMORY_MODEL"] = "1"
    if "--force-recurrence-active" in forwarded:
        forwarded.remove("--force-recurrence-active")
        os.environ["PG_FORCE_RECURRENCE_ACTIVE"] = "1"
    if "--force-recurrence-inactive" in forwarded:
        forwarded.remove("--force-recurrence-inactive")
        os.environ["PG_FORCE_RECURRENCE_INACTIVE"] = "1"
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
    if "--enable-overlap-linear-bwd-gemms" in forwarded:
        forwarded.remove("--enable-overlap-linear-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_LINEAR_BWD_GEMMS"] = "1"
    if "--disable-overlap-linear-bwd-gemms" in forwarded:
        forwarded.remove("--disable-overlap-linear-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_LINEAR_BWD_GEMMS"] = "0"
    if "--enable-graph-side-gemm-capture" in forwarded:
        forwarded.remove("--enable-graph-side-gemm-capture")
        os.environ["PG_GPU_GRAPH_SIDE_GEMM_CAPTURE"] = "1"
        os.environ.setdefault("PG_CUDA_GRAPH_CAPTURE_MODE", "relaxed")
    if "--disable-graph-side-gemm-capture" in forwarded:
        forwarded.remove("--disable-graph-side-gemm-capture")
        os.environ["PG_GPU_GRAPH_SIDE_GEMM_CAPTURE"] = "0"
    if "--enable-overlap-mlp-bwd-gemms" in forwarded:
        forwarded.remove("--enable-overlap-mlp-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS"] = "1"
        os.environ["PG_GPU_OVERLAP_MLP_UP_BWD_GEMMS"] = "1"
    if "--disable-overlap-mlp-bwd-gemms" in forwarded:
        forwarded.remove("--disable-overlap-mlp-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS"] = "0"
        os.environ["PG_GPU_OVERLAP_MLP_UP_BWD_GEMMS"] = "0"
    if "--enable-overlap-mlp-down-bwd-gemms" in forwarded:
        forwarded.remove("--enable-overlap-mlp-down-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS"] = "1"
    if "--enable-overlap-mlp-up-bwd-gemms" in forwarded:
        forwarded.remove("--enable-overlap-mlp-up-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_MLP_UP_BWD_GEMMS"] = "1"
    if "--enable-overlap-qkv-bwd-gemms" in forwarded:
        forwarded.remove("--enable-overlap-qkv-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_QKV_BWD_GEMMS"] = "1"
    if "--disable-overlap-qkv-bwd-gemms" in forwarded:
        forwarded.remove("--disable-overlap-qkv-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_QKV_BWD_GEMMS"] = "0"
    if "--enable-overlap-attn-out-bwd-gemms" in forwarded:
        forwarded.remove("--enable-overlap-attn-out-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_ATTN_OUT_BWD_GEMMS"] = "1"
    if "--disable-overlap-attn-out-bwd-gemms" in forwarded:
        forwarded.remove("--disable-overlap-attn-out-bwd-gemms")
        os.environ["PG_GPU_OVERLAP_ATTN_OUT_BWD_GEMMS"] = "0"
    if "--enable-compact-attn-gate-grad-input" in forwarded:
        forwarded.remove("--enable-compact-attn-gate-grad-input")
        os.environ["PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT"] = "1"
    if "--disable-compact-attn-gate-grad-input" in forwarded:
        forwarded.remove("--disable-compact-attn-gate-grad-input")
        os.environ["PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT"] = "0"
    if "--enable-fast-mlp-act-bwd" in forwarded:
        forwarded.remove("--enable-fast-mlp-act-bwd")
        os.environ["PG_GPU_FAST_MLP_ACT_BWD"] = "1"
    if "--disable-fast-mlp-act-bwd" in forwarded:
        forwarded.remove("--disable-fast-mlp-act-bwd")
        os.environ["PG_GPU_FAST_MLP_ACT_BWD"] = "0"
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
        os.environ["PG_GPU_OUTPUT_CE_BACKEND"] = "tiled_repeated_gemm"
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "1"
    if "--disable-tiled-output-ce" in forwarded:
        forwarded.remove("--disable-tiled-output-ce")
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    if "--enable-chunked-output-ce-cache" in forwarded:
        forwarded.remove("--enable-chunked-output-ce-cache")
        os.environ["PG_GPU_OUTPUT_CE_BACKEND"] = "chunked_bf16_cache"
        os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "1"
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    if "--disable-chunked-output-ce-cache" in forwarded:
        forwarded.remove("--disable-chunked-output-ce-cache")
        os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "0"
    if "--enable-fused-exact-output-ce" in forwarded:
        forwarded.remove("--enable-fused-exact-output-ce")
        os.environ["PG_GPU_OUTPUT_CE_BACKEND"] = "fused_exact_wmma"
        os.environ["PG_GPU_FUSED_EXACT_OUTPUT_CE"] = "1"
        os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "0"
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "0"
    if "--disable-fused-exact-output-ce" in forwarded:
        forwarded.remove("--disable-fused-exact-output-ce")
        os.environ["PG_GPU_FUSED_EXACT_OUTPUT_CE"] = "0"
        os.environ["PG_GPU_OUTPUT_CE_BACKEND"] = "chunked_bf16_cache"
    if "--output-ce-backend" in forwarded:
        idx = forwarded.index("--output-ce-backend")
        if idx + 1 >= len(forwarded):
            raise RuntimeError(
                "--output-ce-backend requires chunked_bf16_cache|tiled_repeated_gemm|fused_exact_wmma"
            )
        backend = forwarded[idx + 1]
        allowed = {"chunked_bf16_cache", "tiled_repeated_gemm", "fused_exact_wmma"}
        if backend not in allowed:
            raise RuntimeError(f"--output-ce-backend must be one of {sorted(allowed)}, got {backend!r}")
        os.environ["PG_GPU_OUTPUT_CE_BACKEND"] = backend
        os.environ["PG_GPU_CHUNKED_OUTPUT_CE_CACHE"] = "1" if backend == "chunked_bf16_cache" else "0"
        os.environ["PG_GPU_TILED_OUTPUT_CE"] = "1" if backend == "tiled_repeated_gemm" else "0"
        os.environ["PG_GPU_FUSED_EXACT_OUTPUT_CE"] = "1" if backend == "fused_exact_wmma" else "0"
        del forwarded[idx : idx + 2]
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
    if "--enable-record-require-device-batch" in forwarded:
        forwarded.remove("--enable-record-require-device-batch")
        os.environ["PG_RECORD_REQUIRE_DEVICE_BATCH"] = "1"
    if "--disable-record-require-device-batch" in forwarded:
        forwarded.remove("--disable-record-require-device-batch")
        os.environ["PG_RECORD_REQUIRE_DEVICE_BATCH"] = "0"
    if "--enable-fused-qkv-rope-prepack-fwd" in forwarded:
        forwarded.remove("--enable-fused-qkv-rope-prepack-fwd")
        os.environ["PG_GPU_FUSED_QKV_ROPE_PREPACK_FWD"] = "1"
    if "--disable-fused-qkv-rope-prepack-fwd" in forwarded:
        forwarded.remove("--disable-fused-qkv-rope-prepack-fwd")
        os.environ["PG_GPU_FUSED_QKV_ROPE_PREPACK_FWD"] = "0"
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
    if "--enable-sparse-xsa-warphead-fwd" in forwarded:
        forwarded.remove("--enable-sparse-xsa-warphead-fwd")
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_FWD"] = "1"
    if "--disable-sparse-xsa-warphead-fwd" in forwarded:
        forwarded.remove("--disable-sparse-xsa-warphead-fwd")
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_FWD"] = "0"
    if "--enable-bigram-embed-merge" in forwarded:
        forwarded.remove("--enable-bigram-embed-merge")
        os.environ["PG_GPU_BIGRAM_EMBED_MERGE"] = "1"
    if "--disable-bigram-embed-merge" in forwarded:
        forwarded.remove("--disable-bigram-embed-merge")
        os.environ["PG_GPU_BIGRAM_EMBED_MERGE"] = "0"
    if "--enable-sparse-xsa-warphead-bwd" in forwarded:
        forwarded.remove("--enable-sparse-xsa-warphead-bwd")
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "1"
    if "--disable-sparse-xsa-warphead-bwd" in forwarded:
        forwarded.remove("--disable-sparse-xsa-warphead-bwd")
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "0"
    if "--enable-sparse-xsa-grouped-kv-bwd" in forwarded:
        forwarded.remove("--enable-sparse-xsa-grouped-kv-bwd")
        os.environ["PG_GPU_SPARSE_XSA_GROUPED_KV_BWD"] = "1"
        os.environ["PG_GPU_SPARSE_XSA_WARPHEAD_BWD"] = "1"
    if "--disable-sparse-xsa-grouped-kv-bwd" in forwarded:
        forwarded.remove("--disable-sparse-xsa-grouped-kv-bwd")
        os.environ["PG_GPU_SPARSE_XSA_GROUPED_KV_BWD"] = "0"
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
        # Per-layer/bucket overlap launches collectives as soon as gradients
        # land. BF16-on-wire requires F32->BF16 packing on the main GEMM stream
        # before each bucket and has measured slower than F32 wire for this
        # path. Keep BF16 wire explicit for A/B only.
        if not explicit_bf16_bank_grad_wire:
            os.environ["PG_NCCL_BF16_BANK_GRAD_WIRE"] = "0"
    if "--disable-backward-nccl-bucket-overlap" in forwarded:
        forwarded.remove("--disable-backward-nccl-bucket-overlap")
        os.environ["PG_NCCL_BACKWARD_BUCKET_OVERLAP"] = "0"
    if "--backward-nccl-bucket-layers" in forwarded:
        idx = forwarded.index("--backward-nccl-bucket-layers")
        try:
            value = forwarded[idx + 1]
        except IndexError as exc:
            raise ValueError("--backward-nccl-bucket-layers requires a positive integer") from exc
        del forwarded[idx : idx + 2]
        if int(value) <= 0:
            raise ValueError("--backward-nccl-bucket-layers requires a positive integer")
        os.environ["PG_NCCL_BACKWARD_BUCKET_LAYERS"] = value
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
    if "--muon-ns-steps" in forwarded:
        idx = forwarded.index("--muon-ns-steps")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--muon-ns-steps requires a positive integer")
        os.environ["PG_GPU_MUON_NS_STEPS"] = forwarded[idx + 1]
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
    if "--enable-chunked-residual-scale-bwd" in forwarded:
        forwarded.remove("--enable-chunked-residual-scale-bwd")
        os.environ["PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD"] = "1"
    if "--disable-chunked-residual-scale-bwd" in forwarded:
        forwarded.remove("--disable-chunked-residual-scale-bwd")
        os.environ["PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD"] = "0"
    if "--enable-tiled-residual-scale-bwd" in forwarded:
        forwarded.remove("--enable-tiled-residual-scale-bwd")
        os.environ["PG_GPU_TILED_RESIDUAL_SCALE_BWD"] = "1"
        os.environ["PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD"] = "1"
    if "--disable-tiled-residual-scale-bwd" in forwarded:
        forwarded.remove("--disable-tiled-residual-scale-bwd")
        os.environ["PG_GPU_TILED_RESIDUAL_SCALE_BWD"] = "0"
    if "--enable-bf16-mlp-down-dx" in forwarded:
        forwarded.remove("--enable-bf16-mlp-down-dx")
        os.environ["PG_GPU_BF16_MLP_DOWN_DX"] = "1"
    if "--disable-bf16-mlp-down-dx" in forwarded:
        forwarded.remove("--disable-bf16-mlp-down-dx")
        os.environ["PG_GPU_BF16_MLP_DOWN_DX"] = "0"
    if "--enable-vec2-mlp-act-bwd" in forwarded:
        forwarded.remove("--enable-vec2-mlp-act-bwd")
        os.environ["PG_GPU_VEC2_MLP_ACT_BWD"] = "1"
    if "--disable-vec2-mlp-act-bwd" in forwarded:
        forwarded.remove("--disable-vec2-mlp-act-bwd")
        os.environ["PG_GPU_VEC2_MLP_ACT_BWD"] = "0"
    if "--enable-vec4-mlp-act-bwd" in forwarded:
        forwarded.remove("--enable-vec4-mlp-act-bwd")
        os.environ["PG_GPU_VEC4_MLP_ACT_BWD"] = "1"
    if "--disable-vec4-mlp-act-bwd" in forwarded:
        forwarded.remove("--disable-vec4-mlp-act-bwd")
        os.environ["PG_GPU_VEC4_MLP_ACT_BWD"] = "0"
    if "--bf16-shadow-all-gather" in forwarded:
        forwarded.remove("--bf16-shadow-all-gather")
        os.environ["PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER"] = "1"
    if "--disable-bf16-shadow-all-gather" in forwarded:
        forwarded.remove("--disable-bf16-shadow-all-gather")
        os.environ["PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER"] = "0"
    if "--enable-sharded-muon-fused-global-clip" in forwarded:
        forwarded.remove("--enable-sharded-muon-fused-global-clip")
        os.environ["PG_GPU_SHARDED_MUON_FUSED_GLOBAL_CLIP"] = "1"
    if "--disable-sharded-muon-fused-global-clip" in forwarded:
        forwarded.remove("--disable-sharded-muon-fused-global-clip")
        os.environ["PG_GPU_SHARDED_MUON_FUSED_GLOBAL_CLIP"] = "0"
    if "--enable-sharded-muon-phase-timing" in forwarded:
        forwarded.remove("--enable-sharded-muon-phase-timing")
        os.environ["PG_GPU_SHARDED_MUON_PHASE_TIMING"] = "1"
    if "--disable-sharded-muon-phase-timing" in forwarded:
        forwarded.remove("--disable-sharded-muon-phase-timing")
        os.environ["PG_GPU_SHARDED_MUON_PHASE_TIMING"] = "0"
    if "--enable-sharded-muon-parallel-local" in forwarded:
        forwarded.remove("--enable-sharded-muon-parallel-local")
        os.environ["PG_GPU_SHARDED_MUON_PARALLEL_LOCAL"] = "1"
    if "--disable-sharded-muon-parallel-local" in forwarded:
        forwarded.remove("--disable-sharded-muon-parallel-local")
        os.environ["PG_GPU_SHARDED_MUON_PARALLEL_LOCAL"] = "0"
    if "--enable-sharded-muon-local-graph" in forwarded:
        forwarded.remove("--enable-sharded-muon-local-graph")
        os.environ["PG_GPU_SHARDED_MUON_LOCAL_GRAPH"] = "1"
    if "--disable-sharded-muon-local-graph" in forwarded:
        forwarded.remove("--disable-sharded-muon-local-graph")
        os.environ["PG_GPU_SHARDED_MUON_LOCAL_GRAPH"] = "0"
    if "--enable-sharded-muon-pre-norm-graph" in forwarded:
        forwarded.remove("--enable-sharded-muon-pre-norm-graph")
        os.environ["PG_GPU_SHARDED_MUON_PRE_NORM_GRAPH"] = "1"
    if "--disable-sharded-muon-pre-norm-graph" in forwarded:
        forwarded.remove("--disable-sharded-muon-pre-norm-graph")
        os.environ["PG_GPU_SHARDED_MUON_PRE_NORM_GRAPH"] = "0"
    if "--enable-adamw-bf16-shadow-update" in forwarded:
        forwarded.remove("--enable-adamw-bf16-shadow-update")
        os.environ["PG_GPU_ADAMW_BF16_SHADOW_UPDATE"] = "1"
    if "--disable-adamw-bf16-shadow-update" in forwarded:
        forwarded.remove("--disable-adamw-bf16-shadow-update")
        os.environ["PG_GPU_ADAMW_BF16_SHADOW_UPDATE"] = "0"
    if "--enable-deferred-weight-gemms" in forwarded:
        forwarded.remove("--enable-deferred-weight-gemms")
        os.environ["PG_GPU_DEFER_MLP_UP_BWD_DW"] = "1"
        os.environ["PG_GPU_DEFER_ATTN_OUT_BWD_DW"] = "1"
    if "--enable-deferred-all-weight-gemms" in forwarded:
        forwarded.remove("--enable-deferred-all-weight-gemms")
        os.environ["PG_GPU_DEFER_LINEAR_BACKWARD_WEIGHT_GEMMS"] = "1"
        os.environ["PG_GPU_DEFER_QKV_BWD_DW"] = "1"
    if "--disable-deferred-weight-gemms" in forwarded:
        forwarded.remove("--disable-deferred-weight-gemms")
        os.environ["PG_GPU_DEFER_LINEAR_BACKWARD_WEIGHT_GEMMS"] = "0"
        os.environ["PG_GPU_DEFER_MLP_DOWN_BWD_DW"] = "0"
        os.environ["PG_GPU_DEFER_MLP_UP_BWD_DW"] = "0"
        os.environ["PG_GPU_DEFER_QKV_BWD_DW"] = "0"
        os.environ["PG_GPU_DEFER_ATTN_OUT_BWD_DW"] = "0"
    if "--enable-chunked-q-gain-bwd" in forwarded:
        forwarded.remove("--enable-chunked-q-gain-bwd")
        os.environ["PG_GPU_CHUNKED_Q_GAIN_BWD"] = "1"
    if "--disable-chunked-q-gain-bwd" in forwarded:
        forwarded.remove("--disable-chunked-q-gain-bwd")
        os.environ["PG_GPU_CHUNKED_Q_GAIN_BWD"] = "0"
    if "--q-gain-bwd-chunk-tokens" in forwarded:
        idx = forwarded.index("--q-gain-bwd-chunk-tokens")
        try:
            value = forwarded[idx + 1]
        except IndexError as exc:
            raise ValueError("--q-gain-bwd-chunk-tokens requires an integer >= 256") from exc
        del forwarded[idx : idx + 2]
        if int(value) < 256:
            raise ValueError("--q-gain-bwd-chunk-tokens requires an integer >= 256")
        os.environ["PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS"] = value
    if "--enable-combined-qkv-rope-tail-bwd" in forwarded:
        forwarded.remove("--enable-combined-qkv-rope-tail-bwd")
        os.environ["PG_GPU_COMBINED_QKV_ROPE_TAIL_BWD"] = "1"
    if "--disable-combined-qkv-rope-tail-bwd" in forwarded:
        forwarded.remove("--disable-combined-qkv-rope-tail-bwd")
        os.environ["PG_GPU_COMBINED_QKV_ROPE_TAIL_BWD"] = "0"
    if "--residual-scale-bwd-rows-per-chunk" in forwarded:
        idx = forwarded.index("--residual-scale-bwd-rows-per-chunk")
        try:
            value = forwarded[idx + 1]
        except IndexError as exc:
            raise ValueError(
                "--residual-scale-bwd-rows-per-chunk requires an integer >= 64"
            ) from exc
        del forwarded[idx : idx + 2]
        if int(value) < 64:
            raise ValueError("--residual-scale-bwd-rows-per-chunk requires an integer >= 64")
        os.environ["PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK"] = value
    if "--enable-bf16-backward-chain" in forwarded:
        forwarded.remove("--enable-bf16-backward-chain")
        os.environ["PG_GPU_BF16_BACKWARD_CHAIN"] = "1"
        os.environ.setdefault("PG_GPU_BF16_BACKWARD_CHAIN_STRICT", "1")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "all"
        os.environ["PG_GPU_DIRECT_SAVED_ACTS"] = "1"
        os.environ["PG_GPU_CUDNN_PREPACKED_BF16_ATTN"] = "1"
        os.environ["PG_GPU_FUSED_QKV_PROJ"] = "1"
        os.environ["PG_GPU_FUSED_QKV_PROJ_RECORD_OK"] = "1"
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK"] = "1"
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO"] = "1"
        os.environ["PG_GPU_BF16_QKV_DX_OUTPUT"] = "1"
        os.environ["PG_GPU_BF16_MLP_DOWN_DX"] = "1"
        reducer = os.environ.setdefault(
            "PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER", "direct_compact"
        )
        os.environ["PG_GPU_SPLIT_QKV_NORM_RESID_BWD"] = (
            "1" if reducer == "split_compact" else "0"
        )
        os.environ["PG_GPU_CHUNKED_QKV_NORM_RESID_BWD"] = (
            "1" if reducer == "chunked_compact" else "0"
        )
        os.environ["PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT"] = "1"
        os.environ["PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS"] = "0"
        os.environ["PG_GPU_SPLIT_RESIDUAL_MIX_GRAD"] = "0"
    if "--disable-bf16-backward-chain" in forwarded:
        forwarded.remove("--disable-bf16-backward-chain")
        os.environ["PG_GPU_BF16_BACKWARD_CHAIN"] = "0"
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "0"
        os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "0"
        os.environ["PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK"] = "0"
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO"] = "0"
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
    if "--enable-bf16-attn-tail-direct-qkv-pack" in forwarded:
        forwarded.remove("--enable-bf16-attn-tail-direct-qkv-pack")
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK"] = "1"
    if "--disable-bf16-attn-tail-direct-qkv-pack" in forwarded:
        forwarded.remove("--disable-bf16-attn-tail-direct-qkv-pack")
        os.environ["PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK"] = "0"
    if "--enable-bf16-attn-bhsd-do" in forwarded:
        forwarded.remove("--enable-bf16-attn-bhsd-do")
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_TAIL"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_QKV_PACK"] = "1"
        os.environ["PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK"] = "1"
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO"] = "1"
    if "--disable-bf16-attn-bhsd-do" in forwarded:
        forwarded.remove("--disable-bf16-attn-bhsd-do")
        os.environ["PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO"] = "0"
    if "--enable-shifted-u16-batch-upload" in forwarded:
        forwarded.remove("--enable-shifted-u16-batch-upload")
        os.environ["PG_GPU_SHIFTED_U16_BATCH_UPLOAD"] = "1"
    if "--disable-shifted-u16-batch-upload" in forwarded:
        forwarded.remove("--disable-shifted-u16-batch-upload")
        os.environ["PG_GPU_SHIFTED_U16_BATCH_UPLOAD"] = "0"
    if "--enable-gpu-resident-synthetic-sampler" in forwarded:
        forwarded.remove("--enable-gpu-resident-synthetic-sampler")
        os.environ["PG_GPU_RESIDENT_SYNTHETIC_SAMPLER"] = "1"
        os.environ["PG_SYNTHETIC_TRAIN_DATA"] = "1"
    if "--disable-gpu-resident-synthetic-sampler" in forwarded:
        forwarded.remove("--disable-gpu-resident-synthetic-sampler")
        os.environ["PG_GPU_RESIDENT_SYNTHETIC_SAMPLER"] = "0"
    if "--enable-gpu-token-ring-sampler" in forwarded:
        forwarded.remove("--enable-gpu-token-ring-sampler")
        os.environ["PG_GPU_TOKEN_RING_SAMPLER"] = "1"
    if "--disable-gpu-token-ring-sampler" in forwarded:
        forwarded.remove("--disable-gpu-token-ring-sampler")
        os.environ["PG_GPU_TOKEN_RING_SAMPLER"] = "0"
    if "--gpu-token-ring-steps" in forwarded:
        idx = forwarded.index("--gpu-token-ring-steps")
        try:
            value = forwarded[idx + 1]
        except IndexError as exc:
            raise ValueError("--gpu-token-ring-steps requires a positive integer") from exc
        del forwarded[idx : idx + 2]
        if int(value) <= 0:
            raise ValueError("--gpu-token-ring-steps requires a positive integer")
        os.environ["PG_GPU_TOKEN_RING_STEPS"] = value
    if "--enable-gpu-token-full-schedule" in forwarded:
        forwarded.remove("--enable-gpu-token-full-schedule")
        os.environ["PG_GPU_TOKEN_RING_SAMPLER"] = "1"
        os.environ["PG_GPU_TOKEN_RING_FULL_SCHEDULE"] = "1"
        if "PG_GPU_TOKEN_RING_STEPS" not in os.environ:
            total_iterations = _forwarded_option(forwarded, "--total-iterations")
            if total_iterations is None:
                spec_iterations = _spec_total_iterations(forwarded)
                if spec_iterations is not None:
                    total_iterations = str(spec_iterations)
            if total_iterations is None:
                raise ValueError(
                    "--enable-gpu-token-full-schedule requires --total-iterations, --gpu-token-ring-steps, or a spec with [train].total_iterations"
                )
            if int(total_iterations) <= 0:
                raise ValueError("--total-iterations requires a positive integer")
            os.environ["PG_GPU_TOKEN_RING_STEPS"] = total_iterations
    if "--disable-gpu-token-full-schedule" in forwarded:
        forwarded.remove("--disable-gpu-token-full-schedule")
        os.environ["PG_GPU_TOKEN_RING_FULL_SCHEDULE"] = "0"
    if (
        os.environ.get("PG_GPU_TOKEN_RING_FULL_SCHEDULE", "").lower()
        in {"1", "true", "yes", "on"}
        and "PG_GPU_TOKEN_RING_STEPS" not in os.environ
    ):
        total_iterations = _forwarded_option(forwarded, "--total-iterations")
        if total_iterations is None:
            spec_iterations = _spec_total_iterations(forwarded)
            if spec_iterations is not None:
                total_iterations = str(spec_iterations)
        if total_iterations is None:
            raise ValueError(
                "PG_GPU_TOKEN_RING_FULL_SCHEDULE=1 requires --total-iterations, "
                "--gpu-token-ring-steps, or a spec with [train].total_iterations"
            )
        if int(total_iterations) <= 0:
            raise ValueError("--total-iterations requires a positive integer")
        os.environ["PG_GPU_TOKEN_RING_STEPS"] = total_iterations
    if "--require-gpu-data-sampler" in forwarded:
        forwarded.remove("--require-gpu-data-sampler")
        os.environ["PG_RECORD_REQUIRE_GPU_DATA_SAMPLER"] = "1"
    if "--allow-host-data-sampler" in forwarded:
        forwarded.remove("--allow-host-data-sampler")
        os.environ["PG_RECORD_REQUIRE_GPU_DATA_SAMPLER"] = "0"
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
    if "--skip-residual-mix-grad" in forwarded:
        forwarded.remove("--skip-residual-mix-grad")
        os.environ["PG_GPU_SKIP_RESIDUAL_MIX_GRAD"] = "1"
    if "--keep-residual-mix-grad" in forwarded:
        forwarded.remove("--keep-residual-mix-grad")
        os.environ["PG_GPU_SKIP_RESIDUAL_MIX_GRAD"] = "0"
    if "--skip-recurrent-pass1-bank-grads" in forwarded:
        forwarded.remove("--skip-recurrent-pass1-bank-grads")
        os.environ["PG_GPU_SKIP_RECURRENT_PASS1_BANK_GRADS"] = "1"
    if "--keep-recurrent-pass1-bank-grads" in forwarded:
        forwarded.remove("--keep-recurrent-pass1-bank-grads")
        os.environ["PG_GPU_SKIP_RECURRENT_PASS1_BANK_GRADS"] = "0"
    if "--recurrent-pass1-straight-through" in forwarded:
        forwarded.remove("--recurrent-pass1-straight-through")
        os.environ["PG_GPU_RECURRENT_PASS1_STRAIGHT_THROUGH"] = "1"
    if "--disable-recurrent-pass1-straight-through" in forwarded:
        forwarded.remove("--disable-recurrent-pass1-straight-through")
        os.environ["PG_GPU_RECURRENT_PASS1_STRAIGHT_THROUGH"] = "0"
    if "--recurrent-all-straight-through" in forwarded:
        forwarded.remove("--recurrent-all-straight-through")
        os.environ["PG_GPU_RECURRENT_ALL_STRAIGHT_THROUGH"] = "1"
    if "--disable-recurrent-all-straight-through" in forwarded:
        forwarded.remove("--disable-recurrent-all-straight-through")
        os.environ["PG_GPU_RECURRENT_ALL_STRAIGHT_THROUGH"] = "0"


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
    os.environ.setdefault("RUST_BACKTRACE", "1")
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    mode = "smoke"
    if "--mode" in forwarded:
        mode_idx = forwarded.index("--mode")
        if mode_idx + 1 < len(forwarded):
            mode = forwarded[mode_idx + 1]
    if mode == "record" and "PG_FORCE_CARGO_CLEAN" not in os.environ:
        os.environ["PG_FORCE_CARGO_CLEAN"] = "1"
    if (
        (mode == "record" or os.environ.get("PG_RECORD_SHAPED_EXPORT_ARTIFACT") == "1")
        and "PG_SUBMISSION_CODE_BYTES" not in os.environ
        and "PG_SUBMISSION_CODE_DIR" not in os.environ
    ):
        os.environ["PG_SUBMISSION_CODE_DIR"] = _prepare_submission_code_dir()
    if mode == "record-shaped-proxy" and "--allow-unsupported-variants" not in forwarded:
        forwarded.append("--allow-unsupported-variants")
    use_synthetic_train_data = os.environ.get("PG_SYNTHETIC_TRAIN_DATA") == "1"
    if (
        not use_synthetic_train_data
        and os.environ.get("PG_TRAIN_GLOB")
        and "--train-data" not in forwarded
    ):
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
    cmd = _pg_train_command() + forwarded
    print(f"Running {label} command:", " ".join(cmd), flush=True)
    print(
        "Data environment:",
        {
            "DATA_DIR": os.environ.get("DATA_DIR"),
            "PG_TRAIN_GLOB": os.environ.get("PG_TRAIN_GLOB"),
            "PG_VAL_GLOB": os.environ.get("PG_VAL_GLOB"),
            "PG_TOKENIZER_VOCAB": os.environ.get("PG_TOKENIZER_VOCAB"),
            "PG_CASEOPS_BYTE_SIDECAR": os.environ.get("PG_CASEOPS_BYTE_SIDECAR"),
            "PG_SYNTHETIC_TRAIN_DATA": os.environ.get("PG_SYNTHETIC_TRAIN_DATA"),
            "PG_GPU_RESIDENT_SYNTHETIC_SAMPLER": os.environ.get("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER"),
            "PG_GPU_TOKEN_RING_SAMPLER": os.environ.get("PG_GPU_TOKEN_RING_SAMPLER"),
            "PG_GPU_TOKEN_RING_FULL_SCHEDULE": os.environ.get("PG_GPU_TOKEN_RING_FULL_SCHEDULE"),
            "PG_GPU_TOKEN_RING_STEPS": os.environ.get("PG_GPU_TOKEN_RING_STEPS"),
            "PG_RECORD_REQUIRE_GPU_DATA_SAMPLER": os.environ.get("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER"),
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
    result["json_events"] = _parse_json_events(result["tail"])
    result["metrics"] = _merge_json_event_metrics(
        _parse_key_value_metrics(result["tail"]), result["json_events"]
    )
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
    if "--force-cargo-clean" in forwarded:
        forwarded.remove("--force-cargo-clean")
        os.environ["PG_FORCE_CARGO_CLEAN"] = "1"
    requested_eval_world_size = None
    if "--eval-gpu-world-size" in forwarded:
        idx = forwarded.index("--eval-gpu-world-size")
        if idx + 1 < len(forwarded):
            requested_eval_world_size = forwarded[idx + 1]
    _apply_gpu_env_flags(forwarded)
    if requested_eval_world_size is not None:
        os.environ["PG_EVAL_GPU_WORLD_SIZE"] = requested_eval_world_size
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
    os.environ.setdefault("RUST_BACKTRACE", "1")
    cmd = _pg_eval_command() + forwarded
    print("Running eval command:", " ".join(cmd), flush=True)
    print(
        "Eval environment:",
        {
            "PG_EVAL_GPU_WORLD_SIZE": os.environ.get("PG_EVAL_GPU_WORLD_SIZE"),
            "PG_TTT_AUDIT": os.environ.get("PG_TTT_AUDIT"),
            "PG_TTT_ASSERT_SCORE_NO_MUTATION": os.environ.get("PG_TTT_ASSERT_SCORE_NO_MUTATION"),
            "PG_GPU_BF16_BACKWARD_CHAIN": os.environ.get("PG_GPU_BF16_BACKWARD_CHAIN"),
        },
        flush=True,
    )

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
    result["json_events"] = _parse_json_events(result["tail"])
    result["metrics"] = _merge_json_event_metrics(
        _parse_key_value_metrics(result["tail"]), result["json_events"]
    )
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
    result["json_events"] = _parse_json_events(result["tail"])
    result["metrics"] = _merge_json_event_metrics(
        _parse_key_value_metrics(result["tail"]), result["json_events"]
    )
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
    if "--world-size" in forwarded:
        idx = forwarded.index("--world-size")
        if idx + 1 < len(forwarded):
            try:
                return int(forwarded[idx + 1]) > 1
            except ValueError:
                return False
    if "--backend" in forwarded:
        idx = forwarded.index("--backend")
        if idx + 1 < len(forwarded) and forwarded[idx + 1] == "cuda-distributed":
            return True
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
        "/build/target": build_cache_volume,
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
        "/build/target": build_cache_volume,
    },
)
def run_command_multi(args: list[str]):
    return _run_pg_train(args, "multi-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
        "/build/target": build_cache_volume,
    },
)
def run_command_multi_string(args: str):
    """CLI-friendly multi-GPU entrypoint for detached validation jobs."""

    return _run_pg_train(shlex.split(args), "multi-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
        "/build/target": build_cache_volume,
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
        "/build/target": build_cache_volume,
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
