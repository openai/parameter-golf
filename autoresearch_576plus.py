"""
Karpathy-style autoresearch loop for train_gpt_576plus.py.

Key properties:
- Creates a NEW shell script per test under scripts/edge_autoresearch/
- Runs each test script and waits for completion
- Reads structured results from results/autoruns/<RUN_ID>/result_summary.json
- Appends a compact ledger to autoresearch_576plus_results.csv
- Uses local Ollama model (Qwen) for micro-adjustments
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

TRAIN_SCRIPT = "train_gpt_576plus.py"
RESULTS_FILE = Path("autoresearch_576plus_results.csv")
RUN_SCRIPTS_DIR = Path("scripts/edge_autoresearch")
AUTORUN_ROOT = Path("results/autoruns")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")
# DGX Spark / local hosts are often single-GPU; explicit env can still override.
NPROC = int(os.environ.get("NPROC", "1"))
SEED = int(os.environ.get("SEED", "1337"))
EDGE_TARGET_BPB = float(os.environ.get("EDGE_TARGET_BPB", "1.1220"))
RUN_TIMEOUT_SECONDS = int(os.environ.get("RUN_TIMEOUT_SECONDS", "2400"))

FIELDS = [
    "timestamp",
    "run_id",
    "status",
    "primary_bpb",
    "final_intq_sliding_window_bpb",
    "final_intq_roundtrip_bpb",
    "legal_ttt_bpb",
    "post_ttt_temp_bpb",
    "quant_artifact_bytes",
    "model_params",
    "reasoning",
    "notes",
    "config_json",
    "script_path",
    "summary_path",
]

RUN_DEFAULTS: dict[str, Any] = {
    "num_layers": 11,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 8,
    "mlp_mult": 3.5,
    "bigram_vocab_size": 8192,
    "bigram_dim": 128,
    "xsa_last_n": 11,
    "rope_dims": 16,
    "train_seq_len": 2048,
    "eval_seq_len": 2048,
    "train_batch_tokens": 786432,
    "max_wallclock_seconds": 600,
    "warmdown_iters": 3500,
    "val_loss_every": 0,
    "eval_stride": 64,
    "matrix_lr": 0.025,
    "scalar_lr": 0.025,
    "tied_embed_lr": 0.035,
    "muon_momentum": 0.99,
    "muon_wd": 0.04,
    "qat_enabled": 0,
    "late_qat_threshold": 0.50,
    "quant_int_categories": "mlp,attn",
    "quant_mlp_clip_range": 15,
    "quant_attn_clip_range": 15,
    "quant_embed_clip_range": 31,
    "quant_other_clip_range": 31,
    "gptq_block_size": 64,
    "gptq_percdamp": 0.01,
    "gptq_calibration_samples": 256,
    "quant_artifact_name": "final_model.intq.ptz",
    # Default to no TTT because latest evidence showed degradation in this lane.
    "ttt_eval_enabled": 0,
    "ttt_optimizer": "adamw",
    "ttt_lr": 1e-4,
    "ttt_epochs": 3,
    "ttt_chunk_tokens": 131072,
    "ttt_freeze_blocks": 9,
    "ttt_freeze_embed": 1,
    "ttt_grad_clip": 1.0,
    "ttt_max_train_chunks": 200,
    "ttt_ema_decay": 0.995,
    "post_ttt_temp_enabled": 0,
    "post_ttt_temperature": 0.98,
}

SYSTEM_PROMPT = """You are optimizing a competitive 8xGPU training/eval script for best val_bpb.

Primary objective:
- Minimize final_intq_sliding_window_bpb.

Important context from recent runs:
- Enabling TTT in this lane often worsened metric; default is TTT OFF.
- Pure int5 on both MLP/attn was strong on size but quality-sensitive.
- Mixed quant (MLP int5, attn int6) may recover quality.

You can propose only these knobs:
- quant_attn_clip_range: 15 or 31
- quant_mlp_clip_range: 15
- gptq_block_size: 64 or 128
- gptq_percdamp: 0.002, 0.01, 0.03
- bigram_vocab_size: 6144, 8192, 10240
- xsa_last_n: 8 or 11
- muon_wd: 0.03, 0.04, 0.05
- tied_embed_lr: 0.030, 0.035, 0.040
- ttt_eval_enabled: 0 or 1
- if ttt_eval_enabled=1 also set:
  - ttt_lr: 0.0001 or 0.0002
  - ttt_freeze_blocks: 8 or 9
  - post_ttt_temp_enabled: 0 or 1
  - post_ttt_temperature: 0.98 or 0.99

Rules:
- Do not repeat previously tested configs.
- Keep everything else fixed.
- Prefer changes with clear expected upside, not random churn.

Return ONLY JSON:
{
  "reasoning": "short rationale",
  "config": { ...knobs... }
}
"""

SEEDS: list[dict[str, Any]] = [
    {
        "quant_attn_clip_range": 15,
        "quant_mlp_clip_range": 15,
        "gptq_block_size": 64,
        "gptq_percdamp": 0.01,
        "ttt_eval_enabled": 0,
        "post_ttt_temp_enabled": 0,
        "notes": "seed: pure int5 no TTT",
    },
    {
        "quant_attn_clip_range": 31,
        "quant_mlp_clip_range": 15,
        "gptq_block_size": 64,
        "gptq_percdamp": 0.01,
        "ttt_eval_enabled": 0,
        "post_ttt_temp_enabled": 0,
        "notes": "seed: mixed mlp5/attn6 no TTT",
    },
    {
        "quant_attn_clip_range": 31,
        "quant_mlp_clip_range": 15,
        "gptq_block_size": 128,
        "gptq_percdamp": 0.002,
        "ttt_eval_enabled": 0,
        "post_ttt_temp_enabled": 0,
        "notes": "seed: mixed mlp5/attn6 b128 pd002",
    },
    {
        "quant_attn_clip_range": 15,
        "quant_mlp_clip_range": 15,
        "gptq_block_size": 64,
        "gptq_percdamp": 0.01,
        "ttt_eval_enabled": 1,
        "ttt_lr": 1e-4,
        "ttt_freeze_blocks": 9,
        "post_ttt_temp_enabled": 0,
        "notes": "seed: pure int5 with conservative TTT",
    },
]


def canonicalize_config(overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = {**RUN_DEFAULTS, **overrides}

    cfg["quant_attn_clip_range"] = int(cfg["quant_attn_clip_range"])
    cfg["quant_mlp_clip_range"] = int(cfg["quant_mlp_clip_range"])
    cfg["gptq_block_size"] = int(cfg["gptq_block_size"])
    cfg["gptq_percdamp"] = float(cfg["gptq_percdamp"])
    cfg["bigram_vocab_size"] = int(cfg["bigram_vocab_size"])
    cfg["xsa_last_n"] = int(cfg["xsa_last_n"])
    cfg["muon_wd"] = float(cfg["muon_wd"])
    cfg["tied_embed_lr"] = float(cfg["tied_embed_lr"])

    cfg["ttt_eval_enabled"] = int(bool(int(cfg["ttt_eval_enabled"])))
    cfg["post_ttt_temp_enabled"] = int(bool(int(cfg["post_ttt_temp_enabled"])))

    if cfg["ttt_eval_enabled"] == 0:
        cfg["post_ttt_temp_enabled"] = 0

    cfg["ttt_lr"] = float(cfg["ttt_lr"])
    cfg["ttt_freeze_blocks"] = int(cfg["ttt_freeze_blocks"])
    cfg["post_ttt_temperature"] = float(cfg["post_ttt_temperature"])

    if cfg["quant_attn_clip_range"] not in (15, 31):
        cfg["quant_attn_clip_range"] = 15
    if cfg["quant_mlp_clip_range"] != 15:
        cfg["quant_mlp_clip_range"] = 15
    if cfg["gptq_block_size"] not in (64, 128):
        cfg["gptq_block_size"] = 64
    if cfg["bigram_vocab_size"] not in (6144, 8192, 10240):
        cfg["bigram_vocab_size"] = 8192
    if cfg["xsa_last_n"] not in (8, 11):
        cfg["xsa_last_n"] = 11
    if cfg["ttt_lr"] not in (1e-4, 2e-4):
        cfg["ttt_lr"] = 1e-4
    if cfg["ttt_freeze_blocks"] not in (8, 9):
        cfg["ttt_freeze_blocks"] = 9
    if cfg["post_ttt_temperature"] not in (0.98, 0.99):
        cfg["post_ttt_temperature"] = 0.98

    return cfg


def config_key(cfg: dict[str, Any]) -> str:
    key_fields = [
        "quant_attn_clip_range",
        "quant_mlp_clip_range",
        "gptq_block_size",
        "gptq_percdamp",
        "bigram_vocab_size",
        "xsa_last_n",
        "muon_wd",
        "tied_embed_lr",
        "ttt_eval_enabled",
        "ttt_lr",
        "ttt_freeze_blocks",
        "post_ttt_temp_enabled",
        "post_ttt_temperature",
    ]
    compact = {k: cfg[k] for k in key_fields}
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def write_run_script(run_id: str, cfg: dict[str, Any]) -> Path:
    RUN_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = RUN_SCRIPTS_DIR / f"run_{run_id}.sh"
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
        'cd "$REPO_DIR"',
        'export PYTHONPATH="$REPO_DIR/flash-attention/hopper:${PYTHONPATH:-}"',
        f'NPROC="${{NPROC:-{NPROC}}}"',
        f'SEED="${{SEED:-{SEED}}}"',
        f'RUN_ID="{run_id}"',
        'echo "RUN_ID=$RUN_ID"',
        "env \\",
        "  RUN_ID=\"$RUN_ID\" SEED=\"$SEED\" \\",
        f"  NUM_LAYERS={cfg['num_layers']} MODEL_DIM={cfg['model_dim']} NUM_HEADS={cfg['num_heads']} NUM_KV_HEADS={cfg['num_kv_heads']} MLP_MULT={cfg['mlp_mult']} \\",
        f"  BIGRAM_VOCAB_SIZE={cfg['bigram_vocab_size']} BIGRAM_DIM={cfg['bigram_dim']} XSA_LAST_N={cfg['xsa_last_n']} ROPE_DIMS={cfg['rope_dims']} \\",
        f"  TRAIN_SEQ_LEN={cfg['train_seq_len']} EVAL_SEQ_LEN={cfg['eval_seq_len']} TRAIN_BATCH_TOKENS={cfg['train_batch_tokens']} \\",
        f"  MAX_WALLCLOCK_SECONDS={cfg['max_wallclock_seconds']} WARMDOWN_ITERS={cfg['warmdown_iters']} VAL_LOSS_EVERY={cfg['val_loss_every']} EVAL_STRIDE={cfg['eval_stride']} \\",
        f"  MATRIX_LR={cfg['matrix_lr']} SCALAR_LR={cfg['scalar_lr']} TIED_EMBED_LR={cfg['tied_embed_lr']} MUON_MOMENTUM={cfg['muon_momentum']} MUON_WD={cfg['muon_wd']} \\",
        f"  QAT_ENABLED={cfg['qat_enabled']} LATE_QAT_THRESHOLD={cfg['late_qat_threshold']} \\",
        f"  QUANT_INT_CATEGORIES={cfg['quant_int_categories']} QUANT_MLP_CLIP_RANGE={cfg['quant_mlp_clip_range']} QUANT_ATTN_CLIP_RANGE={cfg['quant_attn_clip_range']} \\",
        f"  QUANT_EMBED_CLIP_RANGE={cfg['quant_embed_clip_range']} QUANT_OTHER_CLIP_RANGE={cfg['quant_other_clip_range']} \\",
        f"  GPTQ_BLOCK_SIZE={cfg['gptq_block_size']} GPTQ_PERCDAMP={cfg['gptq_percdamp']} GPTQ_CALIBRATION_SAMPLES={cfg['gptq_calibration_samples']} \\",
        "  QUANT_ARTIFACT_NAME=final_model.intq.ptz \\",
        f"  TTT_EVAL_ENABLED={cfg['ttt_eval_enabled']} TTT_OPTIMIZER={cfg['ttt_optimizer']} TTT_LR={cfg['ttt_lr']} TTT_EPOCHS={cfg['ttt_epochs']} \\",
        f"  TTT_CHUNK_TOKENS={cfg['ttt_chunk_tokens']} TTT_FREEZE_BLOCKS={cfg['ttt_freeze_blocks']} TTT_FREEZE_EMBED={cfg['ttt_freeze_embed']} \\",
        f"  TTT_GRAD_CLIP={cfg['ttt_grad_clip']} TTT_MAX_TRAIN_CHUNKS={cfg['ttt_max_train_chunks']} TTT_EMA_DECAY={cfg['ttt_ema_decay']} \\",
        f"  POST_TTT_TEMP_ENABLED={cfg['post_ttt_temp_enabled']} POST_TTT_TEMPERATURE={cfg['post_ttt_temperature']} \\",
        f"  torchrun --standalone --nproc_per_node=\"$NPROC\" {TRAIN_SCRIPT}",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def ask_qwen(history_text: str, last_result_text: str) -> str | None:
    prompt = f"""History (best first):
{history_text}

Most recent:
{last_result_text}

Propose ONE next config that is not a duplicate."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 384},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001
        print(f"  qwen_error: {e}")
        return None


def parse_qwen_response(text: str | None) -> tuple[dict[str, Any] | None, str]:
    if not text:
        return None, "no response"
    clean = text.strip()
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break
    s = clean.find("{")
    e = clean.rfind("}") + 1
    if s < 0 or e <= s:
        return None, f"no json: {clean[:120]}"
    try:
        obj = json.loads(clean[s:e])
        cfg = obj.get("config", obj)
        reasoning = str(obj.get("reasoning", ""))
        return cfg, reasoning
    except Exception as ex:  # noqa: BLE001
        return None, f"parse_error: {ex}"


def fallback_config() -> dict[str, Any]:
    return {
        "quant_attn_clip_range": random.choice([15, 31]),
        "quant_mlp_clip_range": 15,
        "gptq_block_size": random.choice([64, 128]),
        "gptq_percdamp": random.choice([0.002, 0.01, 0.03]),
        "bigram_vocab_size": random.choice([6144, 8192, 10240]),
        "xsa_last_n": random.choice([8, 11]),
        "muon_wd": random.choice([0.03, 0.04, 0.05]),
        "tied_embed_lr": random.choice([0.03, 0.035, 0.04]),
        "ttt_eval_enabled": random.choice([0, 1]),
        "ttt_lr": random.choice([1e-4, 2e-4]),
        "ttt_freeze_blocks": random.choice([8, 9]),
        "post_ttt_temp_enabled": random.choice([0, 1]),
        "post_ttt_temperature": random.choice([0.98, 0.99]),
    }


def load_results() -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_result(row: dict[str, Any]) -> None:
    exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_history(rows: list[dict[str, str]], n: int = 25) -> str:
    valid = []
    for r in rows:
        try:
            bpb = float(r.get("primary_bpb", "nan"))
            if bpb == bpb:
                valid.append((bpb, r))
        except Exception:  # noqa: BLE001
            continue
    valid.sort(key=lambda x: x[0])
    lines = []
    for bpb, r in valid[:n]:
        lines.append(
            f"bpb={bpb:.6f} "
            f"attn_clip={r.get('config_json','').find('\"quant_attn_clip_range\":31')!=-1 and 31 or 15} "
            f"notes={r.get('notes','')[:80]}"
        )
    return "\n".join(lines) if lines else "no runs yet"


def format_last(r: dict[str, Any] | None) -> str:
    if not r:
        return "none"
    return (
        f"run_id={r.get('run_id')} status={r.get('status')} primary_bpb={r.get('primary_bpb')} "
        f"notes={r.get('notes','')}"
    )


def run_experiment(run_id: str, cfg: dict[str, Any], reasoning: str, notes: str) -> dict[str, Any]:
    script_path = write_run_script(run_id, cfg)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
        status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
        tail = (proc.stdout or "")[-1200:]
        if proc.returncode != 0 and proc.stderr:
            tail += "\nSTDERR:\n" + proc.stderr[-1200:]
    except subprocess.TimeoutExpired:
        status = "timeout"
        tail = "run timed out"

    summary_path = AUTORUN_ROOT / run_id / "result_summary.json"
    primary_bpb = ""
    final_intq_sw_bpb = ""
    final_intq_rt_bpb = ""
    legal_ttt_bpb = ""
    post_ttt_bpb = ""
    quant_bytes = ""
    model_params = ""
    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        m = obj.get("metrics", {})
        final_intq_sw_bpb = m.get("final_intq_sliding_window", {}).get("val_bpb", "")
        final_intq_rt_bpb = m.get("final_intq_roundtrip", {}).get("val_bpb", "")
        legal_ttt_bpb = m.get("legal_ttt", {}).get("val_bpb", "")
        post_ttt_bpb = m.get("post_ttt_temp_rescore", {}).get("val_bpb", "")
        primary_bpb = final_intq_sw_bpb if final_intq_sw_bpb != "" else final_intq_rt_bpb
        quant_bytes = obj.get("quant_artifact_bytes", "")
        model_params = obj.get("model_params", "")
    else:
        notes = (notes + " | missing_summary").strip(" |")

    return {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "status": status,
        "primary_bpb": primary_bpb,
        "final_intq_sliding_window_bpb": final_intq_sw_bpb,
        "final_intq_roundtrip_bpb": final_intq_rt_bpb,
        "legal_ttt_bpb": legal_ttt_bpb,
        "post_ttt_temp_bpb": post_ttt_bpb,
        "quant_artifact_bytes": quant_bytes,
        "model_params": model_params,
        "reasoning": reasoning[:300],
        "notes": (notes + f" | elapsed={time.time() - t0:.1f}s | tail={tail[-260:].replace(chr(10), ' ')}")[:1000],
        "config_json": json.dumps(cfg, sort_keys=True),
        "script_path": str(script_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    print("=" * 80)
    print("AUTORESEARCH 576+ — continuous edge hunting")
    print(f"started: {datetime.now().isoformat()}")
    print(f"ollama: {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print(f"results: {RESULTS_FILE}")
    print(f"target_bpb: <= {EDGE_TARGET_BPB:.6f}")
    print("=" * 80)

    if not Path(TRAIN_SCRIPT).exists():
        raise FileNotFoundError(f"{TRAIN_SCRIPT} not found")

    rows = load_results()
    tested = set()
    for r in rows:
        c = r.get("config_json")
        if c:
            tested.add(c)

    last: dict[str, Any] | None = None
    run_index = len(rows)

    # Seed phase
    for s in SEEDS:
        cfg = canonicalize_config({k: v for k, v in s.items() if k != "notes"})
        key = config_key(cfg)
        if key in tested:
            continue
        run_index += 1
        run_id = f"edge_auto_{run_index:03d}"
        print(f"\n[seed] {run_id} {s.get('notes', '')}")
        row = run_experiment(run_id, cfg, reasoning="seed", notes=str(s.get("notes", "")))
        save_result(row)
        rows.append({k: str(v) for k, v in row.items()})
        tested.add(key)
        last = row
        print(f"  status={row['status']} primary_bpb={row['primary_bpb']}")

    # Main loop
    qwen_failures = 0
    while True:
        # stop if edge found
        best = min(
            (float(r["primary_bpb"]) for r in rows if r.get("primary_bpb") not in ("", "None")),
            default=999.0,
        )
        print(f"\n{'-'*80}\nnext_run best={best:.6f} qwen_failures={qwen_failures}")
        if best <= EDGE_TARGET_BPB:
            print(f"edge_found best={best:.6f} <= target={EDGE_TARGET_BPB:.6f}")
            break

        response = ask_qwen(format_history(rows), format_last(last))
        cfg_raw, reasoning = parse_qwen_response(response)
        if cfg_raw is None:
            qwen_failures += 1
            cfg_raw = fallback_config()
            reasoning = f"fallback_after_qwen_failure:{qwen_failures}"
        else:
            qwen_failures = 0

        # enforce novel config, retry fallback a few times
        attempts = 0
        cfg = canonicalize_config(cfg_raw)
        key = config_key(cfg)
        while key in tested and attempts < 12:
            cfg = canonicalize_config(fallback_config())
            key = config_key(cfg)
            attempts += 1
        if key in tested:
            print("no novel config found; sleeping 30s")
            time.sleep(30)
            continue

        run_index += 1
        run_id = f"edge_auto_{run_index:03d}"
        print(f"run_id={run_id} cfg={key}")
        row = run_experiment(run_id, cfg, reasoning=reasoning, notes=f"novel_attempts={attempts}")
        save_result(row)
        rows.append({k: str(v) for k, v in row.items()})
        tested.add(key)
        last = row
        print(f"  status={row['status']} primary_bpb={row['primary_bpb']}")

        if run_index % 5 == 0:
            leaderboard = sorted(
                (
                    (float(r["primary_bpb"]), r["run_id"])
                    for r in rows
                    if r.get("primary_bpb") not in ("", "None")
                ),
                key=lambda x: x[0],
            )
            print("top5:")
            for i, (bpb, rid) in enumerate(leaderboard[:5], start=1):
                print(f"  {i}. {bpb:.6f}  {rid}")


if __name__ == "__main__":
    main()
