"""
SOTA Auto-Research: Qwen-Guided Edge Finding
==============================================
Takes the current SOTA config (1.1233 BPB) and uses Qwen to find
improvements testable on DGX Spark. Tests relative improvements
that transfer to H100.

Usage:
  source .venv/bin/activate
  nohup python autoresearch_sota.py > autoresearch_sota.log 2>&1 &
  tail -f autoresearch_sota.log
"""

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

SCRIPT = "train_local.py"
RESULTS_FILE = "autoresearch_sota_results.csv"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")

FIELDS = [
    "timestamp", "run_id", "val_bpb",
    "num_layers", "model_dim", "num_heads", "num_kv_heads",
    "mlp_mult", "lr", "seq_len",
    "steps", "avg_ms", "time_s", "params",
    "reasoning", "notes"
]

RUN_DEFAULTS = {
    "iterations": 300,
    "eval_tokens": 100000,
    "max_seconds": 300,
    "batch_tokens": 32768,
    "seq_len": 1024,
    "seed": 1337,
}

SYSTEM_PROMPT = """You are an ML research assistant optimizing a small transformer language model for a competition.

GOAL: Minimize val_bpb (bits per byte). Current world record is 1.1233 BPB on 8xH100 (10 min training).
We are testing on a DGX Spark (1 GPU, 300 steps) to find RELATIVE improvements that transfer to H100.

CURRENT SOTA CONFIG (on H100):
- 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion with relu-squared activation
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections
- SmearGate + BigramHash (2048 buckets, dim=128)
- Shared Value Embedding (dim=128)
- XSA (cross-self attention) on last 4 layers
- Logit softcap 30.0, tied embeddings
- Muon optimizer (matrices), AdamW (embeddings/scalars)
- EMA + self-distillation (50 steps, temp=2.0, alpha=0.7)
- Int6 per-row quantization + zstd compression

LOCAL TEST SETUP (DGX Spark):
- 1 GPU (GB10), uses PyTorch SDPA (no FlashAttention 3)
- AdamW optimizer (no Muon — local simplification)
- 300 steps, ~2-3 minutes per run
- Same architecture, data, tokenizer, BPB metric

WHAT WE CAN TEST LOCALLY:
- Number of layers (7-15)
- Model dimension (384-640, must be divisible by 2*num_heads)
- Number of attention heads (4, 8, 12, 16)
- Number of KV heads (must divide num_heads)
- MLP expansion multiplier (2, 3, 4)
- Learning rate (1e-4 to 2e-3)
- Sequence length (512, 1024, 2048)

WHAT TRANSFERS TO H100:
- Architecture shape improvements (layers, dim, heads) transfer well
- Relative ranking of configs transfers (if A > B locally, usually A > B on H100)
- Absolute BPB values do NOT transfer (H100 gets ~7000 steps vs our 300)
- LR optimal values don't transfer (different optimizer)

STRATEGY: Find architecture improvements. The SOTA uses 11L/512d which was hand-tuned.
Maybe 12L/480d, or 10L/544d, or different head configs perform better.
Also test MLP multiplier — 3x is default but 2x or 4x might be better.

Respond with ONLY a JSON object (no markdown, no code fences):
{
  "reasoning": "Brief explanation of why this config (2-3 sentences)",
  "config": {
    "num_layers": <int>,
    "model_dim": <int>,
    "num_heads": <int>,
    "num_kv_heads": <int>,
    "mlp_mult": <int>,
    "lr": <float>,
    "seq_len": <int>
  }
}"""

# ─── OLLAMA ───────────────────────────────────────────────────────────────────

def ask_qwen(history_text, last_result_text):
    prompt = f"""Here are ALL experiment results so far (sorted by val_bpb, best first):

{history_text}

The most recent experiment result:
{last_result_text}

Based on the patterns, propose the NEXT config. Look for:
1. Which architecture changes improve BPB most
2. Promising regions to explore deeper
3. Whether to exploit (refine near best) or explore (try something different)

Do NOT repeat a configuration already tested."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512}
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"  Qwen error: {e}")
        return None


def parse_response(text):
    if not text:
        return None, "no response"
    clean = text.strip()
    if "```" in clean:
        for p in clean.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                clean = p
                break
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None, f"no JSON: {text[:100]}"
    try:
        obj = json.loads(clean[start:end])
        reasoning = obj.get("reasoning", "")
        cfg = obj.get("config", obj)
        v = {}
        if "num_layers" in cfg:
            v["num_layers"] = max(4, min(20, int(cfg["num_layers"])))
        if "model_dim" in cfg:
            v["model_dim"] = max(256, min(1024, int(cfg["model_dim"])))
        if "num_heads" in cfg:
            v["num_heads"] = max(2, min(16, int(cfg["num_heads"])))
        if "num_kv_heads" in cfg:
            v["num_kv_heads"] = max(1, min(v.get("num_heads", 8), int(cfg["num_kv_heads"])))
        if "mlp_mult" in cfg:
            v["mlp_mult"] = max(1, min(6, int(cfg["mlp_mult"])))
        if "lr" in cfg:
            v["lr"] = max(1e-5, min(0.01, float(cfg["lr"])))
        if "seq_len" in cfg:
            v["seq_len"] = int(cfg["seq_len"])
            if v["seq_len"] not in [512, 1024, 2048]:
                v["seq_len"] = 1024
        # Fix dim divisibility
        if "model_dim" in v and "num_heads" in v:
            step = 2 * v["num_heads"]
            v["model_dim"] = (v["model_dim"] // step) * step
            if v["model_dim"] < 256:
                v["model_dim"] = 256
        return v, reasoning
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, f"parse error: {e}"


# ─── RUNNER ───────────────────────────────────────────────────────────────────

def run_experiment(config, run_id):
    cfg = {**RUN_DEFAULTS, **config}
    cfg.setdefault("num_layers", 9)
    cfg.setdefault("model_dim", 512)
    cfg.setdefault("num_heads", 8)
    cfg.setdefault("num_kv_heads", 4)
    cfg.setdefault("mlp_mult", 2)
    cfg.setdefault("lr", 3e-4)

    # Use baseline mode with configurable layers/dim
    cmd = [
        sys.executable, SCRIPT,
        "--mode", "baseline",
        "--model-dim", str(cfg["model_dim"]),
        "--num-heads", str(cfg["num_heads"]),
        "--num-kv-heads", str(cfg["num_kv_heads"]),
        "--mlp-mult", str(cfg["mlp_mult"]),
        "--lr", str(cfg["lr"]),
        "--seq-len", str(cfg["seq_len"]),
        "--iterations", str(cfg["iterations"]),
        "--eval-tokens", str(cfg["eval_tokens"]),
        "--max-seconds", str(cfg["max_seconds"]),
        "--batch-tokens", str(cfg["batch_tokens"]),
        "--seed", str(cfg["seed"]),
        "--run-id", run_id,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        stderr = result.stderr
        if stderr:
            print(f"  {stderr[-300:]}")
        return None

    parsed = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "num_layers": cfg.get("num_layers", 9),
        "model_dim": cfg["model_dim"],
        "num_heads": cfg["num_heads"],
        "num_kv_heads": cfg["num_kv_heads"],
        "mlp_mult": cfg["mlp_mult"],
        "lr": cfg["lr"],
        "seq_len": cfg["seq_len"],
    }
    stdout = result.stdout
    for line in stdout.split("\n"):
        if "val_bpb:" in line and "val_bpb:enabled" not in line:
            try:
                # Handle both "val_bpb:1.234" and "val_bpb:  1.234" formats
                bpb_str = line.split("val_bpb:")[1].strip().split()[0]
                parsed["val_bpb"] = float(bpb_str)
            except (ValueError, IndexError):
                pass
        if line.startswith("params:"):
            try:
                parsed["params"] = line.split("params:")[1].strip().split()[0].replace(",", "")
            except (ValueError, IndexError):
                pass
        if "step_avg:" in line:
            try:
                parsed["avg_ms"] = float(line.split("step_avg:")[1].strip().split()[0].rstrip("ms"))
            except (ValueError, IndexError):
                pass
        if line.startswith("time:"):
            try:
                parsed["time_s"] = float(line.split("time:")[1].strip().split()[0].rstrip("ms")) / 1000
            except (ValueError, IndexError):
                pass
        if line.startswith("steps:"):
            try:
                parsed["steps"] = int(line.split()[0].split(":")[1])
            except (ValueError, IndexError):
                pass

    return parsed


def format_history(results):
    if not results:
        return "No experiments yet."
    valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
    valid.sort(key=lambda r: float(r["val_bpb"]))
    lines = []
    for r in valid[:30]:
        lines.append(
            f"bpb={float(r['val_bpb']):.4f} | "
            f"L={r.get('num_layers','?')} dim={r.get('model_dim','?')} "
            f"heads={r.get('num_heads','?')}/{r.get('num_kv_heads','?')} "
            f"mlp={r.get('mlp_mult','?')} lr={float(r.get('lr',0)):.1e} "
            f"seq={r.get('seq_len','?')} | {r.get('notes','')}"
        )
    return "\n".join(lines)


def format_last(result):
    if not result:
        return "First run."
    return (
        f"bpb={result.get('val_bpb','?')} | L={result.get('num_layers','?')} "
        f"dim={result.get('model_dim','?')} heads={result.get('num_heads','?')} "
        f"mlp={result.get('mlp_mult','?')} lr={result.get('lr','?')}"
    )


def load_results():
    results = []
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                results.append(row)
    return results


def save_result(result):
    exists = Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(result)


def fallback_config():
    return {
        "num_layers": random.choice([9, 10, 11, 12, 13]),
        "model_dim": random.choice([384, 416, 448, 480, 512, 544, 576]),
        "num_heads": random.choice([4, 8]),
        "num_kv_heads": random.choice([2, 4]),
        "mlp_mult": random.choice([2, 3]),
        "lr": random.choice([1e-4, 2e-4, 3e-4, 5e-4, 8e-4]),
        "seq_len": 1024,
    }


# ─── SEED CONFIGS ─────────────────────────────────────────────────────────────

SEEDS = [
    # SOTA reference
    {"num_layers": 9, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 3e-4, "notes": "seed: baseline 9L/512d (reference)"},
    # SOTA-like with 11 layers (matches H100 config)
    {"num_layers": 11, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 3e-4, "notes": "seed: 11L/512d (H100 match)"},
    # More layers, narrower
    {"num_layers": 13, "model_dim": 448, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 3e-4, "notes": "seed: 13L/448d deeper"},
    # Fewer layers, wider
    {"num_layers": 8, "model_dim": 576, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 3e-4, "notes": "seed: 8L/576d wider"},
    # MLP 3x (matches H100)
    {"num_layers": 9, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 3, "lr": 3e-4, "notes": "seed: 9L/512d mlp3x"},
    # Higher LR
    {"num_layers": 9, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 8e-4, "notes": "seed: baseline high lr"},
    # More heads
    {"num_layers": 9, "model_dim": 512, "num_heads": 16, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 3e-4, "notes": "seed: 16 heads"},
    # 12 layers (sweet spot?)
    {"num_layers": 12, "model_dim": 480, "num_heads": 8, "num_kv_heads": 4,
     "mlp_mult": 2, "lr": 3e-4, "notes": "seed: 12L/480d"},
]

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SOTA AUTO-RESEARCH — Qwen-Guided Edge Finding")
    print(f"Model: {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 70)

    results = load_results()
    run_count = len(results)
    last_result = None

    # Seed runs
    if run_count < len(SEEDS):
        print(f"\n>>> SEED PHASE: {len(SEEDS)} configs")
        for i, cfg in enumerate(SEEDS):
            if i < run_count:
                continue
            run_count += 1
            rid = f"sota_{run_count:03d}"
            notes = cfg.pop("notes", "")
            print(f"\n[seed {run_count}] {notes}")
            print(f"  L={cfg.get('num_layers',9)} dim={cfg.get('model_dim',512)} "
                  f"heads={cfg.get('num_heads',8)}/{cfg.get('num_kv_heads',4)} "
                  f"mlp={cfg.get('mlp_mult',2)} lr={cfg.get('lr',3e-4):.1e}")
            r = run_experiment(cfg, rid)
            if r:
                r["notes"] = notes
                r["reasoning"] = "seed"
                save_result(r)
                results.append(r)
                last_result = r
                print(f"  >>> val_bpb={r.get('val_bpb', '?')}")

    # Qwen-guided loop
    qwen_fails = 0
    while True:
        run_count += 1
        best = min((float(r.get("val_bpb", 999)) for r in results if r.get("val_bpb")), default=999)
        print(f"\n{'='*70}")
        print(f"RUN {run_count} | {datetime.now().strftime('%H:%M:%S')} | best={best:.4f}")
        print(f"{'='*70}")

        print("  Asking Qwen...")
        response = ask_qwen(format_history(results), format_last(last_result))

        config = None
        reasoning = ""
        if response:
            config, reasoning = parse_response(response)
            if config:
                print(f"  Qwen: {reasoning[:120]}")
                qwen_fails = 0
            else:
                print(f"  Parse fail: {reasoning[:100]}")
                qwen_fails += 1
        else:
            qwen_fails += 1

        if config is None:
            config = fallback_config()
            reasoning = f"fallback (fails:{qwen_fails})"

        # Ensure dim divisibility
        nh = config.get("num_heads", 8)
        step = 2 * nh
        dim = config.get("model_dim", 512)
        config["model_dim"] = max(step, (dim // step) * step)

        print(f"  Config: L={config.get('num_layers','?')} dim={config['model_dim']} "
              f"heads={config.get('num_heads','?')}/{config.get('num_kv_heads','?')} "
              f"mlp={config.get('mlp_mult','?')} lr={config.get('lr',3e-4):.1e}")

        r = run_experiment(config, f"sota_{run_count:03d}")
        if r:
            r["reasoning"] = reasoning[:200]
            r["notes"] = reasoning[:100]
            save_result(r)
            results.append(r)
            last_result = r
            bpb = r.get("val_bpb", "?")
            print(f"  >>> val_bpb={bpb}")
        else:
            last_result = None

        if run_count % 5 == 0:
            valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
            valid.sort(key=lambda r: float(r["val_bpb"]))
            print(f"\n{'='*80}")
            print(f"LEADERBOARD (top 10 of {len(valid)})")
            print(f"{'='*80}")
            for i, r in enumerate(valid[:10]):
                print(f"  {i+1:>2}. bpb={float(r['val_bpb']):>7.4f} | "
                      f"L={r.get('num_layers','?')} dim={r.get('model_dim','?')} "
                      f"h={r.get('num_heads','?')}/{r.get('num_kv_heads','?')} "
                      f"mlp={r.get('mlp_mult','?')} lr={float(r.get('lr',0)):.1e}")

if __name__ == "__main__":
    main()
