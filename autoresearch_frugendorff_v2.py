"""
Frugendorff V2 Auto-Research: Applying SOTA Techniques
========================================================
Takes the Frugendorff Squared (1.1478 BPB) and systematically tests
techniques from the SOTA channel to close the gap to 1.1233.

Key techniques to test (from v7/session state):
1. Post-Quant Burst (PQB) — repair quant damage with STE fine-tuning
2. Freeze-block TTT — freeze early blocks, only adapt deep blocks
3. TTT early stopping — peak at chunk ~50-60 then stop
4. Batch size reduction — more steps in 600s (524K vs 786K)
5. Quantization improvements — different int6 clipping, per-row tuning

Runs on DGX Spark. Qwen guides the search.

Usage:
  source .venv/bin/activate
  nohup python autoresearch_frugendorff_v2.py > autoresearch_frug2.log 2>&1 &
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

SCRIPT = "train_fractal_cadence.py"
RESULTS_FILE = "autoresearch_frug2_results.csv"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")

FIELDS = [
    "timestamp", "run_id", "val_bpb",
    "cadence", "cadence_offset", "num_unique_layers", "num_loops",
    "lr", "grad_clip", "mlp_mult", "model_dim",
    "steps", "f_steps", "n_steps", "avg_ms", "time_s", "params",
    "reasoning", "notes"
]

RUN_DEFAULTS = {
    "iterations": 500,
    "eval_tokens": 100000,
    "max_seconds": 600,
    "batch_tokens": 32768,
    "seq_len": 1024,
    "seed": 1337,
}

SYSTEM_PROMPT = """You are optimizing the "Frugendorff" — a recursive weight-shared transformer that just hit 1.1478 BPB on H100.

CURRENT BEST: 6 unique blocks x 2 loops = 12 effective depth, dim=640, 10H/5KV, MLP 4x, cadence 3 (F/N/N).
TARGET: Close the gap to 1.1233 (SOTA with conventional 11-layer architecture).
GAP: 0.025 BPB. Main bottleneck is quantization gap (0.015 vs SOTA's 0.008).

CRITICAL FINDINGS FROM OTHER RESEARCH CHANNEL:
1. TTT peaks at chunk ~50-60 (running BPB 1.1119!) then degrades — early stopping is key
2. Post-Quant Burst (PQB): fine-tune dequantized model with STE to repair quant damage
3. Freezing early blocks + embeddings during TTT prevents catastrophic drift
4. SGD with momentum 0.9 works better than Adam for TTT
5. Smaller batch size (524K vs 786K) = more steps in 600s
6. Higher LR doesn't transfer from AdamW to Muon (tested, hurt)
7. The Frugendorff has 4x leverage from weight sharing during TTT — each update improves all loops

WHAT WE CAN TEST ON SPARK (relative improvements transfer):
- Architecture: layers (4-8), loops (2-4), cadence (1-4), dim (auto), mlp (2-4)
- Training: lr, grad_clip, batch size effects
- The interaction between fractal loop count and training dynamics

WHAT WE NEED TO UNDERSTAND:
1. Does the Frugendorff benefit MORE from MLP 4x than MLP 3x? (confirmed on H100, verify on Spark)
2. Is 6x2 really optimal or could 5x2, 4x3, 8x2 be better WITH MLP 4x?
3. Does the cadence pattern (F/N/N) interact with MLP mult? (maybe MLP 4x needs different cadence)
4. Can we shrink the model slightly to be faster per step and get more total steps?
5. What's the speed/quality pareto frontier for loop count?

Respond with ONLY a JSON object:
{
  "reasoning": "2-3 sentences",
  "config": {
    "num_unique_layers": <int>,
    "num_loops": <int>,
    "cadence": <int>,
    "cadence_offset": <int>,
    "lr": <float>,
    "grad_clip": <float>,
    "mlp_mult": <int>
  }
}"""

def ask_qwen(history_text, last_result_text):
    prompt = f"""Results so far (sorted best first):

{history_text}

Most recent:
{last_result_text}

Propose the next experiment. Focus on finding the optimal Frugendorff config."""

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
        for k, bounds in [("num_unique_layers", (2, 8)), ("num_loops", (1, 4)),
                           ("cadence", (0, 5)), ("cadence_offset", (0, 4)),
                           ("mlp_mult", (2, 4))]:
            if k in cfg:
                v[k] = max(bounds[0], min(bounds[1], int(cfg[k])))
        if "lr" in cfg:
            v["lr"] = max(1e-5, min(0.01, float(cfg["lr"])))
        if "grad_clip" in cfg:
            v["grad_clip"] = max(0.1, min(10.0, float(cfg["grad_clip"])))
        if v.get("cadence", 2) > 0:
            v["cadence_offset"] = min(v.get("cadence_offset", 0), max(v.get("cadence", 2) - 1, 0))
        return v, reasoning
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, f"parse error: {e}"

def run_experiment(config, run_id):
    cfg = {**RUN_DEFAULTS, **config}
    cfg.setdefault("cadence", 3)
    cfg.setdefault("cadence_offset", 0)
    cfg.setdefault("num_unique_layers", 6)
    cfg.setdefault("num_loops", 2)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("mlp_mult", 4)

    cmd = [
        sys.executable, SCRIPT,
        "--cadence", str(cfg["cadence"]),
        "--cadence-offset", str(cfg["cadence_offset"]),
        "--num-unique-layers", str(cfg["num_unique_layers"]),
        "--num-loops", str(cfg["num_loops"]),
        "--lr", str(cfg["lr"]),
        "--grad-clip", str(cfg["grad_clip"]),
        "--mlp-mult", str(cfg["mlp_mult"]),
        "--iterations", str(cfg["iterations"]),
        "--eval-tokens", str(cfg["eval_tokens"]),
        "--max-seconds", str(cfg["max_seconds"]),
        "--batch-tokens", str(cfg["batch_tokens"]),
        "--seq-len", str(cfg["seq_len"]),
        "--seed", str(cfg["seed"]),
        "--run-id", run_id,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    if result.returncode != 0:
        print(f"  FAILED")
        if result.stderr:
            print(f"  {result.stderr[-200:]}")
        return None

    parsed = {
        "timestamp": datetime.now().isoformat(), "run_id": run_id,
        "cadence": cfg["cadence"], "cadence_offset": cfg["cadence_offset"],
        "num_unique_layers": cfg["num_unique_layers"], "num_loops": cfg["num_loops"],
        "lr": cfg["lr"], "grad_clip": cfg["grad_clip"],
        "mlp_mult": cfg["mlp_mult"], "model_dim": cfg.get("model_dim", 0),
    }
    for line in result.stdout.split("\n"):
        if "val_bpb:" in line and "RESULTS" not in line and "val_bpb:enabled" not in line:
            try:
                bpb_str = line.split("val_bpb:")[1].strip().split()[0]
                parsed["val_bpb"] = float(bpb_str)
            except:
                pass
        for tag in ["steps:", "avg_ms:", "time:", "params:"]:
            if line.startswith(tag.rstrip(":")):
                try:
                    parts = line.split()
                    parsed[tag.rstrip(":")] = parts[0].split(":")[1].replace(",","")
                except:
                    pass
    return parsed

def format_history(results):
    if not results:
        return "No experiments yet. Start with the Frugendorff Squared baseline config."
    valid = sorted([r for r in results if r.get("val_bpb") and float(r.get("val_bpb",999)) < 100],
                   key=lambda r: float(r["val_bpb"]))
    return "\n".join(
        f"bpb={float(r['val_bpb']):.4f} | L={r.get('num_unique_layers','?')}x{r.get('num_loops','?')} "
        f"cad={r.get('cadence','?')} lr={float(r.get('lr',0)):.1e} clip={float(r.get('grad_clip',0)):.1f} "
        f"mlp={r.get('mlp_mult','?')}"
        for r in valid[:40]
    )

def load_results():
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE) as f:
        return list(csv.DictReader(f))

def save_result(result):
    exists = Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(result)

SEEDS = [
    # H100 winner config
    {"num_unique_layers": 6, "num_loops": 2, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "H100 winner: 6x2 mlp4"},
    # Vary MLP with winner shape
    {"num_unique_layers": 6, "num_loops": 2, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "6x2 mlp3 (how much does mlp4 help?)"},
    # Fewer layers, more loops (overnight finding)
    {"num_unique_layers": 4, "num_loops": 3, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "4x3 mlp4 (more loops, fewer layers)"},
    # More layers, fewer loops (faster per step)
    {"num_unique_layers": 8, "num_loops": 2, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "8x2 mlp4 (more unique, fast)"},
    # Cadence variations with winner
    {"num_unique_layers": 6, "num_loops": 2, "cadence": 1, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "6x2 always fractal"},
    {"num_unique_layers": 6, "num_loops": 2, "cadence": 2, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "6x2 cadence 2 (F/N)"},
    # Speed test: smaller model, more steps
    {"num_unique_layers": 5, "num_loops": 2, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "5x2 mlp4 (faster, more steps)"},
    # Controls
    {"num_unique_layers": 6, "num_loops": 1, "cadence": 1, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "6x1 no loops (flat mlp4 control)"},
]

def main():
    print("=" * 70)
    print("FRUGENDORFF V2 — Closing the Gap to SOTA")
    print(f"Model: {OLLAMA_MODEL} | Started: {datetime.now().isoformat()}")
    print("=" * 70)

    results = load_results()
    run_count = len(results)
    last_result = None

    if run_count < len(SEEDS):
        print(f"\n>>> SEED PHASE: {len(SEEDS)} configs")
        for i, cfg in enumerate(SEEDS):
            if i < run_count:
                continue
            run_count += 1
            notes = cfg.pop("notes", "")
            print(f"\n[seed {run_count}] {notes}")
            r = run_experiment(cfg, f"frug2_{run_count:03d}")
            if r:
                r["notes"] = notes
                r["reasoning"] = "seed"
                save_result(r)
                results.append(r)
                last_result = r
                print(f"  >>> val_bpb={r.get('val_bpb', '?')}")

    while True:
        run_count += 1
        best = min((float(r.get("val_bpb",999)) for r in results if r.get("val_bpb")), default=999)
        print(f"\n{'='*70}")
        print(f"RUN {run_count} | {datetime.now().strftime('%H:%M:%S')} | best={best:.4f}")

        response = ask_qwen(format_history(results),
                           f"bpb={last_result.get('val_bpb','?')}" if last_result else "First run")
        config, reasoning = (None, "")
        if response:
            config, reasoning = parse_response(response)
            if config:
                print(f"  Qwen: {reasoning[:120]}")

        if config is None:
            config = {
                "num_unique_layers": random.choice([4, 5, 6, 7, 8]),
                "num_loops": random.choice([2, 3]),
                "cadence": random.choice([1, 2, 3]),
                "lr": random.choice([1e-3, 1.5e-3, 2e-3, 3e-3]),
                "grad_clip": random.choice([2.0, 5.0, 8.0]),
                "mlp_mult": random.choice([3, 4]),
            }
            reasoning = "fallback"

        print(f"  L={config.get('num_unique_layers','?')}x{config.get('num_loops','?')} "
              f"cad={config.get('cadence','?')} mlp={config.get('mlp_mult','?')}")
        r = run_experiment(config, f"frug2_{run_count:03d}")
        if r:
            r["reasoning"] = reasoning[:200]
            r["notes"] = reasoning[:100]
            save_result(r)
            results.append(r)
            last_result = r
            print(f"  >>> val_bpb={r.get('val_bpb', '?')}")

        if run_count % 5 == 0:
            valid = sorted([r for r in results if r.get("val_bpb") and float(r.get("val_bpb",999))<100],
                          key=lambda r: float(r["val_bpb"]))
            print(f"\nLEADERBOARD ({len(valid)} runs)")
            for i, r in enumerate(valid[:10]):
                print(f"  {i+1}. {float(r['val_bpb']):.4f} | L={r.get('num_unique_layers','?')}x{r.get('num_loops','?')} "
                      f"cad={r.get('cadence','?')} mlp={r.get('mlp_mult','?')}")

if __name__ == "__main__":
    main()
