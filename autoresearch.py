"""
Fractal Auto-Research: LLM-Guided Overnight Optimization
==========================================================
Uses local Qwen (via Ollama) to analyze experiment results and
propose the next configuration. The LLM sees the full history
and reasons about what to try next.

Loop:
  1. Run experiment with current config
  2. Send results + history to Qwen
  3. Qwen proposes next config with reasoning
  4. Parse config, run it
  5. Repeat forever

Usage:
  source .venv/bin/activate
  nohup python autoresearch.py > autoresearch.log 2>&1 &
  tail -f autoresearch.log
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
RESULTS_FILE = "autoresearch_results.csv"
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
    "iterations": 300,
    "eval_tokens": 100000,
    "max_seconds": 300,
    "batch_tokens": 32768,
    "seq_len": 1024,
    "seed": 1337,
}

SYSTEM_PROMPT = """You are an ML research assistant optimizing a fractal transformer architecture for a language modeling competition.

GOAL: Find the configuration that minimizes val_bpb (bits per byte) on a validation set.

ARCHITECTURE: Fractal weight-shared transformer. A small number of unique transformer blocks are looped multiple times to create effective depth.

TRAINING PATTERN: "Cadence" alternates between fractal steps (all loops fire, deep computation) and normalize steps (single clean pass, fast). cadence=2 means F/N/F/N, cadence=3 means one fractal every 3 steps, cadence=1 means always fractal, cadence=0 means never fractal.

CONFIGURABLE PARAMETERS:
- num_unique_layers: number of unique transformer blocks (2-8). More layers = more unique capacity but narrower model (auto-sized to match param budget)
- num_loops: how many times to loop through the blocks (1-5). More loops = deeper effective network but slower fractal steps
- cadence: how often fractal fires (0=never, 1=always, 2=every other, 3=every 3rd, etc.)
- cadence_offset: which position in the cadence cycle is fractal (0 to cadence-1)
- lr: learning rate (1e-4 to 2e-3). Higher = faster learning but risk instability
- grad_clip: gradient clipping norm (0.1 to 5.0). Fractal accumulates gradients from multiple loops — may need higher clip than standard
- mlp_mult: MLP expansion factor (2 or 3). 3x = more params per block but fewer blocks fit in budget

CONSTRAINTS:
- Total unique params are auto-sized to match ~17M parameter budget
- More unique layers with same budget = narrower dim (less expressive per layer)
- More loops = proportionally slower fractal steps (2 loops = 2x, 3 loops = 3x)
- Normalize steps are always fast (~10ms), fractal steps scale with loops (~100ms per loop)
- 300 training steps per experiment, each ~2-3 minutes

KEY INSIGHTS FROM PRIOR WORK:
- Orthogonal loop position embeddings help (each loop and normalize operate in non-interfering subspaces)
- Cadence 2 (F/N) works well — normalize steps become beneficial after ~500 steps
- Weight sharing lets wider layers compensate for fewer unique blocks
- Gradient clipping may need to be looser for fractal (3 loops = ~3x gradient accumulation)

Respond with ONLY a JSON object (no markdown, no code fences):
{
  "reasoning": "Brief explanation of why this config (2-3 sentences)",
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

# ─── OLLAMA ───────────────────────────────────────────────────────────────────

def ask_qwen(history_text, last_result_text):
    prompt = f"""Here are ALL experiment results so far (sorted by val_bpb, best first):

{history_text}

The most recent experiment result:
{last_result_text}

Based on the patterns in these results, propose the NEXT experiment configuration to try. Look for:
1. Which axes (layers, loops, cadence, lr, clip) have the most impact
2. What promising regions haven't been explored yet
3. Whether to exploit (refine near the best) or explore (try something different)

Do NOT repeat a configuration that has already been tested. Try something new."""

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
            content = data.get("message", {}).get("content", "")
            return content
    except Exception as e:
        print(f"  Qwen error: {e}")
        return None


def parse_qwen_response(text):
    """Extract JSON config from Qwen's response."""
    if not text:
        return None, "no response"

    # Try to find JSON in the response
    # Handle potential markdown code fences
    clean = text.strip()
    if "```" in clean:
        parts = clean.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                clean = p
                break

    # Find the JSON object
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None, f"no JSON found in: {text[:100]}"

    try:
        obj = json.loads(clean[start:end])
        reasoning = obj.get("reasoning", "")
        config = obj.get("config", obj)
        # Validate
        validated = {}
        if "num_unique_layers" in config:
            validated["num_unique_layers"] = max(1, min(8, int(config["num_unique_layers"])))
        if "num_loops" in config:
            validated["num_loops"] = max(1, min(5, int(config["num_loops"])))
        if "cadence" in config:
            validated["cadence"] = max(0, min(10, int(config["cadence"])))
        if "cadence_offset" in config:
            cad = validated.get("cadence", 2)
            validated["cadence_offset"] = max(0, min(cad - 1, int(config["cadence_offset"]))) if cad > 0 else 0
        if "lr" in config:
            validated["lr"] = max(1e-5, min(0.01, float(config["lr"])))
        if "grad_clip" in config:
            validated["grad_clip"] = max(0.05, min(10.0, float(config["grad_clip"])))
        if "mlp_mult" in config:
            validated["mlp_mult"] = int(config["mlp_mult"])
            if validated["mlp_mult"] not in [2, 3, 4]:
                validated["mlp_mult"] = 2
        return validated, reasoning
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, f"parse error: {e} | {text[:200]}"


# ─── RUNNER ───────────────────────────────────────────────────────────────────

def format_history(results):
    if not results:
        return "No experiments run yet. Start with a diverse exploration."
    valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
    valid.sort(key=lambda r: float(r["val_bpb"]))
    lines = []
    for r in valid[:30]:  # top 30
        lines.append(
            f"bpb={float(r['val_bpb']):.4f} | "
            f"layers={r.get('num_unique_layers','?')} loops={r.get('num_loops','?')} "
            f"cadence={r.get('cadence','?')} offset={r.get('cadence_offset','?')} "
            f"lr={float(r.get('lr',0)):.1e} clip={float(r.get('grad_clip',0)):.1f} "
            f"mlp={r.get('mlp_mult','?')} | {r.get('notes','')}"
        )
    return "\n".join(lines)


def format_last_result(result):
    if not result:
        return "First run — no previous result."
    return (
        f"val_bpb={result.get('val_bpb','?')} | "
        f"layers={result.get('num_unique_layers','?')} loops={result.get('num_loops','?')} "
        f"cadence={result.get('cadence','?')} lr={result.get('lr','?')} "
        f"clip={result.get('grad_clip','?')} mlp={result.get('mlp_mult','?')} "
        f"steps={result.get('steps','?')} avg_ms={result.get('avg_ms','?')}"
    )


def run_experiment(config, run_id):
    cfg = {**RUN_DEFAULTS, **config}
    # Fill defaults for missing keys
    cfg.setdefault("cadence", 2)
    cfg.setdefault("cadence_offset", 0)
    cfg.setdefault("num_unique_layers", 3)
    cfg.setdefault("num_loops", 3)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("mlp_mult", 2)

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
    if cfg.get("model_dim", 0) > 0:
        cmd.extend(["--model-dim", str(cfg["model_dim"])])
    if cfg.get("gravity", False):
        cmd.append("--gravity")

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
        "cadence": cfg["cadence"], "cadence_offset": cfg["cadence_offset"],
        "num_unique_layers": cfg["num_unique_layers"], "num_loops": cfg["num_loops"],
        "lr": cfg["lr"], "grad_clip": cfg["grad_clip"],
        "mlp_mult": cfg["mlp_mult"], "model_dim": cfg.get("model_dim", 0),
    }
    stdout = result.stdout
    for line in stdout.split("\n"):
        if "val_bpb:" in line and "RESULTS" not in line and "val_bpb:enabled" not in line:
            try:
                for p in line.split():
                    if p.startswith("val_bpb:"):
                        parsed["val_bpb"] = float(p.split(":")[1])
            except (ValueError, IndexError):
                pass
        if line.startswith("steps:"):
            try:
                parts = line.split()
                parsed["steps"] = int(parts[0].split(":")[1])
                for p in parts:
                    if p.startswith("(F:"):
                        parsed["f_steps"] = int(p.split(":")[1])
                    if p.startswith("N:"):
                        parsed["n_steps"] = int(p.rstrip(")").split(":")[1])
            except (ValueError, IndexError):
                pass
        if "avg_ms:" in line:
            try:
                for p in line.split():
                    if p.startswith("avg_ms:"):
                        parsed["avg_ms"] = float(p.split(":")[1].rstrip("ms/step"))
            except (ValueError, IndexError):
                pass
        if "time:" in line and "train_time" not in line:
            try:
                for p in line.split():
                    if p.startswith("time:"):
                        parsed["time_s"] = float(p.split(":")[1].rstrip("s"))
            except (ValueError, IndexError):
                pass
        if "params:" in line and "model_params" not in line:
            try:
                for p in line.split():
                    if p.startswith("params:"):
                        parsed["params"] = p.split(":")[1].replace(",", "")
            except (ValueError, IndexError):
                pass

    return parsed


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


def fallback_config(results):
    """If Qwen fails, generate a random config."""
    return {
        "num_unique_layers": random.choice([2, 3, 4, 5, 6]),
        "num_loops": random.choice([1, 2, 3, 4]),
        "cadence": random.choice([0, 1, 2, 3]),
        "cadence_offset": 0,
        "lr": random.choice([1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3]),
        "grad_clip": random.choice([0.3, 0.5, 1.0, 1.5, 2.0]),
        "mlp_mult": random.choice([2, 3]),
    }


# ─── SEED RUNS ────────────────────────────────────────────────────────────────

SEED_CONFIGS = [
    {"num_unique_layers": 3, "num_loops": 3, "cadence": 2, "lr": 3e-4, "grad_clip": 1.0, "mlp_mult": 2,
     "notes": "seed: 3x3 cadence2 (our baseline)"},
    {"num_unique_layers": 3, "num_loops": 3, "cadence": 1, "lr": 3e-4, "grad_clip": 1.0, "mlp_mult": 2,
     "notes": "seed: always fractal control"},
    {"num_unique_layers": 3, "num_loops": 3, "cadence": 0, "lr": 3e-4, "grad_clip": 1.0, "mlp_mult": 2,
     "notes": "seed: never fractal control"},
    {"num_unique_layers": 4, "num_loops": 3, "cadence": 2, "lr": 3e-4, "grad_clip": 0.5, "mlp_mult": 2,
     "notes": "seed: 4x3 loose clip"},
    {"num_unique_layers": 3, "num_loops": 2, "cadence": 2, "lr": 5e-4, "grad_clip": 1.0, "mlp_mult": 2,
     "notes": "seed: 3x2 high lr"},
]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FRACTAL AUTO-RESEARCH — Qwen-Guided Overnight Optimization")
    print(f"Model: {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 70)

    # Verify Qwen is reachable
    try:
        test = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        print("Ollama: connected")
    except Exception as e:
        print(f"WARNING: Ollama not reachable ({e}). Will use fallback random search.")

    results = load_results()
    run_count = len(results)
    last_result = None

    # Run seed configs first (if not already done)
    if run_count < len(SEED_CONFIGS):
        print(f"\n>>> SEED PHASE: {len(SEED_CONFIGS)} initial configs")
        for i, cfg in enumerate(SEED_CONFIGS):
            if i < run_count:
                continue
            run_count += 1
            rid = f"seed_{run_count:03d}"
            print(f"\n[seed {run_count}] {cfg.get('notes', '')}")
            print(f"  L={cfg['num_unique_layers']} lp={cfg['num_loops']} "
                  f"cad={cfg['cadence']} lr={cfg['lr']:.1e} clip={cfg['grad_clip']}")
            r = run_experiment(cfg, rid)
            if r:
                r["notes"] = cfg.get("notes", "")
                r["reasoning"] = "seed config"
                save_result(r)
                results.append(r)
                last_result = r
                bpb = r.get("val_bpb", "?")
                print(f"  >>> val_bpb={bpb}")

    # Main LLM-guided loop
    qwen_failures = 0
    while True:
        run_count += 1
        print(f"\n{'='*70}")
        print(f"RUN {run_count} | {datetime.now().strftime('%H:%M:%S')} | "
              f"best={min((float(r.get('val_bpb',999)) for r in results if r.get('val_bpb')), default=999):.4f}")
        print(f"{'='*70}")

        # Ask Qwen for next config
        history_text = format_history(results)
        last_text = format_last_result(last_result)

        print("  Asking Qwen...")
        response = ask_qwen(history_text, last_text)

        config = None
        reasoning = ""
        if response:
            config, reasoning = parse_qwen_response(response)
            if config:
                print(f"  Qwen says: {reasoning[:100]}")
                print(f"  Config: {json.dumps(config)}")
                qwen_failures = 0
            else:
                print(f"  Parse failed: {reasoning[:100]}")
                qwen_failures += 1
        else:
            print("  Qwen unavailable")
            qwen_failures += 1

        # Fallback if Qwen fails
        if config is None:
            config = fallback_config(results)
            reasoning = f"fallback random (qwen failures: {qwen_failures})"
            print(f"  Fallback: {json.dumps(config)}")

        # Fix cadence_offset
        cad = config.get("cadence", 2)
        if cad > 0:
            config["cadence_offset"] = min(config.get("cadence_offset", 0), cad - 1)
        else:
            config["cadence_offset"] = 0

        # Run it
        rid = f"qwen_{run_count:03d}"
        print(f"\n  Running: L={config.get('num_unique_layers',3)} "
              f"lp={config.get('num_loops',3)} cad={config.get('cadence',2)} "
              f"lr={config.get('lr',3e-4):.1e} clip={config.get('grad_clip',1.0)}")

        r = run_experiment(config, rid)
        if r:
            r["reasoning"] = reasoning[:200]
            r["notes"] = reasoning[:100]
            save_result(r)
            results.append(r)
            last_result = r
            bpb = r.get("val_bpb", "?")
            print(f"\n  >>> val_bpb={bpb}")
        else:
            print("  Run failed")
            last_result = None

        # Print leaderboard every 5 runs
        if run_count % 5 == 0:
            valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
            valid.sort(key=lambda r: float(r["val_bpb"]))
            print(f"\n{'='*80}")
            print(f"LEADERBOARD (top 10 of {len(valid)} runs)")
            print(f"{'='*80}")
            for i, r in enumerate(valid[:10]):
                print(f"  {i+1:>2}. bpb={float(r['val_bpb']):>7.4f} | "
                      f"L={r.get('num_unique_layers','?')}x{r.get('num_loops','?')} "
                      f"cad={r.get('cadence','?')} lr={float(r.get('lr',0)):.1e} "
                      f"clip={float(r.get('grad_clip',0)):.1f}")


if __name__ == "__main__":
    main()
