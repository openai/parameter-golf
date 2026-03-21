#!/usr/bin/env python3
"""
Smart Autoresearch Loop for Parameter Golf
Runs on PC1 (RTX 4090), uses LLM to propose experiments based on history.
LLM: PC2 Ollama (qwen3:8b) via HTTP API — free, always running.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Config
REPO_DIR = Path.home() / "parameter-golf"
RESULTS_FILE = REPO_DIR / "autoresearch_results.tsv"
LOGS_DIR = REPO_DIR / "logs"
PYTHON = str(REPO_DIR / ".venv" / "bin" / "python")
TRAIN_SCRIPT = str(REPO_DIR / "train_gpt.py")

# LLM endpoint — PC2 Ollama
LLM_URL = "http://100.122.180.1:11434/api/generate"
LLM_MODEL = "qwen3:8b"

# Fallback: if LLM is unreachable, use this static experiment queue
FALLBACK_EXPERIMENTS = [
    # Phase 1: Baseline
    {},

    # Phase 2: Single-variable tests (isolate what helps)
    {"TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048"},
    {"NUM_LAYERS": "10"},
    {"MATRIX_LR": "0.06", "WARMDOWN_ITERS": "3600"},  # FP16 embed winner's LR
    {"MATRIX_LR": "0.032", "SCALAR_LR": "0.032", "TIED_EMBED_LR": "0.04"},  # Seq2048 winner's LR
    {"MATRIX_LR": "0.02", "SCALAR_LR": "0.02", "TIED_EMBED_LR": "0.03"},  # 10L winner's LR
    {"WARMDOWN_ITERS": "3600"},  # Just warmdown alone
    {"TRAIN_SEQ_LEN": "4096", "TTT_EVAL_SEQ_LEN": "4096"},

    # Phase 3: Combine winners from leaderboard configs
    {"TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048", "MATRIX_LR": "0.032", "SCALAR_LR": "0.032", "TIED_EMBED_LR": "0.04"},  # Exact seq2048 leaderboard config
    {"NUM_LAYERS": "10", "MATRIX_LR": "0.02", "SCALAR_LR": "0.02", "TIED_EMBED_LR": "0.03"},  # Exact 10L leaderboard config
    {"TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048", "NUM_LAYERS": "10"},
    {"TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048", "NUM_LAYERS": "10", "MATRIX_LR": "0.032", "SCALAR_LR": "0.032"},
    {"MATRIX_LR": "0.06", "WARMDOWN_ITERS": "3600", "NUM_LAYERS": "10"},

    # Phase 4: Aggressive combos
    {"TRAIN_SEQ_LEN": "4096", "TTT_EVAL_SEQ_LEN": "4096", "MATRIX_LR": "0.032", "SCALAR_LR": "0.032"},
    {"TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048", "NUM_LAYERS": "10", "MATRIX_LR": "0.06", "WARMDOWN_ITERS": "3600"},
    {"TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048", "NUM_LAYERS": "10", "MATRIX_LR": "0.02", "SCALAR_LR": "0.02", "WARMDOWN_ITERS": "3600"},

    # Phase 5: TTT tuning on best config
    {"TTT_LORA_RANK": "16", "TTT_LORA_LR": "0.005"},
    {"TTT_LORA_RANK": "16", "TTT_LORA_LR": "0.02"},
    {"TTT_EVAL_SEQ_LEN": "2048", "TTT_LORA_RANK": "16"},

    # Phase 6: Architecture exploration
    {"MODEL_DIM": "640", "NUM_HEADS": "10", "NUM_KV_HEADS": "5", "NUM_LAYERS": "8"},
    {"MODEL_DIM": "448", "NUM_HEADS": "8", "NUM_KV_HEADS": "4", "NUM_LAYERS": "12"},
    {"QK_GAIN_INIT": "2.0"},
    {"LOGIT_SOFTCAP": "50.0"},
    {"ROPE_BASE": "50000.0", "TRAIN_SEQ_LEN": "2048", "TTT_EVAL_SEQ_LEN": "2048"},
    {"MUON_MOMENTUM": "0.90"},
    {"MUON_MOMENTUM": "0.98"},
    {"EMBED_LR": "0.3", "TIED_EMBED_LR": "0.03"},
    {"EMBED_LR": "1.0", "TIED_EMBED_LR": "0.08"},
]

# All tunable env vars with defaults and descriptions
HYPERPARAMETER_SPEC = """
Environment variable hyperparameters for train_gpt.py:

MODEL ARCHITECTURE:
- NUM_LAYERS (default: 9) — transformer blocks. More = more capacity but slower per step
- MODEL_DIM (default: 512) — hidden dimension. Wider = more capacity
- NUM_HEADS (default: 8) — attention heads
- NUM_KV_HEADS (default: 4) — KV heads for GQA. Must divide NUM_HEADS
- MLP_MULT (default: 2) — MLP expansion factor
- TIE_EMBEDDINGS (default: 1) — tie input/output embeddings (saves params)
- ROPE_BASE (default: 10000.0) — RoPE frequency base
- LOGIT_SOFTCAP (default: 30.0) — logit soft capping value
- QK_GAIN_INIT (default: 1.5) — QK normalization gain init

TRAINING:
- TRAIN_SEQ_LEN (default: 1024) — sequence length. Longer = better modeling but fewer steps. Top teams use 2048-4096.
- TRAIN_BATCH_TOKENS (default: 524288) — tokens per training step
- ITERATIONS (default: 20000) — max training steps (usually wallclock-capped)
- WARMUP_STEPS (default: 20) — LR warmup steps
- WARMDOWN_ITERS (default: 1200) — LR warmdown iterations. Top team used 3600 (default assumes more steps than 10min allows)

OPTIMIZER (Muon + AdamW):
- EMBED_LR (default: 0.6) — embedding learning rate
- HEAD_LR (default: 0.008) — output head learning rate
- TIED_EMBED_LR (default: 0.05) — tied embedding learning rate
- MATRIX_LR (default: 0.04) — Muon matrix learning rate
- SCALAR_LR (default: 0.04) — scalar parameter learning rate
- MUON_MOMENTUM (default: 0.95) — Muon momentum

TEST-TIME TRAINING (TTT):
- TTT_LORA_RANK (default: 8) — LoRA rank for TTT
- TTT_LORA_LR (default: 0.01) — LoRA learning rate for TTT
- TTT_EVAL_SEQ_LEN (default: 1024) — sequence length for TTT evaluation
- TTT_BATCH_SIZE (default: 64) — batch size for TTT
- TTT_CHUNK_SIZE (default: 256) — chunk size for TTT

CONSTRAINTS:
- Model must fit in 16MB when quantized (int8+zlib)
- Training capped at 600 seconds wall clock (MAX_WALLCLOCK_SECONDS)
- On this GPU (RTX 4090, 24GB VRAM) we get ~320 steps in 10 min without torch.compile
- Submission scored on val_bpb (bits per byte) — LOWER IS BETTER
- NUM_HEADS must be divisible by NUM_KV_HEADS
- MODEL_DIM must be divisible by NUM_HEADS
"""

LEADERBOARD_CONTEXT = """
Current leaderboard (scored on 8xH100, full 20k steps):
1. 1.1748 bpb — 10 layers + Muon WD(0.02) + spectral embed init + FP16 embedding + sliding window eval stride=64 + residual mixing
2. 1.1925 bpb — Sliding window eval stride=64 (same training as baseline, eval trick only)
3. 1.1928 bpb — LoRA test-time training
4. 1.2014 bpb — Seq length 4096 + TIED_EMBED_LR=0.04 MATRIX_LR=0.032 SCALAR_LR=0.032
5. 1.2060 bpb — Seq length 2048 + same LR tuning as #4
6. 1.2147 bpb — 10 layers + mixed int8/int6 compression + MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
7. 1.2197 bpb — FP16 embedding + WARMDOWN_ITERS=3600 + MATRIX_LR=0.06
8. 1.2244 bpb — Naive baseline (9 layers, 512 dim, 1024 seq)

KEY INSIGHTS FROM TOP SUBMISSIONS:
- Default MATRIX_LR=0.04 is suboptimal. #4/#5 used 0.032, #6 used 0.02, #7 used 0.06. Try range 0.02-0.06.
- WARMDOWN_ITERS=3600 helped #7 (default 1200 assumes more steps than you get)
- Seq length 2048-4096 consistently helps (#4, #5)
- Lower LRs with more layers helps (#6: 10 layers + LR=0.02)
- Higher LR (0.06) with default layers also helps (#7)
- TIE_EMBEDDINGS=1 is always on

Our runs are on a single RTX 4090, wall-clock capped at 10min (~320 steps without torch.compile).
Absolute bpb will be higher than H100, but RELATIVE improvements between configs transfer.
A config that scores better here will also score better on H100.
"""


def init_results():
    """Initialize results file if it doesn't exist."""
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write("experiment\tval_bpb\tval_bpb_int8_zlib\tval_bpb_ttt_lora\tsteps\tpeak_vram_mb\tsubmission_mb\tstatus\tconfig\n")


def read_results():
    """Read all results so far."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE) as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return []
        header = lines[0].strip().split("\t")
        for line in lines[1:]:
            vals = line.strip().split("\t")
            if len(vals) >= len(header):
                results.append(dict(zip(header, vals)))
    return results


def format_results_for_llm(results):
    """Format results history for the LLM."""
    if not results:
        return "No experiments run yet. Start with baseline (all defaults)."
    
    lines = ["Previous experiments (sorted by val_bpb, lower is better):"]
    sorted_results = sorted(results, key=lambda r: float(r.get("val_bpb", "99")))
    for r in sorted_results:
        status = r.get("status", "unknown")
        bpb = r.get("val_bpb", "?")
        bpb_int8 = r.get("val_bpb_int8_zlib", "?")
        bpb_ttt = r.get("val_bpb_ttt_lora", "?")
        config = r.get("config", "defaults")
        steps = r.get("steps", "?")
        vram = r.get("peak_vram_mb", "?")
        sub_mb = r.get("submission_mb", "?")
        name = r.get("experiment", "?")
        lines.append(f"  {name}: val_bpb={bpb} int8_zlib={bpb_int8} ttt={bpb_ttt} steps={steps} vram={vram}MB sub={sub_mb}MB [{status}] config: {config}")
    
    return "\n".join(lines)


def ask_llm_for_experiment(results, experiment_num):
    """Ask the LLM to propose the next experiment."""
    import urllib.request
    
    results_text = format_results_for_llm(results)
    
    prompt = f"""You are an ML researcher optimizing a small language model for the Parameter Golf competition. Respond with ONLY a JSON object, no thinking tags, no explanation before or after the JSON.

{HYPERPARAMETER_SPEC}

{LEADERBOARD_CONTEXT}

{results_text}

Propose the NEXT experiment. Based on what worked and didn't work above, choose env var overrides that you think will improve val_bpb.

RULES:
- Only use the env vars listed above
- NUM_HEADS must be divisible by NUM_KV_HEADS  
- MODEL_DIM must be divisible by NUM_HEADS
- Keep submission size under 16MB (watch MODEL_DIM and NUM_LAYERS)
- Be creative but informed by the results
- If something worked, try combining it with other improvements
- If something hurt, avoid that direction
- Try ONE new thing at a time when exploring, combine winners when exploiting

Respond with ONLY a JSON object, no other text. Format:
{{"name": "short_descriptive_name", "config": {{"ENV_VAR": "value", ...}}, "reasoning": "one sentence why"}}
"""

    try:
        data = json.dumps({
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 2000}
        }).encode()
        
        req = urllib.request.Request(LLM_URL, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            # qwen3 puts everything in 'thinking' field, 'response' is often empty
            response_text = result.get("response", "") or result.get("thinking", "")
            
            # Strip thinking tags if present
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            # Extract JSON from response — try multiple patterns
            json_match = None
            # Try finding a JSON block with nested config object
            for pattern in [
                r'\{[^{}]*"name"[^{}]*"config"\s*:\s*\{[^{}]*\}[^{}]*\}',
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{[^{}]*"name".*\})',
            ]:
                m = re.search(pattern, response_text, re.DOTALL)
                if m:
                    json_match = m
                    break
            if not json_match:
                # Last resort — find any JSON-like object
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group()
                proposal = json.loads(json_str)
                print(f"LLM proposed: {proposal.get('name', '?')} — {proposal.get('reasoning', '?')}")
                return proposal
            else:
                print(f"LLM response didn't contain valid JSON: {response_text[:200]}")
                return None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


def run_experiment(name, env_overrides, experiment_num):
    """Run a single training experiment."""
    log_file = LOGS_DIR / f"autoresearch_{experiment_num}_{name}.log"
    
    print(f"\n{'='*60}")
    print(f"Experiment {experiment_num}: {name}")
    print(f"Config: {env_overrides}")
    print(f"Log: {log_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Build environment
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    # Set predictable RUN_ID so we can find the log
    run_id = f"autoresearch_{experiment_num}_{name}"
    env["RUN_ID"] = run_id
    for k, v in env_overrides.items():
        env[k] = str(v)
    
    # The real log will be at logs/<run_id>.txt (written by train_gpt.py internally)
    real_log = LOGS_DIR / f"{run_id}.txt"
    
    # Run training
    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            proc = subprocess.run(
                [PYTHON, TRAIN_SCRIPT],
                cwd=str(REPO_DIR),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=2400  # 40 min hard timeout (10 min training + 15 min int8 eval + 15 min TTT eval)
            )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        print("TIMEOUT — killed after 20 min")
        exit_code = -1
    except Exception as e:
        print(f"ERROR: {e}")
        exit_code = -1
    
    # Use the train_gpt.py internal log if our captured log is empty
    if log_file.exists() and log_file.stat().st_size == 0 and real_log.exists():
        log_file = real_log
    
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.0f}s (exit code: {exit_code})")
    
    # Parse results
    result = {
        "experiment": f"{experiment_num}_{name}",
        "val_bpb": "0.000000",
        "val_bpb_int8_zlib": "0.000000",
        "val_bpb_ttt_lora": "0.000000",
        "steps": "0",
        "peak_vram_mb": "0",
        "submission_mb": "0.0",
        "status": "crash",
        "config": json.dumps(env_overrides) if env_overrides else "defaults"
    }
    
    if exit_code == 0 and log_file.exists():
        log_content = log_file.read_text()
        
        # Get last val_bpb from step lines
        step_bpbs = re.findall(r'step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', log_content)
        if step_bpbs:
            result["val_bpb"] = step_bpbs[-1]
        
        # Get steps completed
        steps = re.findall(r'step:(\d+)/\d+', log_content)
        if steps:
            result["steps"] = steps[-1]
        
        # Get int8+zlib bpb
        int8_match = re.search(r'final_int8_zlib_roundtrip val_loss:[\d.]+ val_bpb:([\d.]+)', log_content)
        if int8_match:
            result["val_bpb_int8_zlib"] = int8_match.group(1)
        
        # Get TTT LoRA bpb
        ttt_match = re.search(r'final_int8_ttt_lora val_loss:[\d.]+ val_bpb:([\d.]+)', log_content)
        if ttt_match:
            result["val_bpb_ttt_lora"] = ttt_match.group(1)
        
        # Get peak VRAM
        vram_match = re.search(r'peak memory allocated: (\d+) MiB', log_content)
        if vram_match:
            result["peak_vram_mb"] = vram_match.group(1)
        
        # Get submission size
        sub_match = re.search(r'Total submission size int8\+zlib: (\d+) bytes', log_content)
        if sub_match:
            result["submission_mb"] = f"{int(sub_match.group(1)) / 1024 / 1024:.1f}"
        
        result["status"] = "done"
    elif exit_code != 0 and log_file.exists():
        # Print last few lines for debugging
        lines = log_file.read_text().strip().split("\n")
        print("--- CRASH LOG (last 10 lines) ---")
        for line in lines[-10:]:
            print(f"  {line}")
        print("--- END ---")
    
    # Append to results file
    with open(RESULTS_FILE, "a") as f:
        cols = ["experiment", "val_bpb", "val_bpb_int8_zlib", "val_bpb_ttt_lora", 
                "steps", "peak_vram_mb", "submission_mb", "status", "config"]
        f.write("\t".join(result[c] for c in cols) + "\n")
    
    print(f"\nResults: val_bpb={result['val_bpb']} | int8={result['val_bpb_int8_zlib']} | ttt={result['val_bpb_ttt_lora']} | steps={result['steps']} | vram={result['peak_vram_mb']}MB | submission={result['submission_mb']}MB")
    
    return result


def main():
    os.chdir(REPO_DIR)
    LOGS_DIR.mkdir(exist_ok=True)
    init_results()
    
    print("=" * 60)
    print("Parameter Golf — Smart Autoresearch Loop")
    print(f"GPU: RTX 4090 | LLM: {LLM_MODEL} @ PC2")
    print(f"Results: {RESULTS_FILE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    experiment_num = len(read_results())
    # Skip baseline in fallback if we already have results
    fallback_idx = 1 if experiment_num > 0 else 0
    use_llm = True
    
    # Test LLM connectivity
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://100.122.180.1:11434/api/tags",
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
        print("✓ LLM (PC2 Ollama) reachable")
    except Exception as e:
        print(f"✗ LLM unreachable ({e}), using fallback experiment queue")
        use_llm = False
    
    while True:
        results = read_results()
        
        # Get next experiment
        name = None
        config = {}
        
        if experiment_num == 0:
            # Always start with baseline
            name = "baseline"
            config = {}
            print("\nStarting with baseline (all defaults)...")
        elif use_llm:
            # Ask LLM for next experiment
            print(f"\nAsking LLM for experiment {experiment_num}...")
            proposal = ask_llm_for_experiment(results, experiment_num)
            if proposal:
                name = proposal.get("name", f"llm_exp_{experiment_num}")
                config = proposal.get("config", {})
                # Sanitize name
                name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:50]
            else:
                # LLM failed, use fallback
                if fallback_idx < len(FALLBACK_EXPERIMENTS):
                    config = FALLBACK_EXPERIMENTS[fallback_idx]
                    name = f"fallback_{fallback_idx}"
                    fallback_idx += 1
                    print(f"LLM failed, using fallback experiment: {name}")
                else:
                    print("No more fallback experiments and LLM failed. Stopping.")
                    break
        else:
            # No LLM, use fallback queue
            if fallback_idx < len(FALLBACK_EXPERIMENTS):
                config = FALLBACK_EXPERIMENTS[fallback_idx]
                name = f"fallback_{fallback_idx}"
                fallback_idx += 1
            else:
                print("All fallback experiments done. Stopping.")
                break
        
        # Run the experiment
        result = run_experiment(name, config, experiment_num)
        experiment_num += 1
        
        # Print running best
        all_results = read_results()
        done_results = [r for r in all_results if r.get("status") == "done" and float(r.get("val_bpb", "99")) > 0]
        if done_results:
            best = min(done_results, key=lambda r: float(r["val_bpb"]))
            print(f"\n🏆 Best so far: {best['experiment']} — val_bpb={best['val_bpb']} (int8={best.get('val_bpb_int8_zlib', '?')})")
        
        print(f"\n{'─'*60}")
        print(f"Completed {experiment_num} experiments. Continuing...")
        print(f"{'─'*60}")


if __name__ == "__main__":
    # Force unbuffered output for nohup
    import functools
    print = functools.partial(print, flush=True)
    main()
