#!/usr/bin/env python3
"""
Ablation runner for parameter-golf experiments.
Runs v5_sota_beater.py with different configurations on 1xH100,
collecting pre-quant and post-quant BPB for each.

Usage: python3 ablation_runner.py [--experiment NAME]
       python3 ablation_runner.py --list
"""

import subprocess, sys, os, time, json, argparse

# All experiments: name -> env var overrides
EXPERIMENTS = {
    # Baseline: current v5 with TTT disabled
    "baseline": {
        "TTT_ENABLED": "0",
        "XSA_LAST_N": "5",
    },

    # === TTT VARIANTS ===
    # 1. SGD TTT (our current v5 recipe)
    "ttt_sgd_3ep_freeze2": {
        "TTT_ENABLED": "1",
        "TTT_LR": "0.002",
        "TTT_EPOCHS": "3",
        "TTT_MOMENTUM": "0.9",
        "TTT_FREEZE_BLOCKS": "2",
        "TTT_OPTIMIZER": "sgd",
    },
    # 2. AdamW TTT (SOTA recipe from PR #442)
    "ttt_adamw_10ep_nofreeze": {
        "TTT_ENABLED": "1",
        "TTT_LR": "0.0005",
        "TTT_EPOCHS": "10",
        "TTT_FREEZE_BLOCKS": "0",
        "TTT_OPTIMIZER": "adamw",
        "TTT_WEIGHT_DECAY": "0.0",
    },
    # 3. AdamW TTT aggressive (20 epochs)
    "ttt_adamw_20ep_nofreeze": {
        "TTT_ENABLED": "1",
        "TTT_LR": "0.0005",
        "TTT_EPOCHS": "20",
        "TTT_FREEZE_BLOCKS": "0",
        "TTT_OPTIMIZER": "adamw",
        "TTT_WEIGHT_DECAY": "0.0",
    },

    # === ARCHITECTURE VARIANTS ===
    # 4. No XSA (saves step time = more training steps)
    "no_xsa": {
        "TTT_ENABLED": "0",
        "XSA_LAST_N": "0",
    },
    # 5. No XSA + AdamW TTT (PR #398 finding: XSA hurts with TTT)
    "no_xsa_ttt_adamw": {
        "TTT_ENABLED": "1",
        "TTT_LR": "0.0005",
        "TTT_EPOCHS": "10",
        "TTT_FREEZE_BLOCKS": "0",
        "TTT_OPTIMIZER": "adamw",
        "XSA_LAST_N": "0",
    },

    # === QUANTIZATION VARIANTS ===
    # 6. GPTQ-lite (per-row optimal clip percentile)
    "gptq_lite": {
        "TTT_ENABLED": "0",
        "GPTQ_LITE": "1",
    },
    # 7. QAT threshold 0.15 + warmdown 3500
    "qat15_wd3500": {
        "TTT_ENABLED": "0",
        "QAT_THRESHOLD": "0.15",
        "WARMDOWN_ITERS": "3500",
    },

    # === NOVEL IDEAS ===
    # 8. Learned quantization grid (non-uniform levels)
    "learned_quant": {
        "TTT_ENABLED": "0",
        "LEARNED_QUANT": "1",
    },
    # 9. Value Residual (ResFormer - cache V from layer 0)
    "value_residual": {
        "TTT_ENABLED": "0",
        "VALUE_RESIDUAL": "1",
    },
    # 10. Sparse MoE (2 experts per layer)
    "sparse_moe": {
        "TTT_ENABLED": "0",
        "SPARSE_MOE": "1",
        "NUM_EXPERTS": "2",
    },
    # 11. Adaptive depth (early exit)
    "adaptive_depth": {
        "TTT_ENABLED": "0",
        "ADAPTIVE_DEPTH": "1",
    },
    # 12. Cross-layer weight interpolation
    "weight_interp": {
        "TTT_ENABLED": "0",
        "WEIGHT_INTERP": "1",
        "NUM_ANCHOR_LAYERS": "3",
    },
    # 13. Gated attention (per-head sigmoid gate)
    "gated_attention": {
        "TTT_ENABLED": "0",
        "GATED_ATTENTION": "1",
    },

    # === COMBO: best novel + best TTT ===
    # Run after individual results are in
}

# Common env for all experiments
COMMON_ENV = {
    "SEED": "42",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "11",
    "MAX_WALLCLOCK_SECONDS": "600",  # 10 min
}


def parse_results(log_text):
    """Extract key metrics from training log."""
    results = {}
    for line in log_text.split('\n'):
        line = line.strip()
        # Get last val BPB before quant
        if 'val_bpb:' in line and 'final_' not in line:
            try:
                bpb = float(line.split('val_bpb:')[1].split()[0])
                results['pre_quant_bpb'] = bpb
                # Also get step count
                if 'step:' in line:
                    step = int(line.split('step:')[1].split('/')[0])
                    results['final_step'] = step
            except: pass
        # Get step timing
        if 'step_avg:' in line:
            try:
                avg = float(line.split('step_avg:')[1].split('ms')[0])
                results['step_avg_ms'] = avg
            except: pass
        # Get post-quant results
        if 'final_int' in line and 'val_bpb:' in line:
            key = line.split()[0] if not line.startswith('final') else line.split('val_loss')[0].strip()
            # e.g. final_int6_roundtrip or final_int6_sliding_window
            for prefix in ['final_int6_roundtrip_exact', 'final_int6_sliding_window_exact',
                          'final_int6_roundtrip', 'final_int6_sliding_window',
                          'final_int5_roundtrip_exact', 'final_int5_sliding_window_exact']:
                if prefix in line:
                    bpb = float(line.split('val_bpb:')[1].split()[0])
                    results[prefix] = bpb
                    break
        # TTT results
        if 'ttt_epoch:' in line:
            results['ttt_ran'] = True
            try:
                loss = float(line.split('loss:')[1].split()[0])
                results['ttt_final_loss'] = loss
            except: pass
        # Submission size
        if 'Total submission size' in line:
            try:
                size = int(line.split(':')[1].strip().split()[0])
                results['submission_bytes'] = size
            except: pass
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--results-file', default='ablation_results.json')
    args = parser.parse_args()

    if args.list:
        for name, env in EXPERIMENTS.items():
            print(f"  {name}: {env}")
        return

    # Load existing results
    results = {}
    if os.path.exists(args.results_file):
        with open(args.results_file) as f:
            results = json.load(f)

    experiments = {args.experiment: EXPERIMENTS[args.experiment]} if args.experiment else EXPERIMENTS

    for name, exp_env in experiments.items():
        if name in results:
            print(f"SKIP {name} (already have results)")
            continue

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}")

        # Build env
        env = os.environ.copy()
        env.update(COMMON_ENV)
        env.update(exp_env)
        env["RUN_ID"] = f"ablation_{name}"

        # Run training
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "experiments/v5_sota_beater.py"],
            env=env, capture_output=True, text=True, timeout=900  # 15 min max
        )
        elapsed = time.time() - t0

        log = proc.stdout + proc.stderr

        # Save full log
        log_path = f"logs/ablation_{name}.txt"
        os.makedirs("logs", exist_ok=True)
        with open(log_path, 'w') as f:
            f.write(log)

        # Parse results
        exp_results = parse_results(log)
        exp_results['elapsed_seconds'] = elapsed
        exp_results['env'] = exp_env
        exp_results['returncode'] = proc.returncode

        results[name] = exp_results

        # Save results incrementally
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  Steps: {exp_results.get('final_step', '?')}")
        print(f"  Pre-quant BPB: {exp_results.get('pre_quant_bpb', '?')}")
        print(f"  Post-quant BPB: {exp_results.get('final_int6_roundtrip_exact', exp_results.get('final_int6_roundtrip', '?'))}")
        print(f"  Step avg: {exp_results.get('step_avg_ms', '?')}ms")
        print(f"  Time: {elapsed:.0f}s")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<30} {'Steps':>6} {'Pre-Q BPB':>10} {'Post-Q BPB':>11} {'ms/step':>8}")
    print(f"{'='*80}")
    for name, r in sorted(results.items(), key=lambda x: x[1].get('pre_quant_bpb', 99)):
        post_q = r.get('final_int6_sliding_window_exact', r.get('final_int6_roundtrip_exact', '?'))
        print(f"{name:<30} {r.get('final_step', '?'):>6} {r.get('pre_quant_bpb', '?'):>10} {post_q:>11} {r.get('step_avg_ms', '?'):>8}")


if __name__ == '__main__':
    main()
