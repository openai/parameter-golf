#!/usr/bin/env python3
"""
Sequential sweep — one experiment at a time (M3 Ultra can't do parallel MLX).
Prioritized by expected impact. 1000 iters each, ~28 min per experiment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import run_experiment, print_leaderboard, RESULTS_FILE
import json

ITERS = 1000

# Prioritized — highest impact first (based on leaderboard research)
experiments = [
    # Already known from parallel sweep: 13L best, 12L close, mlp2x worse, muon095 worse
    # Skip those and test remaining high-impact ones
    ("12L_bigram8192", {"BIGRAM_HASH_SIZE": "8192"}),
    ("12L_nomtp", {"MTP_NUM_HEADS": "0"}),
    ("12L_mtp3", {"MTP_NUM_HEADS": "3"}),
    ("12L_rope32", {"ROPE_DIMS": "32"}),
    ("12L_rope0_full", {"ROPE_DIMS": "0"}),
    ("12L_qat020", {"QAT_START_FRACTION": "0.20"}),
    ("12L_noclip", {"GRAD_CLIP_NORM": "0"}),
    ("12L_zloss0", {"Z_LOSS_WEIGHT": "0"}),
    ("12L_warmdown5000", {"WARMDOWN_ITERS": "5000"}),
    ("12L_ema0998", {"EMA_DECAY": "0.998"}),
    ("13L_mtp2", {"NUM_LAYERS": "13"}),  # best layer count + confirm MTP
]

# Skip already done
def get_done():
    if not RESULTS_FILE.exists():
        return set()
    done = set()
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                r = json.loads(line)
                if r.get("val_bpb") and r.get("exit_code") == 0:
                    done.add(r["name"])
            except:
                pass
    return done

if __name__ == "__main__":
    done = get_done()
    remaining = [(n, e) for n, e in experiments if n not in done]
    print(f"Done: {len(done)}, Remaining: {len(remaining)}")
    print(f"Estimated: {len(remaining) * 28 / 60:.1f} hours (sequential)\n")

    for name, env in remaining:
        try:
            run_experiment(name, ITERS, env)
        except Exception as e:
            print(f"FAILED: {name}: {e}")
        print()

    print("\nFINAL LEADERBOARD:")
    print_leaderboard()
