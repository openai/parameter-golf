#!/usr/bin/env python3
"""
Codex Spark — autonomous spark research coordinator for Bandit_Wagon.

Manages 0.25-scale architecture signal experiments on the spark (1 GPU, 150s),
interprets results, and decides the best arm to promote to the full 8×H100 run.

Usage:
    python Nitrust/codex_spark.py                  # run signal ablations + analyze
    python Nitrust/codex_spark.py --task signal    # same as default
    python Nitrust/codex_spark.py --task analyze   # analyze most recent results only
    python Nitrust/codex_spark.py --task iterate   # analyze + run follow-up combo if warranted
"""

import anyio
import argparse
import sys
from pathlib import Path

try:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        ResultMessage,
        AssistantMessage,
        TextBlock,
        CLINotFoundError,
        CLIConnectionError,
    )
except ImportError:
    print("ERROR: claude_agent_sdk not found.")
    print("Install with:  pip install claude-agent-sdk")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── System prompt — research context and decision rules ───────────────────────

SYSTEM_PROMPT = """You are the Codex Spark research coordinator for the Bandit_Wagon experiment.
Your job is to manage architecture signal ablations on the spark machine (1 GPU, 150s wallclock)
and decide which arm should be promoted to a full 8×H100 run (24-hour turnaround — expensive and slow).

## Current SOTA context

Bandit: 0.4961 BPB (3-seed mean, std=0.0003), 9.35 MB
Architecture: dim=512, 4 flat layers + 1 crawler×4 loops, inst_dim=32 FLOW, DN=0
Submission budget: 16 MB → ~6.65 MB unused headroom (~9.3M params available)

Bandit_Wagon tests two independent levers for spending that headroom:
  Width  — increase model_dim:     512 → 576 → 640
  Depth  — increase flat layers:   4 → 5 → 6

## Signal arms (0.25 scale: 150s, 1 GPU, train_gpt_h4_compiled.py)

Signal dim is proportionally scaled from production. Same relative ratios apply.

  BW-S00  dim=384  4F   anchor (mirrors production BW-00 dim=512)
  BW-S01  dim=432  4F   +12.5% width (mirrors BW-01 dim=576)
  BW-S02  dim=480  4F   +25% width   (mirrors BW-02 dim=640)
  BW-S03  dim=384  5F   +1 flat layer (mirrors BW-03)
  BW-S04  dim=384  6F   +2 flat layers (mirrors BW-04)

Results land in:
  experiments/Bandit_Wagon/results/signal_<timestamp>/summary.tsv
  experiments/Bandit_Wagon/results/signal_<timestamp>/BW-S*/run.log

## Decision rules

1. Primary metric: sliding_bpb from summary.tsv (roundtrip_bpb is a secondary check).
2. A delta of >0.003 BPB vs anchor is meaningful at this scale. Smaller = noise.
3. Width winner = best of BW-S01/S02 that beats anchor by >0.003.
4. Depth winner = best of BW-S03/S04 that beats anchor by >0.003.
5. Cases:
   - Width wins, depth does not → recommend BW-02 full run (MODEL_DIM=640, max width bet)
   - Depth wins, width does not → recommend BW-04 full run (NUM_FLAT_LAYERS=6)
   - Both win by >0.003       → run a combo arm first: dim=432 + 5F (one extra level each)
   - Neither wins             → no clear winner; recommend investigating CRAWLER_LOOPS or INST_DIM
6. For combo arms: create and run a new signal case using train_gpt_h4_compiled.py with
   MODEL_DIM=432 NUM_FLAT_LAYERS=5, same SHARED_ENV as the signal script.
7. If any arm fails (status != ok): read its run.log, diagnose, note in analysis.

## Output requirements

Always end your analysis with this exact block:

=== SPARK RECOMMENDATION ===
Winner arm: [BW-S0X or "no clear winner"]
Full run command: [exact bash command, e.g. MODEL_DIM=640 SEED=1337 bash experiments/Bandit_Wagon/run.sh]
Signal BPB delta: [e.g. BW-S02: 2.3841 vs anchor 2.4102 = -0.0261]
Confidence: [high / medium / low] — [one sentence reason]
Next step if full run fails: [fallback]
===

Be terse everywhere else. Report exact numbers, not approximations.
"""

# ── Task prompts ──────────────────────────────────────────────────────────────

TASK_PROMPTS = {
    "signal": """\
Run the Bandit_Wagon spark signal ablations, then analyze the results.

1. Check for an existing signal run: glob experiments/Bandit_Wagon/results/signal_*/summary.tsv
   - If found and all 5 arms have non-empty sliding_bpb → skip to step 3.
   - Otherwise, run: bash Nitrust/scripts/spark_bandit_wagon_signal.sh
     (This takes ~15 minutes for 5 arms at 150s each.)

2. Wait for the script to finish. The script prints "Full logs:" at the end.

3. Find the most recent summary.tsv:
     ls -t experiments/Bandit_Wagon/results/signal_*/summary.tsv | head -1

4. Read summary.tsv. For any arm with status != ok, read its run.log to diagnose.

5. Apply decision rules from your system prompt.

6. If both width and depth beat anchor by >0.003 BPB, run the combo arm before deciding:
     env MODEL_DIM=432 NUM_FLAT_LAYERS=5 SEED=1337 \\
       NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 VOCAB_SIZE=1024 \\
       CRAWLER_MLP_MULT=4 NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=4 \\
       CRAWLER_CADENCE_EARLY=1 CRAWLER_CADENCE_MAIN=1 CRAWLER_CADENCE_LATE=1 \\
       XSA_LAST_N=2 ROPE_DIMS=16 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 \\
       TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \\
       ITERATIONS=20000 WARMUP_STEPS=20 GRAD_CLIP_NORM=0.3 \\
       MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=500 \\
       MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 TIED_EMBED_INIT_STD=0.005 \\
       MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5 MUON_WD=0.04 ADAM_WD=0.04 MUON_BETA2=0.95 \\
       MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \\
       SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15 \\
       EVAL_STRIDE=64 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=524288 \\
       DIAG_FIXED_CADENCE=0 DIAG_FAST_VAL=1 \\
       VE_ENABLED=0 TTT_BURST_ENABLED=0 DISTILL_ENABLED=0 POLAR_ENABLED=0 DTG_ENABLED=0 \\
       TS_PD_ENABLED=0 \\
       RUN_ID=BW-combo_432_5flat \\
       torchrun --standalone --nproc_per_node=1 train_gpt_h4_compiled.py \\
       2>&1 | tee experiments/Bandit_Wagon/results/combo_432_5flat.log

7. Write your full analysis + recommendation block to:
     experiments/Bandit_Wagon/results/SPARK_ANALYSIS.md
   (Create the results dir if needed.)

8. Print the recommendation block.
""",
    "analyze": """\
Analyze the most recent Bandit_Wagon spark signal results (do not re-run the script).

1. Find: ls -t experiments/Bandit_Wagon/results/signal_*/summary.tsv | head -1
2. Read summary.tsv.
3. For any arm with status != ok or an empty sliding_bpb, read its run.log.
4. Apply decision rules from your system prompt.
5. Write analysis + recommendation to experiments/Bandit_Wagon/results/SPARK_ANALYSIS.md
6. Print the recommendation block.
""",
    "iterate": """\
Run a targeted follow-up based on whichever Bandit_Wagon signal arms already completed.

1. Find and read the most recent summary.tsv (same as analyze task).
2. Identify gaps: missing arms, failed arms, or arms needing a combo follow-up.
3. Run only the missing/follow-up arms (do not re-run arms that already have results).
4. Analyze the combined results and write updated SPARK_ANALYSIS.md.
5. Print the recommendation block.
""",
}

# ── Main ──────────────────────────────────────────────────────────────────────

async def main(task: str) -> int:
    prompt = TASK_PROMPTS[task]

    print(f"[codex-spark] task={task}  cwd={REPO_ROOT}")
    print("[codex-spark] launching agent (claude-opus-4-6 + adaptive thinking)...\n")
    print("─" * 60)

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(REPO_ROOT),
                allowed_tools=["Bash", "Read", "Glob", "Grep", "Write"],
                permission_mode="bypassPermissions",
                model="claude-opus-4-6",
                thinking={"type": "adaptive"},
                system_prompt=SYSTEM_PROMPT,
                max_turns=60,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(message, ResultMessage):
                print(f"\n{'─' * 60}")
                print(f"[codex-spark] complete. stop_reason={message.stop_reason}")
                return 0

    except CLINotFoundError:
        print("ERROR: Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code")
        return 1
    except CLIConnectionError as e:
        print(f"ERROR: connection error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Codex Spark — Bandit_Wagon signal coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
tasks:
  signal   run 5-arm signal ablations then analyze (default)
  analyze  analyze existing results only (no new runs)
  iterate  run follow-up/combo arms based on existing results
        """,
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_PROMPTS.keys()),
        default="signal",
    )
    args = parser.parse_args()
    sys.exit(anyio.run(main, args.task))
