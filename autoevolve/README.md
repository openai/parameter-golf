# AutoEvolve for Parameter Golf

AutoEvolve is a competition-faithful autonomous research loop for OpenAI's Parameter Golf challenge. It does not try to brute-force hyperparameters or wander through arbitrary code. It starts from the current best local record, mutates only `autoevolve/train_gpt.py`, runs bounded experiments, promotes real improvements, preserves strategically useful near misses, and builds structured memory so the next proposal learns from both wins and failures.

The design goal is simple: spend scarce GPU hours on experiments that are interpretable, recoverable, and increasingly informed by prior evidence.

## Objective

- Minimize `val_bpb` under the hard `16,000,000`-byte artifact cap.
- Search cheaply but meaningfully on `1xH100` in long-horizon proxy mode.
- Validate only the strongest proxy wins in the real `8xH100 / 600s` setting.
- Keep the mutation surface narrow and the control plane stable.

## How It Plays the Competition

AutoEvolve is intentionally aligned with the actual challenge rather than an arbitrary local proxy.

- `1xH100` runs are treated as **proxy research mode**, not final scoring.
- Proxy mode uses a long training cap (`4800s`) so local search better reflects the real `8xH100 / 600s` competition regime.
- `8xH100` runs are treated as **final validation mode** and stay faithful to the official budget.
- The artifact cap and observable evaluation contract remain fixed in both modes.

This is a two-stage research strategy:

1. Discover plausible ideas under a longer `1xH100` proxy.
2. Promote only the strongest ideas to real `8xH100` final checks.

## Search Loop

Each iteration follows the same disciplined sequence:

1. Benchmark the current incumbent in the active mode before mutating anything.
2. Build a prompt from the current code, the structured memory dossier, the active search state, and domain knowledge about the competition.
3. Ask the model for one high-conviction proposal expressed as exact search/replace edits.
4. Validate the proposal for unique matches, syntax, required competition outputs, and other hard invariants.
5. Run the experiment and capture partial or final telemetry.
6. Classify the outcome as `keep`, `near_miss`, `discard`, `crash`, `invalid`, `parse_error`, or `over_size`.
7. Persist the code state, logs, structured results, and git metadata so the next iteration starts from a truthful memory.

## Search State: Incumbent and Frontier

AutoEvolve is not pure greedy hill-climbing.

- The **incumbent** is the best validated script for the current run mode.
- A **near miss** can open a short-lived **frontier** branch if it is close enough to the incumbent to justify a follow-on.
- Frontier continuation is bounded and must improve relative to the prior frontier point, not merely remain "interesting."
- Only a true measured improvement replaces the incumbent.

This keeps the search stable while still allowing short escapes from local optima.

## Memory Model

The system does not dump raw history back into the prompt and hope the model figures it out.

- Experiment history is normalized into `results.tsv`.
- A deterministic `memory_dossier.md` is generated from committed state.
- Infrastructure failures are separated from research evidence.
- Research evidence is summarized by proposal family, status, timeout stage, and timing telemetry.
- A repeat-family guard prevents the model from repeatedly proposing the same failed family unless it gives a concrete exemption and mechanism.

The result is a prompt that carries usable scientific memory instead of noisy transcripts.

## Robustness and Failure Handling

The control plane is designed for overnight unattended runs.

- Dry runs are non-persistent and never contaminate experiment memory.
- Timeouts preserve partial output and are classified by stage (`launch`, `compile`, `train`, `eval`, `post_export`, `complete`).
- Failed or non-improving candidates automatically roll back to the active base state.
- Real outcomes are persisted to git after result logging, with commit SHA written back into the ledger.
- Per-experiment logs are mirrored into `autoevolve/logs/` for later inspection.

This is meant to fail loudly, recover cleanly, and leave behind enough evidence for the next step.

## Project Layout

- `autoevolve/evolve.py`: main autonomous research loop
- `autoevolve/train_gpt.py`: the only script the model is allowed to mutate
- `autoevolve/best_train_gpt.py`: incumbent best script
- `autoevolve/frontier_train_gpt.py`: active exploratory frontier, if any
- `autoevolve/results.tsv`: normalized experiment ledger
- `autoevolve/prompts/`: agent prompt, program knowledge, and generated memory dossier
- `autoevolve/monitor.py`: operator dashboard
- `autoevolve/runpod_launch.sh`: RunPod launcher with tmux and conda integration

## Operating Model

For an actual run:

```bash
bash autoevolve/runpod_launch.sh --nproc 1 --model gpt-5.4
```

For monitoring:

```bash
conda run -n parameter-golf python autoevolve/monitor.py --summary
conda run -n parameter-golf python autoevolve/monitor.py --tail
```

Because the current best baseline includes legal TTT, proxy-mode iterations are intentionally expensive. The system is optimized for high-quality overnight research, not for maximizing raw iteration count.

## Research Philosophy

The core bet behind AutoEvolve is that the bottleneck in this challenge is not writing one more training script by hand. The bottleneck is running a careful, memory-bearing research process under severe artifact and time constraints.

AutoEvolve is therefore designed less like an optimizer sweep and more like a small autonomous scientist:

- narrow editable surface
- explicit local benchmark
- structured memory
- bounded branching
- competition-faithful evaluation
- durable experiment state

That is the standard it is trying to meet each night it runs.
