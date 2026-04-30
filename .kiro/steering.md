# Parameter Golf — Kiro Steering File
# Location: D:\SturdivantAI-Lab\Parameter-Golf\.kiro\steering.md
# Purpose: Intent and constraints layer. Complements parameter-golf.md (rules/constraints).
# This file captures the WHY. parameter-golf.md captures the HOW and the hard limits.

---

## Goal

Train a functional language model whose total artifact fits within 16 megabytes.
Official metric: bits-per-byte on FineWeb evaluation set.
Competition constraint: model must run within an 8×H100 inference window.
Research goal: produce a Pareto frontier chart (size vs. bits-per-byte) suitable for paper Figure 1.

---

## Optimization Space

Two independent axes:

**Axis 1 — Model weights (architecture search)**
- Parameter count vs. compression ratio trade-off
- BitNet ternary weights (-1, 0, 1) + EMA smoothing → maximise zlib level-9 ratio
- Target: 50–70M effective parameters within 16MB compressed artifact
- Primary lever: weight tying, depth recurrence, LoRA-TTT rank 4–8

**Axis 2 — Inference engine configuration (post-NVIDIA tower)**
- max_model_len × max_num_seqs sweep via vLLM harness
- SGLang RadixAttention crossover benchmark (compare KV cache strategies)
- Pareto output: throughput vs. quality across the sweep grid

---

## Active Experiments

### Sprint 001 — Baseline Cleanup (CURRENT)
- [ ] Remove `import numpy as np` from train_gpt.py — replace all numpy ops with pure Torch equivalents
- [ ] Verify Logic Drift returns clean after numpy removal
- [ ] Run first Golf Barrier check on baseline architecture
- [ ] Establish bits-per-byte baseline score on FineWeb

### Sprint 002 — Architecture Search
- [ ] Propose 3 candidate architectures (xhigh effort, first 5 minutes)
- [ ] Estimate compressed artifact size for each candidate
- [ ] Select architecture — weight tying + sliding window + 6-bit quantization as baseline

### Sprint 003+ — Compression Squeeze
- [ ] BitNet ternary weights via Triton kernel
- [ ] LoRA-TTT adaptation (rank 4, Q+V only)
- [ ] EMA weight smoothing to lower entropy before zlib compression
- [ ] Depth recurrence experiment — one transformer block, N passes

### Deferred — Requires NVIDIA Tower
- [ ] vLLM sidecar container (docker/Dockerfile.vllm)
- [ ] Docker Compose eval stack (ao-sandbox + vllm-server)
- [ ] SGLang RadixAttention vs vLLM paged attention benchmark
- [ ] Full max_model_len × max_num_seqs sweep grid (configs/sweep_configs.yaml)
- [ ] Pareto frontier notebook (notebooks/pareto_frontier.ipynb)

---

## Execution Environment

| Phase         | Compute                        | Status          |
|---------------|--------------------------------|-----------------|
| Architecture search | Lenovo Windows (CPU/NPU) | Available now   |
| Training runs | SageMaker spot ml.g5.2xlarge  | Available now   |
| Full eval sweep | NVIDIA Modular Tower         | Pending arrival |
| Inference benchmark | NVIDIA Tower + Docker    | Pending arrival |

**Current GPU path:** SageMaker spot ml.g5.2xlarge (~$0.45–0.60/hr)
**Future GPU path:** NVIDIA Modular Tower (local, zero cloud cost)
**Fallback:** Lenovo NPU for small-scale architecture validation only (no CUDA, no Triton)

---

## Repo Structure

```
D:\SturdivantAI-Lab\Parameter-Golf\
├── .kiro\
│   └── skills\
│       └── parameter-golf.md    ← Hard constraints, whitelist, watchdog rules
│   └── steering.md              ← This file — intent and experiment queue
├── src\
│   ├── sweep.py                 ← Phase 2 vLLM sweep harness (build when tower arrives)
│   ├── benchmark.py             ← Phase 3 Pareto output
│   └── eval\
│       └── fineweb_eval.py      ← Bits-per-byte eval loop
├── configs\
│   └── sweep_configs.yaml       ← max_model_len × max_num_seqs grid
├── docker\
│   └── Dockerfile.vllm          ← vLLM container spec (build when tower arrives)
├── notebooks\
│   └── pareto_frontier.ipynb    ← Visualisation (build when results exist)
├── results\
│   └── .gitkeep
├── data\                        ← Baseline repo data folder
├── records\                     ← Baseline repo records folder
├── hero.md                      ← Master manifest (watchdog.py reads this)
├── watchdog.py                  ← Building Inspector
├── golf_sprint_log.md           ← Sprint audit log
├── train_gpt.py                 ← Competition artifact — ALL training code here
├── requirements.txt             ← Baseline repo deps (Logic Drift monitors this)
└── README.md
```

---

## Known Baseline Violations (Acknowledged)

These exist in the official baseline repo and are not sprint code violations.
They must be resolved in the sprint version of train_gpt.py before submission.

| Library         | Location          | Category               | Sprint Fix                            |
|-----------------|-------------------|------------------------|---------------------------------------|
| numpy           | train_gpt.py L1   | Pure Torch Violation   | Replace all np.* ops with torch.*     |
| tqdm            | requirements.txt  | Visualisation/Logging  | Remove from sprint training loop      |
| datasets        | requirements.txt  | Data Sneaking          | Replace with approved data loader     |
| huggingface_hub | requirements.txt  | Data Sneaking          | Remove — use direct data path         |

---

## Do Not

- Hard-floor `max_model_len` below 2048 without bits-per-byte validation
- Mix LLM-as-Judge quality metrics with throughput metrics in the same Pareto chart
- Apply BitNet post-hoc to a float-trained model — use BitNet from initialisation only
- Add non-whitelisted libraries without Logic Drift flag + Architect review
- Use lookahead attention under any framing — architectural violation
- Run `python watchdog.py --regenerate-manifest` without explicit Kiro signoff
- Restart training after a Golf Barrier KILL without architectural review

---

## Paper Pipeline Hook

The Pareto frontier chart (size MB vs. bits-per-byte on FineWeb) is the target Figure 1
for the Q3 2026 research paper submission. Every sprint result that passes the Golf Barrier
should be logged to golf_sprint_log.md with its bits-per-byte score — this is the raw data
for the figure.

Target venue: arXiv preprint → IEEE Xplore / NeurIPS workshop submission.
