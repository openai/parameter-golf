# Parameter Golf: Full Execution Plan for Claude Code

## MASTER TECHNIQUE CATALOG

Each technique is tagged: ✅ proven in a PR, 🔬 proven in literature but untried here, 🧪 speculative.

### Layer 0: Base Infrastructure (PR sources to clone)
```
SOURCES:
  #569  github.com/gowtham0992    — VRL + LeakyReLU² + Full GPTQ + QAT alignment + 2% pruning
  #505  github.com/JoeProAI       — GEPA arch (Star-ReLU + U-Net Skip Gates + XSA4 + VE128)
  #576  github.com/cmcdnd         — 33.6M int5 GPTQ + legal TTT + post-TTT temp calibration
  #414  github.com/signalrush     — Official SOTA: EMA + GPTQ-lite + warmdown3500 + QAT@0.15
  #727  (n-gram cache submitter)  — Multi-order n-gram backoff + entropy-adaptive alpha
  #738  (kNN-LM submitter)        — Hidden-state kNN-LM
  #265  github.com/unnir          — Efficient Partial XSA (GQA-aware, zero-alloc)
```

### Layer 1: Architecture Techniques
```
ID   TECHNIQUE                        STATUS   SOURCE    EST. BPB   INDEPENDENT?
A1   11L 512-dim 8H/4KV GQA           ✅       #414      baseline   n/a
A2   3× MLP (1536), relu²             ✅       #414      baseline   n/a
A3   U-Net skip connections            ✅       #414      baseline   n/a
A4   Tied embeddings + softcap=30      ✅       #414      baseline   n/a
A5   SmearGate + BigramHash(2048)      ✅       #414      baseline   n/a
A6   Efficient Partial XSA (last 4)    ✅       #265      -0.002     yes
A7   Partial RoPE (16/64) + LN Scale   ✅       #315      -0.002     yes
A8   Shared Value Embedding VE128      ✅       #374      -0.001     yes
A9   LeakyReLU(0.5)² (replaces relu²)  ✅       #535      -0.0015    yes (vs A2)
A10  Value Residual Learning (VRL)     ✅       #569      -0.005     yes
A11  GEPA (Star-ReLU + gated skips)    ✅       #505      -0.005     yes (vs A2,A3)
A12  XSA-all (11 layers not just 4)    ✅       #545      -0.002     yes (vs A6)
A13  MLP 3.5× (1792) w/ 33.6M params  ✅       #545      -0.003     yes (vs A2)
A14  Differential Attention            🔬       arxiv     -0.005-15  yes
A15  HybridNorm (mixed Pre/Post)       🔬       arxiv     -0.002-6   yes
A16  PoPE (Polar Position Embedding)   🔬       arxiv     -0.002-5   yes (vs A7)
A17  WaveletGPT (Haar on half dims)    🔬       arxiv     -0.003-10  yes
A18  PolyCom activations               🔬       arxiv     -0.002-6   yes (vs A2,A9)
A19  VGA (Value-Gated Attention)       🔬       arxiv     -0.002-5   yes
```

### Layer 2: Training Techniques
```
ID   TECHNIQUE                        STATUS   SOURCE    EST. BPB   INDEPENDENT?
T1   Muon (matrices) lr=0.025 m=0.99  ✅       #414      baseline   n/a
T2   AdamW (embeds) lr=0.035           ✅       #414      baseline   n/a
T3   FlashAttention 3                  ✅       #265      baseline   n/a
T4   Warmdown 3500 iters               ✅       #414      -0.0002    n/a
T5   OrthoInit + muP-scaled outputs    ✅       #414      baseline   n/a
T6   EMA decay=0.997 every step        ❌       #414      -0.0006    DISABLED: EMA-averaged weights quantize worse (same issue as SWA)
T7   Tight SWA every 50 (scale<0.2)    ❌       #374      -0.0003    DISABLED: SWA-averaged weights quantize worse (0.33→0.46 BPB after int6 roundtrip)
T8   Late QAT STE int6 (scale<0.15)    ✅       #414      -0.0001    n/a
T9   Gradient clip 0.3                 ✅       #414      baseline   n/a
T10  WD=0.04 both optimizers           ✅       #414      baseline   n/a
T11  Mousse optimizer (curvature Muon) 🔬       arxiv     -0.003-8   replaces T1
T12  Turbo-Muon (precon Newton-Schulz) 🔬       arxiv     -0.002-5   replaces T1
T13  Predictive Batch Scheduling       🔬       arxiv     -0.002-5   yes
T14  Late-Stage SAM (last 5-10%)       🔬       arxiv     -0.002-5   yes
T15  Multi-Token Prediction heads      🔬       meta      -0.002-5   yes (free artifact)
T16  2:4 Structured Sparsity (relu²)   🔬       arxiv     -0.003-8   yes (systems-only)
```

### Layer 3: Quantization & Compression
```
ID   TECHNIQUE                        STATUS   SOURCE    EST. BPB   INDEPENDENT?
Q1   Int6 per-row (MLP+attn)          ✅       #414      baseline   n/a
Q2   Int8 per-row (embeddings)         ✅       #414      baseline   n/a
Q3   GPTQ-lite (5 clip percentiles)    ✅       #414      -0.0006    n/a
Q4   zstd level 22                     ✅       #414      baseline   n/a
Q5   Full GPTQ (calibration in budget) ✅       #535      -0.002     replaces Q3
Q6   Int5 + larger model (33.6M)       ✅       #545      -0.003     replaces Q1
Q7   QAT-export alignment              ✅       #569      -0.001     yes
Q8   2% magnitude pruning              ✅       #569      -0.001     yes
Q9   OptRot pre-quant rotation         🔬       arxiv     -0.002-5   yes
Q10  Entropy-coded (Huffman/codebook)  ✅ partial #532    -0.003-8   replaces Q4
Q11  Lattice VQ (Leech lattice)        🔬       arxiv     -0.005-15  replaces Q1-Q6
```

### Layer 4: Eval-Time Techniques
```
ID   TECHNIQUE                        STATUS   SOURCE    EST. BPB   INDEPENDENT?
E1   Sliding window stride=64          ✅       #414      baseline   n/a
E2   Legal score-first TTT (AdamW)     ✅       #473      -0.005-20  yes
E3   Post-TTT temp calibration T=0.98  ✅       #576      -0.003     stacks w/ E2
E4   qTTT (query-only, cache K/V)      🔬       arxiv     -0.003-10  replaces E2
E5   LaCT (large chunk Muon TTT)       🔬       arxiv     -0.005-15  replaces E2
E6   Multi-order n-gram backoff (2-7)  ✅       #727      -0.15+     yes
E7   Entropy-adaptive alpha            ✅       #727      stacked    part of E6
E8   kNN-LM on hidden states           ✅       #738      -0.007     stacks w/ E6
E9   Multi-Pass Streaming TTT          ⚠️       #573      -0.070     disputed legality
```

---

## COMPATIBILITY MATRIX

Mutually exclusive sets (pick one from each):

```
ACTIVATION:   A2 (relu²) | A9 (LeakyReLU²) | A11/GEPA (Star-ReLU) | A18 (PolyCom)
SKIP CONN:    A3 (U-Net) | A11/GEPA (gated skips)
XSA SCOPE:    A6 (last 4) | A12 (all 11)
MLP WIDTH:    A2 (3×/1536) | A13 (3.5×/1792)
POSITION:     A7 (Partial RoPE) | A16 (PoPE)
OPTIMIZER:    T1 (Muon) | T11 (Mousse) | T12 (Turbo-Muon)
QUANTIZE:     Q1+Q3 (int6+GPTQ-lite) | Q5 (Full GPTQ) | Q6 (int5+bigger model) | Q11 (Lattice VQ)
COMPRESS:     Q4 (zstd-22) | Q10 (entropy-coded)
TTT:          E2 (AdamW TTT) | E4 (qTTT) | E5 (LaCT)
```

Freely stackable (independent of all others):
```
A7/A16, A8, A10, A14, A15, A17, A19
T6, T7, T13, T14, T15, T16
Q7, Q8, Q9
E1, E3, E6, E7, E8
```

---

## BUILD ORDER

### Phase 0: Environment Setup
```
ACTIONS:
  1. Clone openai/parameter-golf main branch
  2. Clone PR #569 branch (VRL + LeakyReLU² + Full GPTQ) — this is our BASE
  3. Clone PR #505 branch (GEPA) — reference for gated skip / Star-ReLU
  4. Clone PR #576 branch (TTT + temp calibration)
  5. Clone PR #727 branch (n-gram cache)
  6. Clone PR #738 branch (kNN-LM)
  7. Download FineWeb dataset: python3 data/cached_challenge_fineweb.py --variant sp1024
  8. Set up ablation infrastructure (see ABLATION FRAMEWORK below)
  9. Verify #569 base reproduces 1.1175 BPB on 1×H100 (scaled steps)

EXPECTED TIME: ~30 min setup, ~15 min verification run
```

### Phase 1: Establish Maximum Neural Base (no TTT, no eval-augment)
**Goal**: Best possible pre-quantization model quality in 600s training.

```
STEP 1.1 — BASELINE VERIFICATION [checkpoint: BASE]
  Action: Run #569's train_gpt.py unmodified on 1×H100 with reduced iterations
  Command: RUN_ID=base_verify SEED=1337 MAX_WALLCLOCK_SECONDS=120 \
           torchrun --standalone --nproc_per_node=1 train_gpt.py
  Record: val_loss, val_bpb, artifact_size
  Expected: Should roughly track toward 1.1175 (won't reach it in 120s but trend confirms)

STEP 1.2 — ADD OPTROT PRE-QUANTIZATION [checkpoint: BASE+Q9]
  Action: Implement OptRot (arXiv:2512.24124) in the quantization pipeline:
    - After training, before quantization, compute Hadamard rotation matrix H
    - For each weight matrix W, compute W' = W @ H
    - Quantize W' instead of W
    - At inference, absorb H^T into the next layer's input projection
    - H is deterministic (no storage cost): use scipy.linalg.hadamard or recursive construction
  Code location: Modify the quantization function in train_gpt.py (the int6_quantize path)
  Test: Compare quantization MSE before/after OptRot on the base checkpoint
  Record: quant_mse_before, quant_mse_after, val_bpb_before, val_bpb_after
  Expected: 30-50% MSE reduction, 0.002-0.005 BPB improvement

STEP 1.3 — ADD VALUE RESIDUAL LEARNING TO GEPA BASE [checkpoint: GEPA+A10]
  Action: This is the key unstacked merge.
    From #505 (GEPA): Extract Star-ReLU activation, gated skip connections, U-Net skip gates
    From #569 (VRL): Extract value residual connections (first-layer V projected to deeper layers)
    Merge: GEPA arch + VRL. These modify different components:
      - GEPA changes: activation function (Star-ReLU), skip connection gating
      - VRL changes: value projection residuals across layers
    No conflict. Direct merge.
  Code: In the model class:
    - Replace relu_squared with star_relu (from #505)
    - Replace standard skip connections with gated skips (from #505)
    - Add VRL residual path from layer 0 value proj to layers 1-10 (from #569)
  Test: Train for 120s on 1×H100, compare val_loss vs BASE and vs each modification alone
  Record: val_loss, val_bpb
  Expected: Better than either alone. ~1.110-1.115 target (at full 600s on 8×H100)

STEP 1.4 — LEAKYRELU² vs STAR-RELU ABLATION [checkpoint: best_activation]
  Action: LeakyReLU(0.5)² and Star-ReLU are both proven but MUTUALLY EXCLUSIVE.
    Run 3 short training runs (120s each, 1×H100):
      a) GEPA+VRL with Star-ReLU (as in 1.3)
      b) Standard arch+VRL with LeakyReLU(0.5)²
      c) GEPA gated skips + VRL + LeakyReLU(0.5)² (hybrid — untested!)
  Record: val_loss for each at step 1000
  Decision: Pick whichever is lowest. The hybrid (c) is the interesting bet.

STEP 1.5 — ADD HYBRIDNORM [checkpoint: +A15]
  Action: Replace all Pre-Norm with mixed Pre/Post-Norm per arXiv:2503.04598
    - Layers 0-5: Pre-Norm (standard)
    - Layers 6-10: Post-Norm (better gradient flow in deep layers)
    - Or: alternating Pre/Post per the paper's recommendation
  Code: In the TransformerBlock, add a flag per layer for norm placement
  Test: 120s training run
  Expected: 0.002-0.006 improvement. Very low risk.

STEP 1.6 — SCALE MLP TO 3.5× WITH INT5 TARGET [checkpoint: +A13]
  Action: If VRAM allows (int5 means more params fit in 16MB):
    - Increase MLP hidden from 1536 to 1792 (3.5× expansion)
    - Switch quantization target from int6 to int5
    - This adds ~6M parameters but int5 saves enough bits to fit in 16MB
  Code: Change MLP_HIDDEN constant, change quantize bit-width
  Test: 120s run, compare val_loss
  Trade-off: More params × lower precision. #545/#576 proved this works.

STEP 1.7 — ADD FULL GPTQ + QAT-EXPORT ALIGNMENT + PRUNING [checkpoint: +Q5+Q7+Q8]
  Action: From #569:
    - Replace GPTQ-lite (clip search) with Full GPTQ (Hessian-based, within 600s budget)
    - Add QAT-export alignment (match quantization noise during training to export)
    - Add 2% magnitude pruning before compression
  Code: Substantial — GPTQ requires calibration data forward pass. Budget ~15s of training time.
  Test: Compare artifact size and val_bpb vs GPTQ-lite
  Expected: -0.002 from better quantization

STEP 1.8 — OPTIMIZER EXPERIMENT: MOUSSE [checkpoint: +T11 or revert]
  Action: Implement Mousse (arXiv:2603.09697) as drop-in Muon replacement
    - Shampoo preconditioning before Newton-Schulz orthogonalization
    - ~12% more effective training at 3% overhead
  Code: Modify the Muon optimizer class
  Test: Run full 600s training on 1×H100, compare final val_loss vs Muon
  Decision: Keep if improvement > 0.002, revert if not
  Risk: Moderate — optimizer changes can interact badly with existing LR schedule

STEP 1.9 — ADD MULTI-TOKEN PREDICTION HEADS [checkpoint: +T15]
  Action: Add 2 auxiliary prediction heads during training only:
    - head_t2 predicts token at position t+2
    - head_t3 predicts token at position t+3
    - Each head: single linear layer (512 → vocab_size)
    - Loss = CE_main + 0.3*CE_t2 + 0.1*CE_t3
    - Strip heads before quantization (zero artifact cost)
  Code: In the model forward(), add extra linear projections from final hidden state
  Test: 120s run, compare val_loss of the MAIN head only
  Expected: Better representations → lower main loss. Free at eval time.

NEURAL BASE CHECKPOINT: After steps 1.1-1.9, we should have the strongest pre-TTT model.
Target: ~1.105-1.115 BPB without any eval-time augmentation.
```

### Phase 2: Eval-Time Augmentation Stack
**Goal**: Maximize BPB improvement during the 10-minute eval budget.

```
STEP 2.1 — ADD LEGAL SCORE-FIRST TTT [checkpoint: +E2]
  Action: From #576/#473, implement backward-looking TTT:
    - During eval, after scoring each chunk of tokens, perform gradient updates
    - Use LoRA adapters on Q/V projections (rank 4-8)
    - AdamW optimizer, lr=1e-4, 3 epochs per chunk
    - Score tokens FIRST, then adapt (legal: only backward-looking)
  Code: Add TTT loop in the evaluation function
  Test: Run eval on validation set with and without TTT
  Expected: -0.005 to -0.020 BPB improvement

STEP 2.2 — ADD POST-TTT TEMPERATURE CALIBRATION [checkpoint: +E3]
  Action: From #576 (novel technique):
    - TTT causes overconfidence (distribution sharpens)
    - Apply T=0.98 to logits after TTT adaptation
    - Calibrate T on first 1000 tokens of eval
  Code: Add temperature scaling in the eval scoring loop
  Expected: -0.003 BPB on top of TTT

STEP 2.3 — IMPLEMENT qTTT (QUERY-ONLY TTT) [checkpoint: +E4]
  Action: Replace standard TTT with query-only variant (arXiv:2512.13898):
    - On first pass through a document: cache ALL K and V tensors
    - On TTT gradient steps: only update Q projection weights (via LoRA)
    - K/V are frozen and reused → no re-materialization cost
    - Per-epoch time drops from ~15-18s to ~4-6s
    - This enables 3× more TTT epochs in the same eval budget
  Code: Modify TTT loop to freeze K/V computation, only backprop through Q
  Test: Compare BPB and wall-clock time vs standard TTT
  Expected: Same or better BPB in less time, or significantly better BPB in same time

STEP 2.4 — ADD N-GRAM CACHE [checkpoint: +E6+E7]
  Action: From #727, implement multi-order n-gram backoff:
    - Maintain hash tables for n-grams of orders 2 through 7
    - As eval processes tokens, record (context → next_token) counts
    - For each prediction, interpolate LM probability with n-gram probability
    - Use entropy-adaptive alpha: when LM is uncertain (high entropy), weight n-gram more
    - Alpha interpolation: P_final = (1-α)*P_lm + α*P_ngram where α = f(entropy(P_lm))
  Code: Add NgramCache class with update() and predict() methods
  Important: Only use tokens ALREADY SCORED. No lookahead. This is legal.
  Test: Run eval with cache, measure BPB
  Expected: Massive improvement. 1.1271 → 0.9674 demonstrated. On our better base: ~0.95

STEP 2.5 — ADD kNN-LM [checkpoint: +E8]
  Action: From #738, add hidden-state nearest-neighbor lookup:
    - During eval, store (hidden_state, next_token) pairs in a buffer
    - For each new token, find k nearest hidden states via L2 distance
    - Weight their next-token distributions by distance
    - Interpolate with LM + n-gram distribution
  Code: Add KnnLM class with store() and retrieve() methods
  Test: Measure BPB improvement over n-gram cache alone
  Expected: Additional -0.007 BPB

EVAL STACK CHECKPOINT: After steps 2.1-2.5 on top of Phase 1 neural base.
Target: ~0.93-0.96 BPB
```

### Phase 3: Experimental / High-Risk Techniques
**Goal**: Try techniques with uncertain payoff. Run as parallel experiments.

```
STEP 3.1 — DIFFERENTIAL ATTENTION [PARALLEL EXPERIMENT]
  Action: Replace standard softmax attention with DIFF attention:
    - Split Q,K into two halves: Q1,Q2,K1,K2
    - Compute A = softmax(Q1@K1^T) - λ*softmax(Q2@K2^T)
    - λ is a learnable per-head scalar
    - Net parameter cost: roughly neutral (two half-width QK ≈ one full-width)
  Risk: May interact poorly with XSA, RoPE, GQA. High integration complexity.
  Test: Train from scratch for 600s, compare val_loss to Phase 1 base

STEP 3.2 — WAVELETGPT [PARALLEL EXPERIMENT]
  Action: Apply Haar wavelet structure to half of embedding dimensions:
    - Split 512-dim embeddings into 256 "standard" + 256 "wavelet" dims
    - Wavelet dims use multi-scale decomposition (sum/difference at 2,4,8,16 token scales)
    - Zero additional parameters
    - 40-60% faster convergence claimed
  Risk: May not help at this model scale. Very low cost to try.
  Test: 120s training run, compare val_loss at same step count

STEP 3.3 — STRUCTURED 2:4 SPARSITY FOR TRAINING [PARALLEL EXPERIMENT]
  Action: relu² already produces 84-98% sparse activations. Enforce NVIDIA 2:4 pattern:
    - After relu², apply 2:4 mask (keep 2 largest values per group of 4)
    - Use NVIDIA's sparse tensor cores for the subsequent matmul
    - 2× speedup on sparse matmuls → 15-20% more training steps in 600s
    - Systems-only change → significance threshold waived for record
  Risk: Sparse kernels may not be faster in practice at this dimension.
  Test: Measure wall-clock step time with and without sparse enforcement

STEP 3.4 — ENTROPY-CODED COMPRESSION [PARALLEL EXPERIMENT]
  Action: Replace zstd-22 with learned entropy coding:
    - After int6 quantization, compute per-layer symbol frequency tables
    - Build per-layer Huffman trees (or arithmetic coder)
    - Encode weights using layer-specific optimal codes
    - Decoder: ~500 bytes of Python code in artifact
  Size target: 21% reduction demonstrated in #532 → frees ~1.88MB for more parameters
  Risk: Code size counts toward 16MB. Decoder must be small.
  Test: Compare compressed size vs zstd-22 on same quantized weights
```

---

## ABLATION FRAMEWORK

Create this infrastructure FIRST, before any experiments.

### File: `ablation.py`
```python
#!/usr/bin/env python3
"""
Parameter Golf Ablation Framework
Tracks every experiment, records results, computes deltas.
"""
import json, os, time, hashlib, subprocess, sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List

ABLATION_DIR = Path("./ablation_results")
ABLATION_DIR.mkdir(exist_ok=True)

@dataclass
class AblationResult:
    run_id: str
    techniques: List[str]         # List of technique IDs enabled (e.g., ["A1","A9","A10","Q5"])
    base_checkpoint: str          # Which checkpoint this builds on
    seed: int
    val_loss: float
    val_bpb: float
    artifact_size_bytes: int
    training_steps: int
    wall_clock_seconds: float
    gpu_config: str               # e.g., "1xH100" or "8xH100"
    notes: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def delta_vs(self, other: 'AblationResult') -> dict:
        return {
            "bpb_delta": self.val_bpb - other.val_bpb,
            "loss_delta": self.val_loss - other.val_loss,
            "size_delta": self.artifact_size_bytes - other.artifact_size_bytes,
            "added_techniques": sorted(set(self.techniques) - set(other.techniques)),
            "removed_techniques": sorted(set(other.techniques) - set(self.techniques)),
        }

def save_result(result: AblationResult):
    path = ABLATION_DIR / f"{result.run_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[ABLATION] Saved: {path}")
    return path

def load_result(run_id: str) -> AblationResult:
    path = ABLATION_DIR / f"{run_id}.json"
    with open(path) as f:
        return AblationResult(**json.load(f))

def load_all() -> List[AblationResult]:
    results = []
    for p in sorted(ABLATION_DIR.glob("*.json")):
        with open(p) as f:
            results.append(AblationResult(**json.load(f)))
    return results

def print_leaderboard():
    results = load_all()
    results.sort(key=lambda r: r.val_bpb)
    print(f"\n{'='*80}")
    print(f"{'RUN_ID':<35} {'BPB':>8} {'LOSS':>8} {'SIZE':>10} {'TECHNIQUES'}")
    print(f"{'='*80}")
    for r in results:
        techs = ",".join(r.techniques[:8])
        if len(r.techniques) > 8:
            techs += f"...+{len(r.techniques)-8}"
        print(f"{r.run_id:<35} {r.val_bpb:>8.4f} {r.val_loss:>8.4f} {r.artifact_size_bytes:>10,} {techs}")
    print()

def print_ablation_table():
    """Show impact of each technique by comparing runs that differ by exactly one technique."""
    results = load_all()
    print(f"\n{'='*80}")
    print("SINGLE-TECHNIQUE ABLATIONS (pairs differing by exactly 1 technique)")
    print(f"{'='*80}")
    pairs = []
    for i, a in enumerate(results):
        for b in results[i+1:]:
            delta = a.delta_vs(b)
            if len(delta["added_techniques"]) + len(delta["removed_techniques"]) == 1:
                pairs.append((a, b, delta))
    pairs.sort(key=lambda x: abs(x[2]["bpb_delta"]), reverse=True)
    for a, b, d in pairs:
        direction = "+" if d["added_techniques"] else "-"
        tech = (d["added_techniques"] or d["removed_techniques"])[0]
        sign = "-" if d["bpb_delta"] < 0 else "+"
        print(f"  {direction}{tech:<12} {sign}{abs(d['bpb_delta']):.4f} BPB  "
              f"({a.run_id} vs {b.run_id})")
    if not pairs:
        print("  No single-technique pairs found yet. Run more ablations!")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "leaderboard":
        print_leaderboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "ablations":
        print_ablation_table()
    else:
        print("Usage: python ablation.py [leaderboard|ablations]")
```

### File: `run_ablation.sh`
```bash
#!/bin/bash
# Run a single ablation experiment and record the result.
# Usage: ./run_ablation.sh <run_id> <techniques_csv> <base_checkpoint> [extra_env_vars...]
#
# Example:
#   ./run_ablation.sh base_verify "A1,A2,A3,A4,A5" "none" SEED=1337
#   ./run_ablation.sh base_plus_optrot "A1,A2,A3,A4,A5,Q9" "base_verify" SEED=1337

set -euo pipefail

RUN_ID="${1:?Usage: run_ablation.sh <run_id> <techniques_csv> <base_checkpoint> [env...]}"
TECHNIQUES="${2:?}"
BASE_CHECKPOINT="${3:?}"
shift 3

# Apply any extra env vars
for var in "$@"; do export "$var"; done

export RUN_ID
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"

echo "[ABLATION] Starting: $RUN_ID"
echo "[ABLATION] Techniques: $TECHNIQUES"
echo "[ABLATION] Base: $BASE_CHECKPOINT"
echo "[ABLATION] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

START_TIME=$(date +%s)

# Run training — capture output
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py 2>&1 | tee "ablation_results/${RUN_ID}.log"

END_TIME=$(date +%s)
WALL=$((END_TIME - START_TIME))

# Parse results from log
VAL_LOSS=$(grep "val_loss" "ablation_results/${RUN_ID}.log" | tail -1 | grep -oP 'val_loss[= ]+\K[0-9.]+')
VAL_BPB=$(grep "val_bpb" "ablation_results/${RUN_ID}.log" | tail -1 | grep -oP 'val_bpb[= ]+\K[0-9.]+')
ARTIFACT_SIZE=$(grep -i "artifact\|compressed\|final.*size\|zlib\|zstd" "ablation_results/${RUN_ID}.log" | tail -1 | grep -oP '[0-9]+(?= bytes)' || echo "0")
STEPS=$(grep -oP 'step[= ]+\K[0-9]+' "ablation_results/${RUN_ID}.log" | tail -1 || echo "0")
GPU_CONFIG="${NPROC:-1}x$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

# Record result
python3 -c "
from ablation import AblationResult, save_result
r = AblationResult(
    run_id='$RUN_ID',
    techniques='$TECHNIQUES'.split(','),
    base_checkpoint='$BASE_CHECKPOINT',
    seed=${SEED:-1337},
    val_loss=${VAL_LOSS:-0.0},
    val_bpb=${VAL_BPB:-0.0},
    artifact_size_bytes=${ARTIFACT_SIZE:-0},
    training_steps=${STEPS:-0},
    wall_clock_seconds=$WALL,
    gpu_config='$GPU_CONFIG',
    notes='${NOTES:-}'
)
save_result(r)
"

echo "[ABLATION] Done: $RUN_ID — BPB=$VAL_BPB, Loss=$VAL_LOSS, ${WALL}s"
python3 ablation.py leaderboard
```

### File: `run_all_ablations.sh`
```bash
#!/bin/bash
# Master ablation sequence. Run on 1×H100 for fast iteration.
# Each step adds exactly one technique to isolate its contribution.
set -euo pipefail

export NPROC=1
export MAX_WALLCLOCK_SECONDS=120
export SEED=1337

echo "=========================================="
echo "PARAMETER GOLF ABLATION SUITE"
echo "=========================================="

# Phase 1: Neural base ablations
# Each run adds exactly ONE new technique to measure its isolated contribution.

# 1. Pure baseline (#569 as-is: VRL + LeakyReLU² + Full GPTQ)
./run_ablation.sh \
  "p1_base_569" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q4" \
  "none" \
  NOTES="PR #569 base unmodified"

# 2. +OptRot (Q9)
./run_ablation.sh \
  "p1_base_569_optrot" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569" \
  NOTES="+OptRot pre-quantization rotation" ENABLE_OPTROT=1

# 3. +HybridNorm (A15) — on top of OptRot
./run_ablation.sh \
  "p1_optrot_hybridnorm" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A15,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+HybridNorm mixed Pre/Post" ENABLE_HYBRIDNORM=1

# 4. Activation shootout: LeakyReLU² vs Star-ReLU vs hybrid (on same base)
./run_ablation.sh \
  "p1_starelu_test" \
  "A1,A3,A4,A5,A6,A7,A8,A10,A11,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="Star-ReLU (GEPA) instead of LeakyReLU²" ACTIVATION=star_relu

./run_ablation.sh \
  "p1_gepa_gated_skips" \
  "A1,A4,A5,A6,A7,A8,A10,A11,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="GEPA gated skips + Star-ReLU (full GEPA)" ACTIVATION=star_relu SKIP_TYPE=gated

# 5. +XSA-all (A12) — expand XSA from last-4 to all-11 layers
./run_ablation.sh \
  "p1_xsa_all" \
  "A1,A2,A3,A4,A5,A7,A8,A9,A10,A12,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+XSA on all 11 layers" XSA_LAST_N=11

# 6. +MLP 3.5× with int5 target (A13+Q6)
./run_ablation.sh \
  "p1_mlp35_int5" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A13,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q6,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+MLP 3.5× (1792) with int5 quant" MLP_HIDDEN=1792 QUANT_BITS=5

# 7. +Multi-Token Prediction heads (T15) — on best base so far
./run_ablation.sh \
  "p1_mtp_heads" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T15,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+Multi-token prediction (2 aux heads)" MTP_NUM_HEADS=2

# Print results
echo ""
echo "=========================================="
echo "PHASE 1 ABLATION RESULTS"
echo "=========================================="
python3 ablation.py leaderboard
python3 ablation.py ablations

echo ""
echo "DECISION POINT: Review results above."
echo "Pick the best combination of Phase 1 techniques."
echo "Then run Phase 2 (eval-time) on the winning checkpoint."
```

### File: `run_phase2.sh`
```bash
#!/bin/bash
# Phase 2: Eval-time augmentation ablations.
# Run AFTER Phase 1 to find best neural base, then stack eval techniques.
# Usage: ./run_phase2.sh <best_phase1_run_id>
set -euo pipefail

BASE="${1:?Usage: run_phase2.sh <best_phase1_run_id>}"
export NPROC=1
export MAX_WALLCLOCK_SECONDS=120
export SEED=1337

echo "=========================================="
echo "PHASE 2: EVAL-TIME ABLATIONS"
echo "Base: $BASE"
echo "=========================================="

# Load base techniques
BASE_TECHS=$(python3 -c "from ablation import load_result; print(','.join(load_result('$BASE').techniques))")

# 2.1 — +Legal TTT (E2)
./run_ablation.sh \
  "p2_ttt" \
  "${BASE_TECHS},E2" \
  "$BASE" \
  NOTES="+Legal score-first TTT (AdamW, 3 epochs)" ENABLE_TTT=1

# 2.2 — +TTT + Temp Calibration (E2+E3)
./run_ablation.sh \
  "p2_ttt_tempcal" \
  "${BASE_TECHS},E2,E3" \
  "p2_ttt" \
  NOTES="+Post-TTT temp calibration T=0.98" ENABLE_TTT=1 TTT_TEMP=0.98

# 2.3 — +N-gram cache (E6+E7) without TTT
./run_ablation.sh \
  "p2_ngram_nottt" \
  "${BASE_TECHS},E6,E7" \
  "$BASE" \
  NOTES="+Multi-order n-gram cache (no TTT)" ENABLE_NGRAM=1

# 2.4 — +N-gram cache + kNN-LM (E6+E7+E8) without TTT
./run_ablation.sh \
  "p2_ngram_knn_nottt" \
  "${BASE_TECHS},E6,E7,E8" \
  "p2_ngram_nottt" \
  NOTES="+kNN-LM on top of n-gram cache" ENABLE_NGRAM=1 ENABLE_KNN=1

# 2.5 — Full stack: TTT + temp cal + n-gram + kNN
./run_ablation.sh \
  "p2_full_stack" \
  "${BASE_TECHS},E2,E3,E6,E7,E8" \
  "p2_ngram_knn_nottt" \
  NOTES="Full eval stack: TTT+tempcal+ngram+kNN" \
  ENABLE_TTT=1 TTT_TEMP=0.98 ENABLE_NGRAM=1 ENABLE_KNN=1

echo ""
echo "=========================================="
echo "PHASE 2 RESULTS"
echo "=========================================="
python3 ablation.py leaderboard
python3 ablation.py ablations
```

---

## EXECUTION INSTRUCTIONS FOR CLAUDE CODE

```
STEP-BY-STEP INSTRUCTIONS:

1. SETUP
   - Create a working directory: /workspace/pgolf
   - Clone: git clone https://github.com/openai/parameter-golf.git /workspace/pgolf/main
   - Create ablation_results/ directory
   - Write ablation.py, run_ablation.sh, run_all_ablations.sh, run_phase2.sh as specified above
   - chmod +x *.sh

2. CLONE PR BRANCHES (source code for merging)
   - git clone https://github.com/gowtham0992/parameter-golf.git /workspace/pgolf/pr569
     (Check PR #569 for exact branch name — may need: git fetch origin pull/569/head:pr569)
   - Similarly for PRs #505, #576, #727, #738
   - Alternative: Download raw train_gpt.py from each PR's "Files changed" tab

3. BUILD THE BASE
   - Start from #569's train_gpt.py (copy to /workspace/pgolf/main/train_gpt.py)
   - Verify it runs and produces ~1.1175 BPB on 8×H100

4. IMPLEMENT EACH TECHNIQUE AS A TOGGLE
   - Every technique gets an environment variable toggle: ENABLE_OPTROT, ENABLE_HYBRIDNORM, etc.
   - Default: all off (matches #569 base behavior)
   - This allows any combination to be tested via ablation scripts

5. RUN PHASE 1 ABLATIONS
   - Execute: ./run_all_ablations.sh
   - Review: python3 ablation.py leaderboard && python3 ablation.py ablations
   - DECISION: Pick best combination of Phase 1 techniques
   - Build the "champion" checkpoint with all winning techniques enabled

6. RUN PHASE 2 ABLATIONS
   - Execute: ./run_phase2.sh <champion_run_id>
   - Review results
   - The full stack (best neural base + TTT + temp cal + n-gram + kNN) is the submission candidate

7. SCALE UP FOR SUBMISSION
   - Re-run champion config on 8×H100 with MAX_WALLCLOCK_SECONDS=600
   - Run 3 seeds (1337, 42, 2024) for statistical significance
   - Verify artifact < 16MB
   - Compute p-value vs current SOTA (1.1228 or whatever is current)
   - Prepare PR with README.md, submission.json, train logs

8. ITERATE
   - If significance threshold not met, try Phase 3 experiments
   - If met, submit PR
```

---

## TECHNIQUE COUNT SUMMARY

```
PROVEN TECHNIQUES TO MERGE:           19 (from existing PRs)
UNTRIED TECHNIQUES FROM LITERATURE:   13 (never in any submission)
TOTAL INDEPENDENT TECHNIQUES:         32

MAXIMUM SIMULTANEOUS STACK:           ~22 (accounting for mutual exclusions)

Broken down:
  Architecture:   ~9 simultaneous  (A1,A4,A5,A7/A16,A8,A9/A11,A10,A12,A13 + pick from A14-A19)
  Training:       ~9 simultaneous  (T1/T11,T2,T3,T4,T5,T6,T7,T8,T9,T10 + pick from T13-T16)
  Quantization:   ~5 simultaneous  (Q5/Q6,Q7,Q8,Q9,Q4/Q10)
  Eval-time:      ~5 simultaneous  (E1,E2/E4,E3,E6+E7,E8)
```
