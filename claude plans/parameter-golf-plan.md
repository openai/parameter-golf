# Parameter Golf: RYS Layer Duplication at Eval Time

## Context & Goal

Apply RYS (Repeat Your Self) — duplicating mid-stack transformer layers at eval time only — to the current Parameter Golf SOTA. Inspired by David Noel Ng's work showing that repeating "reasoning" layers in trained transformers improves performance with zero retraining.

- **Primary base**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — current SOTA, 11L, LeakyReLU(0.5)², Legal score-first TTT, Parallel Muon optimizer
- **Control base**: PR #505 (`SwiGLU+VE128+NoTTT`, val_bpb=1.1181) — best non-TTT submission, 11L, GEPA architecture. We test BOTH because the SOTA uses Test-Time Training (TTT), which continuously adapts weights during eval. RYS would repeat TTT-adapted layers, making it hard to isolate the RYS effect. PR #505 has frozen weights at eval time, giving a clean control experiment.
- **Storage cost of RYS**: ~0 bytes (a few extra lines in eval code)

### Key Risks
- The model is only 11 layers deep and quantized to int5/int6. The three-phase encode/reason/decode structure that makes RYS work on 64-layer models may not cleanly separate here. Quantization error may also compound through repeated passes.
- TTT interaction is unknown: TTT adapts weights per-document, so repeated layers use adapted weights. Could be synergistic (adapted reasoning layers benefit more from a second pass) or catastrophic (TTT already pushed weights to a fragile optimum).

---

## Phase 1: Setup & Baseline

### 1.1 Clone and reproduce
```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
bash prepare.sh  # downloads dataset + tokenizer
```

### 1.2 Train BOTH base models and record baselines
```bash
# Primary: current SOTA (with TTT)
cd records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/
SEED=42 bash eval/eval.sh
# Record the exact val_bpb number

# Control: PR #505 (no TTT) — checkout the PR branch or copy its train_gpt.py
# into a working directory and run it the same way
```

Train from scratch then evaluate — takes ~10min on 8xH100 SXM per run. Save the trained checkpoints so you don't have to retrain for every RYS config. Check `train_gpt.py` for how the model is saved after training and loaded for eval.

We run the full RYS sweep on BOTH models. Comparing the two tells us:
- Whether RYS works at all at 11 layers
- Whether TTT helps or hurts the RYS effect
- If only the non-TTT model benefits, that's still a useful finding (RYS as a TTT alternative)

---

## Phase 2: Implement RYS

### 2.1 Find the eval forward pass
In `train_gpt.py`, locate the model's forward method used during evaluation. It will look something like:

```python
for layer in self.layers:
    x = layer(x)
logits = self.output(x)
```

### 2.2 Add RYS layer duplication (eval-only)
Modify the forward pass to accept RYS parameters. This should ONLY activate during eval, not during training.

```python
def forward(self, x, rys_start=None, rys_end=None, rys_repeats=2):
    for i, layer in enumerate(self.layers):
        x = layer(x)
        # At the end of the RYS block, loop back
        if rys_start is not None and i == rys_end - 1:
            for _ in range(rys_repeats - 1):  # -1 because we already did one pass
                for j in range(rys_start, rys_end):
                    x = self.layers[j](x)
    logits = self.output(x)
    return logits
```

Be careful with:
- **KV cache**: If eval uses a KV cache for sliding window, the repeated layers will generate additional KV entries. Make sure the cache handling is correct.
- **TTT interaction**: The SOTA submission uses Legal TTT (test-time training). RYS should be applied AFTER TTT has finished adapting the weights, during the final eval forward pass. Don't apply RYS during TTT's adaptation steps. (This is why we also test PR #505 which has no TTT.)

### 2.3 Verify correctness
Before sweeping, test with rys_start=None (disabled) and confirm val_bpb matches baseline exactly. Any discrepancy means you introduced a bug.

---

## Phase 3: Exhaustive Sweep (run on BOTH models)

### 3.1 Sweep all (start, end) block duplications
With 11 layers, there are C(11,2) + 11 = 66 possible contiguous blocks. This is small enough to sweep exhaustively.

```python
import itertools

num_layers = 11
baseline_bpb = <your measured baseline>
results = {}

for start in range(num_layers):
    for end in range(start + 1, num_layers + 1):
        # Skip blocks that include layer 0 or layer 10 (encode/decode boundaries)
        # Actually, don't skip — sweep everything and let the data tell you
        
        val_bpb = evaluate_with_rys(model, val_data, rys_start=start, rys_end=end, rys_repeats=2)
        delta = val_bpb - baseline_bpb
        results[(start, end)] = {'bpb': val_bpb, 'delta': delta, 'extra_layers': end - start}
        print(f"RYS({start},{end}) +{end-start}L: {val_bpb:.4f}  delta={delta:+.4f}")

# Sort by delta (lower is better for BPB)
for k, v in sorted(results.items(), key=lambda x: x[1]['delta']):
    print(f"  {k}: {v['delta']:+.4f} bpb ({v['extra_layers']} extra layers)")
```

### 3.2 Single-layer repeat sweep
Also test repeating individual layers multiple times (Ng's repeat-x8 experiment):

```python
for layer_idx in range(num_layers):
    for num_repeats in [2, 3, 4, 5]:
        val_bpb = evaluate_with_rys(model, val_data, 
                                     rys_start=layer_idx, rys_end=layer_idx+1, 
                                     rys_repeats=num_repeats)
        delta = val_bpb - baseline_bpb
        print(f"Layer {layer_idx} x{num_repeats}: {val_bpb:.4f}  delta={delta:+.4f}")
```

### 3.3 Visualize as heatmap (one per model)
Plot the (start, end) → delta results as a heatmap (upper triangular matrix, like Ng's brain scans). If there's a red zone in the middle, RYS works and you can see the reasoning circuit. Generate one heatmap for the TTT model and one for the non-TTT model — differences between them reveal how TTT affects the internal structure.

```python
import numpy as np
import matplotlib.pyplot as plt

heatmap = np.full((num_layers, num_layers), np.nan)
for (s, e), v in results.items():
    heatmap[s, e-1] = v['delta']

plt.imshow(heatmap, cmap='RdBu', vmin=-0.01, vmax=0.01)
plt.colorbar(label='BPB delta (negative = better)')
plt.xlabel('End layer')
plt.ylabel('Start layer')
plt.title('RYS Heatmap: 11L LeakyReLU LegalTTT ParallelMuon')
plt.savefig('rys_heatmap.png', dpi=150)
```

---

## Phase 4: Interpret & Submit

### 4.1 If RYS improves BPB
- Identify the Pareto-optimal configs (best delta per extra-layer-count)
- Check if the improvement exceeds 0.005 nats (required for SOTA record)
- Run 3 seeds to confirm statistical significance (p < 0.01)
- Submit as a record if it beats SOTA, or non-record if interesting but not SOTA

### 4.2 If RYS hurts or is neutral everywhere
This is still a valuable negative result. Write up:
- The heatmap showing no clear reasoning region
- Comparison to Ng's 64L results — what's different at 11L?
- Hypothesis: 11 layers is below the threshold for clean phase separation
- Suggestion: try RYS on a deeper (13-15L) model if tokenizer pruning frees bytes
- Submit as a non-record submission

### 4.3 Submission format
```
records/track_non_record_16mb/YYYY-MM-DD_RYS_LayerDuplication/
├── README.md          # Full writeup with heatmap, methodology, results
├── submission.json    # {"val_bpb": X.XXXX, "artifact_bytes": XXXXXX}
├── train.log          # Training log from base model
├── train_gpt.py       # Modified script with RYS eval code
└── rys_heatmap.png    # Visualization
```

In the README, cite:
- Ng's RYS Part 1 & 2 (https://dnhkng.github.io/posts/rys/, https://dnhkng.github.io/posts/rys-ii/)
- The base submission you forked from
- The quantization error amplification finding from PR #363

---

## Quick Reference

- **Base submission**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Repo**: https://github.com/openai/parameter-golf
- **RYS blog**: https://dnhkng.github.io/posts/rys-ii/
- **RYS code**: https://github.com/dnhkng/RYS
- **Live commentary**: https://github.com/openai/parameter-golf/issues/140
- **Deadline**: April 30, 2026
