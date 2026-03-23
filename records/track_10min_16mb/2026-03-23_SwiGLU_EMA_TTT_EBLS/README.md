# SwiGLU + EMA + TTT Memorization Analysis

**Author:** Robby Sneiderman ([@Robby955](https://github.com/Robby955))

**Verified BPB:** 1.1679 (standard sliding-window eval, no TTT, 8xH100 SXM)

**Artifact:** 15,528,857 bytes (code: 58,274 + weights: 15,470,583)

## Results

| Metric | Value |
|--------|-------|
| **Verified BPB (no TTT)** | **1.1679** |
| Training steps | 4,238 (8xH100 SXM, ~141ms/step) |
| Model params | 25,517,137 |
| Artifact size | 15.53 MB |

## TTT Memorization Analysis (Research Contribution)

We ran extensive multi-epoch TTT experiments and developed a diagnostic to distinguish genuine domain adaptation from test-set memorization. **Per Issue #402**, multi-epoch TTT is not valid for record claims — we present these results purely as a methodological contribution.

### The Diagnostic

After TTT adaptation, we run standard sliding-window eval (stride=64) on the adapted weights. If the model genuinely learned better representations, sliding eval (with overlapping context) should score *better* than the TTT-loop (non-overlapping chunks). If the model merely memorized token sequences, the TTT-loop score will be artificially low while sliding eval reveals the true performance.

| TTT Config | TTT-Loop BPB | Sliding Diagnostic | Gap | Interpretation |
|------------|-------------|-------------------|-----|----------------|
| 0 epochs (baseline) | — | 1.1679 | — | No adaptation |
| 3 epochs, flat 5e-4 | 1.1032 | 1.0476 | -0.056 | Sliding BETTER — real adaptation |
| 5 epochs, cosine 7e-4 | 1.0101 | 1.0022 | -0.008 | Sliding BETTER — real adaptation |
| **10 epochs, flat 5e-4** | **0.8566** | **0.9229** | **+0.066** | **TTT-loop better — MEMORIZATION** |

**Key finding**: At 10 epochs with flat LR, the TTT-loop reports 0.8566 BPB — but the sliding diagnostic reveals the actual prediction quality is only 0.9229 BPB. The 0.066 gap is pure memorization of token sequences. Submissions reporting sub-0.95 BPB from high-epoch TTT should be scrutinized with this diagnostic.

**Implication for the competition**: Multi-epoch TTT conflates domain adaptation with test-set memorization. We recommend that TTT submissions either (a) use strictly single-pass score-first TTT per Issue #402, or (b) report the sliding diagnostic alongside TTT-loop BPB to verify legitimacy.

### TTT Technical Details (for reproducibility of the analysis)

- Sequential score-then-train on non-overlapping 2048-token chunks
- Batch 8 chunks per forward pass
- Freeze embeddings (tok_emb, bigram) — adapt only attention and MLP 2D weights
- AdamW optimizer, wd=0.0
- **Global cosine LR decay** across all epochs (single cosine curve, not per-epoch reset)
- **Per-layer LR multipliers**: `lr_mult = 0.5 + 0.5 * (layer_idx / (num_layers - 1))`

The global cosine schedule is what enables 5 epochs without crossing into memorization — by epoch 5, the LR has decayed to ~0.000002, minimizing further adaptation. With flat LR, 5+ epochs crosses the memorization boundary.

## Architecture

Built on thwu1 PR #180 (which built on unnir PR #162):

1. **SwiGLU MLP** replacing ReLU-squared. `silu(W_gate @ x) * (W_up @ x)` with `swiglu_mult=2.0`.
2. **EMA** (decay=0.9985) replacing SWA during warmdown.
3. **Int5 quantization for all weights** with 5% magnitude pruning, zstd-22.
4. 512-dim, 8 heads, 4 KV heads, 10 transformer layers
5. BigramHash (10,240 buckets, 128-dim), SmearGate
6. Muon optimizer (WD=0.04, matrix_lr=0.02, momentum=0.99)

## EBLS Exploration

We explored Empirical Bayes Layer Sharing with learned shrinkage gammas (see [companion repo](https://github.com/Robby955/parameter-golf-ebls)):

- **MLP gammas → 0.0000**: Fully shared MLP is optimal under compression constraints
- **Attention gammas near-zero**: Trace specialization in early layers only
- **LoRA rank threshold**: Rank 8 → all sharing; rank 16 → mild specialization
- **Quantization amplification**: 0.19 BPB compiled-vs-eager gap from depth recurrence

## Negative Results

- **Trigram hashing**: 3-token XOR hash did not improve over bigram (1.0532 vs 1.0320)
- **Late QAT**: STE-based int5 simulation added 13ms/step overhead; lost training steps outweighed benefits
- **11 layers**: Either exceeds 16MB (SWIGLU 2.0) or trains too slowly (SWIGLU 1.7)
- **Per-epoch cosine**: Resetting cosine each epoch was worse than flat LR

## Reproducing

```bash
# 8xH100 SXM, 10-minute wallclock training
NUM_LAYERS=10 SWIGLU_MULT=2.0 TTT_STEPS=0 PRUNE_FRAC=0.05 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- thwu1 PR #180 (base architecture, int5/int6, SWA, BigramHash)
- unnir PR #162 (10L, MLP 3x, SmearGate, MuonWD)
- felipe-parodi (EMA concept)
- sjp611 (AdamW TTT concept)
- JoeProAI PR #462 (sequential TTT approach, SwiGLU)
- andrewbaggio1 PR #509, newjordan PR #508 (TTT epoch scaling data, embedding freeze)
- ndokutovich PR #486 (per-layer LR concept, global cosine TTT)
