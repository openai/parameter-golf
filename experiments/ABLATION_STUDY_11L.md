# Ablation Study: 11L + WD=0.04 Configuration

## Base Config (exp197)
```
NUM_LAYERS=11, MUON_WD=0.04, ADAM_WD=0.04, MLP_ACTIVATION=leaky2
XSA_LAST_N=4, VE_ENABLED=1, VE_DIM=128, VE_LAYERS=9,10
LATE_QAT=1, QAT_THRESHOLD=0.15, EMA_ENABLED=1, EMA_DECAY=0.997
TTT_ENABLED=1, TTT_LR=0.002, TTT_EPOCHS=3, TTT_FREEZE_LAYERS=2
BIGRAM_VOCAB_SIZE=8192, BIGRAM_DIM=64, ROPE_BASE=50000
WARMDOWN_ITERS=3500, MATRIX_LR=0.025, SCALAR_LR=0.025
```

## Methodology
- Each test changes ONE variable from base config
- Compare step 1000 val_bpb AND final sliding BPP
- Use Nia to verify implementations match research papers before coding
- All runs on 8xH100, 600s wallclock

---

## Planned A/B Tests (Priority Order)

### Tier 1: High Confidence (zero/tiny size cost, likely to help)

| # | Test | Change | Expected | Rationale | Nia Research Needed? |
|---|------|--------|----------|-----------|---------------------|
| A1 | Star-ReLU | MLP: `relu(x)^2 * learned_scale + learned_bias` | -0.002 to -0.005 | PR505 uses it, learnable activation range per hidden unit. ~0 size cost. | Yes: verify Star-ReLU vs squared ReLU vs leaky^2 literature |
| A2 | Decoder LR 2x | DECODER_LR_MULT=2.0 | -0.001 to -0.003 | PR505 uses it. Decoder layers need faster learning (less context). Safe at 11L. | No |
| A3 | LN Scale | LN_SCALE=1 (1/sqrt(layer+1)) | -0.001 to -0.003 | Dampens deep layer norm inputs. At 11L max dampening is 0.29 (mild). Failed at 14L+WD=0.09 but context changed. | Yes: check original paper, verify placement |
| A4 | Bigram 128 dim | BIGRAM_DIM=128 (was 64) | -0.001 | 2x wider bigram embedding. +~60KB compressed. PR505 uses 128. | No |

### Tier 2: Medium Confidence (untested or mixed results at other configs)

| # | Test | Change | Expected | Rationale | Nia Research Needed? |
|---|------|--------|----------|-----------|---------------------|
| B1 | Partial RoPE 16 | ROPE_DIMS=16 | +/- 0.002 | Rotate only 16 of 64 head dims. Failed at 12L+WD=0.06. May work at 11L+WD=0.04. | Yes: check what dim ratio works best in literature |
| B2 | Star-ReLU + hidden 1792 | MLP_HIDDEN=1792 | -0.003 to -0.008 | Bigger MLP BUT slower per step (~111ms vs ~90ms). Need to check if per-step quality outweighs fewer steps. | No |
| B3 | NUM_KV_HEADS=8 | Full MHA instead of GQA | -0.001 to -0.003 | More attention capacity. Adds ~400KB. May not fit 16MB. | No |
| B4 | ROPE_BASE=10000 | Default RoPE (not NTK 50000) | +/- 0.002 | PR505 uses default. Our 50000 was from top PRs on old config. Test which is better at 11L. | No |
| B5 | LoRA TTT | Replace full-weight TTT SGD with LoRA adaptation | +/- 0.002 | Faster per-window = more adaptation in eval budget. Less expressive per step. | Yes: check LoRA TTT implementations in parameter-golf PRs |

### Tier 3: Novel Techniques (need research before implementing)

| # | Test | Change | Expected | Rationale | Nia Research Needed? |
|---|------|--------|----------|-----------|---------------------|
| C1 | TrigramHash | Add trigram hash embedding alongside bigram | -0.001 to -0.003 | Richer n-gram context. Need to check if it helps at vocab=1024. | Yes: search for trigram hash in LM literature, check if any PRs use it |
| C2 | ValueResidual | Add residual connection within attention: y = attn(x) + alpha*V | unknown | Keeps value info flowing. Different from XSA (which removes self-V). | Yes: search arxiv for "value residual" in transformers |
| C3 | GradQuant | Gradient-guided per-tensor quantization bit-width | -0.001 to -0.003 | Allocate more bits to gradient-sensitive weights, fewer to stable ones. Post-training only. | Yes: search for gradient-aware quantization, per-layer bit allocation |
| C4 | Cautious Weight Decay | Scale WD per-param based on gradient alignment | -0.001 | From modded-nanogpt PR#154. Only decay params whose gradients agree with momentum. | Yes: verify cautious WD implementation from modded-nanogpt |
| C5 | Batch Size Curriculum | Start small batch, increase during training | -0.001 to -0.002 | More gradient updates early when learning is fastest. From modded-nanogpt. | Yes: check optimal batch schedule for 600s wallclock |
| C6 | Partial Key Offset | Shift K forward for induction heads | -0.001 | From modded-nanogpt PR#169. Enables 1-layer induction. Zero params. | Yes: verify implementation details |

### Tier 4: Compression-Focused (for making things fit)

| # | Test | Change | Expected BPP | Expected Size | Rationale |
|---|------|--------|-------------|--------------|-----------|
| D1 | GPTQ-lite clipping | Percentile clip (99.99984%) in quantization | 0 BPP | -20KB | Better outlier handling. PR505 uses it. |
| D2 | Int5 MLP post-quant | MLP weights int5, attn int6 | +0.005 | -1.2MB | Only if we need massive size savings. |
| D3 | Skip gates removal | Remove skip gates if they don't help | 0 BPP | -5KB | Negligible size but clean if unhelpful. |

---

## Results

| Exp | Config | Step 1000 | Final Sliding | Artifact | Verdict |
|-----|--------|-----------|--------------|----------|---------|
| 197 | Base (11L+WD=0.04+XSA+VE+QAT) | pending | pending | pending | BASELINE |
| | | | | | |

---

## Notes
- ALWAYS use Nia to verify technique implementations against papers/PRs before coding
- At 11L + WD=0.04, features that failed at 14L + WD=0.09 may now work (weights are more expressive)
- TTT is our key differentiator (~0.010 BPP) — never disable it
- Brotli compression saves ~364KB over zstd — always use it
- Target: beat 1.1155 (our current best) and fit under 16MB
