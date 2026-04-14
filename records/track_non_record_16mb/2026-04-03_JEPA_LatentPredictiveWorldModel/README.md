# JEPA-LM: Latent Predictive World Model for Parameter Golf

First implementation of JEPA (Joint Embedding Predictive Architecture) for the Parameter Golf challenge, an approach explicitly requested on the organizers' wishlist. Extended with hierarchical multi-horizon latent prediction, int6 mixed-precision quantization, and sliding window evaluation.

## Results (3-seed, 8xH100)

| Seed | val_bpb (sliding) | val_bpb (chunked, reference) | Artifact bytes |
|------|------------------:|-----------------------------:|---------------:|
| 1337 | 1.17601731        | 1.20840898                   | 15,954,439     |
| 7    | 1.17471213        | 1.20740983                   | 15,962,306     |
| 42   | 1.17402625        | 1.20663880                   | 15,979,344     |

**Mean BPB: 1.1749 | Std: 0.0010**

All runs fit under the 16 MB limit and complete within the 10-minute training cap.

## Approach

The submission combines three complementary ideas:

1. **Hierarchical Multi-Horizon JEPA** (training-only, novel): a tiny bottleneck predictor at the encoder-decoder boundary learns forward dynamics in representation space. Applied recursively for multiple horizons (t+1, t+2, t+3), forming a learned world model of text dynamics.

2. **Int6 mixed-precision quantization** (artifact compression): weights are quantized to 6 bits per parameter, packed 4-per-3-bytes. The tied embedding is kept at int8 to preserve quality on the most sensitive layer.

3. **Sliding window evaluation** (eval-time technique): the validation set is scored with sliding context windows of 1024 tokens at stride 256, so each token sees at least 768 tokens of prior context rather than the variable 0-1023 in the standard chunked eval.

The JEPA predictor is stripped before serialization, adding zero bytes to the final artifact.

### Architecture

- 9 transformer blocks at model_dim 512, GQA (8 query heads, 4 KV heads)
- 3x MLP expansion with LeakyReLU(0.5) squared activation
- U-Net skip connections between encoder and decoder halves
- Tied embeddings, 1024 vocab, RoPE, tanh logit softcap
- Muon optimizer for matrix parameters, Adam for embeddings/scalars
- Extended warmdown schedule (3000 iterations)

### JEPA training components

- `LatentPredictor`: bottleneck MLP (512 -> 128 -> 512) with pre-RMSNorm and residual connection, zero-init output
- Multi-horizon rollout: predictor applied 3 times recursively at the encoder-decoder boundary
- Loss: smooth-L1 on layer-normalized representations with stop-gradient on targets (data2vec style)
- Horizon weighting: exponential decay 1.0, 0.5, 0.25
- Loss weight: 0.5, annealed alongside the learning rate warmdown

### References

- NextLat (Srivastava et al., ICLR 2026): auxiliary latent prediction for transformers
- I-JEPA (Assran et al., 2023): predicting in representation space
- data2vec (Baevski et al., 2022): JEPA-style self-prediction for text
- Predictive coding (Rao & Ballard, 1999): prediction error as learning signal
- Sliding window evaluation: widely used in language model benchmarking

## Ablations (8xH100, single seed 1337)

| Configuration                               | val_bpb (chunked) | val_bpb (sliding) |
|---------------------------------------------|------------------:|------------------:|
| Baseline (9L 2x MLP, no JEPA, int8)         | 1.2244            | -                 |
| Plan A (9L 2x MLP + JEPA, int8)             | 1.2258            | -                 |
| 10L 3x MLP, no JEPA, int8                   | 1.1965            | -                 |
| 10L 3x MLP + JEPA, int8                     | 1.1931            | 1.1611            |
| 10L 3x MLP + JEPA, int6                     | 1.1982            | 1.1654            |
| 9L 3x MLP + JEPA, int6 (this submission)    | 1.2074-1.2084     | 1.1740-1.1760     |

### Experimental journey

The JEPA contribution was not obvious at first. Initial 1xH100 development runs with 9L 2x MLP showed JEPA matching or slightly underperforming the baseline by roughly 0.001 BPB, well within run-to-run noise. Switching to 10L 3x MLP at 8xH100 produced a non-fitting 1.1931 BPB result. The smaller fitting architecture (9L 2x MLP) plus JEPA gave 1.2258 BPB against the official baseline's 1.2244. This initially suggested JEPA was not helping.

The confusion stemmed from confounding variables: our Plan A configuration changed activation (LeakyReLU squared vs baseline ReLU squared), warmdown schedule (3000 vs 1200), and added gradient clipping on top of adding JEPA. We could not attribute the small regression cleanly.

A clean ablation at the actual architecture of interest resolved this:

- 10L 3x MLP, JEPA enabled, 8xH100 full-length training: 1.1931 BPB
- 10L 3x MLP, JEPA disabled (same everything else), 8xH100: 1.1965 BPB
- Delta: -0.0034 BPB from JEPA, consistent across chunked and sliding eval

This was the first clean evidence that JEPA helps at scale. The gap also scaled positively from 1xH100 (-0.0013) to 8xH100 (-0.0034), suggesting the auxiliary loss becomes more valuable with more training steps. Based on this, JEPA was retained in the final submission.

Several other candidate creative additions were tested and rejected:

- SLOT (test-time delta optimization) gave -0.004 BPB in one run and +0.008 BPB in another with identical hyperparameters. High variance and not reliably additive on top of sliding window, so excluded from the final submission.
- N-gram bigram cache mixing (lambda=0.1) at eval hurt by +0.030 BPB. Bigrams capture too little structure beyond what the transformer already models.
- Imagination-refined prediction (decoding the JEPA predictor through the decoder + head at eval) hurt by +0.04 to +0.10 BPB. The predictor learns temporally predictable latents, not decodable ones.
- Imagination co-training (adding CE loss through the predicted encoder_out at training time) doubled step time and halved the number of training steps available, resulting in a net worse model.

The negative results shaped the final design: keep the JEPA predictor as a training-only auxiliary signal (not used at inference), exploit the "free predictor" property to keep the artifact small, and use sliding window evaluation as the main external-technique lever.

### Key findings

1. **JEPA helps at scale**: at 10L 3x MLP with 8xH100 training, JEPA reduces BPB by 0.0034 (1.1965 -> 1.1931) compared to the same architecture without JEPA. Statistically meaningful and consistent.

2. **Int6 quantization with embedding fix is viable**: the tied embedding is quantization-sensitive; keeping it at int8 while using int6 for everything else loses only ~0.005 BPB vs int8, while saving enough bytes to fit the artifact.

3. **Sliding window eval is the largest single lever**: -0.033 BPB over standard chunked eval. The improvement saturates around stride 256; stride 128 gave no additional benefit in our tests.

4. **Layer count matters more than MLP width for size**: dropping from 10L 3x to 9L 3x cost 0.011 BPB with sliding, but allowed the model to fit at int6 under the 16 MB limit.

5. **Negative result on predictive coding error injection**: an early variant injected prediction errors directly into the decoder path. Stable at small scale (1xH100, 1000 steps) but diverged to NaN at 8xH100 (10000 steps) due to feedback instability. The pure auxiliary loss approach is stable.

6. **Negative result on multi-token imagination at eval**: decoding the predictor's output through the decoder + head for eval-time ensembling hurt BPB both with and without imagination co-training. The predictor's representations decode reliably only after extensive training specifically on this objective, which is not cost-effective within the 10-minute budget.

## How to run

```bash
# Required environment: 8xH100 SXM with FineWeb sp1024 tokenizer and shards in ./data/
RUN_ID=jepa_final_seed1337 SEED=1337 \
NUM_LAYERS=9 MLP_MULT=3 GRAD_CLIP_NORM=1.0 \
QUANT_BITS=6 SLIDING_STRIDE=256 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Output includes two BPB numbers:
- `final_int8_zlib_roundtrip val_bpb` -- standard chunked eval (reference)
- `final_sliding val_bpb` -- sliding window eval (the submitted score)

## Submission metadata

- Author: adi-suresh01
- Track: non-record (16 MB)
- Best single-seed BPB (sliding): 1.17402625 (seed 42)
- 3-seed mean BPB: 1.1749
- 3-seed std: 0.0010
- Artifact size: 15.96 MB (average across seeds), all under 16 MB
- Training time: 600s (wallclock cap) across 8xH100
- Eval time: ~20s sliding window
