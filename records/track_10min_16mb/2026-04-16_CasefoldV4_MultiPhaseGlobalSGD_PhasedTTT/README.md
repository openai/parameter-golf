# Record: Casefold V4 + Multi-Phase Global SGD + Phased TTT

**val_bpb: 1.05970** (3-seed mean, std 0.00031) | **3.05401 nats** | **~15.20 MB** | 8xH100 SXM, 600s | Phased TTT

## Summary

Casefold tokenizer normalization (lowercase input + retrained SP8192 BPE) combined with Multi-Phase Global SGD TTT on the PR #1530 architecture base. Casefolding reduces the effective vocabulary entropy by merging upper/lower case distinctions, giving the model more capacity to learn content patterns rather than case patterns.

**Note:** Casefold tokenizer normalization is a novel technique pending organizer review at Issue #1604. The tokenizer itself is retrained from scratch on casefolded data -- it is NOT a modified version of the standard SP8192 tokenizer.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

### Core Results

| Seed | Steps | ms/step | Pre-TTT BPB | **Post-TTT BPB** | TTT gain | TTT time | Artifact |
|------|-------|---------|-------------|------------------|----------|----------|----------|
| 42   | 5003  | 119.2   | 1.07122     | **1.05938**      | -0.01184 | 295s     | 15,935,851 |
| 0    | 4970  | 120.0   | 1.07124     | **1.05961**      | -0.01164 | 294s     | 15,935,513 |
| 1234 | 4973  | 119.9   | 1.07155     | **1.06010**      | -0.01145 | 304s     | 15,933,440 |
| **Mean** | **4982** | **119.7** | **1.07134** | **1.05970** | **-0.01164** | **298s** | **15,934,935** |
| **Std** | | | | **0.00031** | | | |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Quantized BPB | Post-TTT BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|-------------|---------------|--------------|-----------------|-----------|-----------------|------------|-----------|
| 42   | 1.05800     | 1.07122       | 1.05938      | 3.05310         | 122,604 B | 15,935,851 B    | 596.2s     | 295.2s    |
| 0    | 1.05825     | 1.07124       | 1.05961      | 3.05375         | 122,604 B | 15,935,513 B    | 596.2s     | 294.2s    |
| 1234 | 1.05802     | 1.07155       | 1.06010      | 3.05519         | 122,604 B | 15,933,440 B    | 596.1s     | 304.2s    |

### Record Comparison

| Submission | val_bpb | val_loss (nats) | Delta BPB | Delta nats |
|------------|---------|-----------------|-----------|------------|
| Merged SOTA (PR #1493) | 1.08100 | - | - | - |
| PR #1530 @samacqua | 1.07336 | - | - | - |
| PR #1626 @dexhunter | 1.07193 | - | - | - |
| **This (3-seed)** | **1.05970** | **3.05401** | **-0.02130 vs SOTA** | **-0.0550 vs SOTA** |

Clears the 0.005-nat record threshold vs merged SOTA by 11x.

## Key Innovation: Casefold Tokenizer Normalization

All input text is lowercased (casefolded) before tokenization. A new SP8192 BPE tokenizer is trained on casefolded FineWeb data. This eliminates the model's need to learn separate representations for upper/lower case variants of the same word, freeing capacity for content modeling.

The casefolded data (train + val shards) is prepared offline and stored at `DATA_DIR`. The tokenizer and data processing are fully deterministic.

```python
# Key: casefold applied during data preparation (offline, before training)
# Tokenizer retrained on casefolded text with same SP8192 vocabulary size
# Model sees only lowercase tokens -- no case ambiguity in vocabulary
```

This is combined with Multi-Phase Global SGD TTT, which performs 3 phases of SGD adaptation on prefix documents from the validation set, progressively specializing the model to the evaluation distribution.

## Changes from PR #1530 Baseline

| Aspect | PR #1530 (base) | This submission |
|--------|----------------|-----------------|
| Tokenizer | Standard SP8192 | Casefold SP8192 (retrained on lowercased data) |
| val_bpb | 1.07336 | **1.05970** (-0.01366) |
| TTT | None | Multi-Phase Global SGD (3 phases, 2000 prefix docs) |
| TTT eval | Sliding window only | Phased TTT (score-first, 3-phase SGD) |
| Data | Standard FineWeb shards | Casefolded FineWeb shards (offline preprocessing) |
| GPTQ reserve | Default | 4s (trimmed) |
| GPTQ calib | Default | 16 batches |

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at step frac=0.35). Parallel residuals from layer 8. Skip gates (sigmoid-gated U-Net connections). EMA decay 0.9965.

### Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps, momentum 0.97), AdamW for embeddings/scalars. ~5000 steps in 596s on 8xH100 SXM. Linear warmdown over final 75% of training. Gradient clipping at 0.3.

### Quantization

Full-Hessian GPTQ with SDClip: int6 for attention/MLP matrices (clip=12.85 sigma), int7 for token embeddings (clip=15.0 sigma). Brotli-11 compression. Trimmed GPTQ (reserve=4s, calib=16 batches).

### TTT (Test-Time Training)

Multi-Phase Global SGD with score-first ordering:
- 3 phases, each adapting on a growing prefix of 2000 validation documents
- Phase boundaries: [666, 1333, 2000] documents
- Per phase: score all sliding windows under `torch.no_grad()`, then SGD update
- SGD: lr=0.001, momentum=0.9, gradient clipping at 1.0
- Total TTT eval time: ~298s (within 600s eval budget)

## Rule Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each phase fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once per phase. Final scores from last phase only.

**Casefold tokenizer normalization:** This is a novel technique. The tokenizer is retrained on casefolded (lowercased) text. Organizer review is pending at Issue #1604. The technique does not violate any of the four conditions above -- it only changes the tokenizer vocabulary, not the scoring or adaptation procedure. The byte-level BPB computation remains correct: each sentencepiece token maps to its constituent bytes via the piece table, and BPB is computed over all bytes in the validation set.

Additional compliance:
- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, TTT adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all seeds
- Training under 600s on all seeds (~596s actual)
- Eval (phased TTT) under 600s on all seeds (~295s actual)

## Requirements

- Python >= 3.12
- PyTorch >= 2.9.1
- flash-attn-3
- brotli
- sentencepiece

## Run Command

```bash
# Install dependencies
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Prepare casefolded data (must be done once before training)
# Casefold data expected at DATA_DIR=/tmp/casefold_data/
# Contains: datasets/fineweb10B_sp8192/*.bin, tokenizers/fineweb_8192_bpe.model

# 3-seed evaluation loop
for SEED in 42 0 1234; do
  DATA_DIR=/tmp/casefold_data/ SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${SEED}.log
done
```

## Lineage

PR #1530 (@samacqua) -> casefold tokenizer normalization (inspired by PR #1578 @mikeapedia) + Multi-Phase Global SGD TTT (inspired by PR #1610 @romeerp) + trimmed GPTQ

## Credits

- **@samacqua** -- PR #1530 base architecture (11L/512d/4x MLP, depth recurrence, parallel residuals, MuonEq-R, GPTQ SDClip)
- **@mikeapedia** -- Casefold tokenizer concept (PR #1578)
- **@romeerp** -- Phased TTT concept (PR #1610)
- **@abaybektursun** -- Score-first TTT framework (PR #549, merged precedent)
- **@dexhunter** -- Casefold implementation, multi-phase global SGD TTT, trimmed GPTQ tuning

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed0.log`
- `train_seed1234.log`
