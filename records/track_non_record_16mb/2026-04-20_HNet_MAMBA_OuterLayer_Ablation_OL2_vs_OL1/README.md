# [Non-record] H-Net with MAMBA Outer Layers: Ablation Reveals Training Instability — Final OL1 Result: 1.5194 INT6 BPB

> **9L H-Net (1 Encoder + 7 Main + 1 Decoder) + byte260 + GQA KV4 + INT6 GPTQ + zlib-9 + Stride-64 Sliding Eval; val_bpb: 1.5194**

Follow-up to [PR #1104](https://github.com/openai/parameter-golf/pull/1104) and the [Scaled H-Net submission](../2026-04-01_Scaled_HNet_Byte260_and_Sp1024), which established that H-Net's outer MAMBA2 layers act as byte-level boundary encoders/decoders. This submission investigates whether increasing the number of MAMBA outer layers (OLs) within a fixed 9-layer budget improves performance — and finds the opposite: more MAMBA outer layers cause training instability, and reducing OL count from 2 to 1 yields a stable, convergent run.

## Key Results

#### Final run (OL1, 20k steps)

| Config | BPB (INT6 Sliding) | BPB (INT6 Roundtrip) | BPB (Float) | Steps | Artifact size |
|--------|-------------------:|---------------------:|------------:|------:|--------------:|
| **H-Net `byte260` 9L OL1 (KV4)** | **1.5194** | 1.5576 | 1.4770 | 20,000 | 13.1 MB |

#### MAMBA outer-layer ablation (fixed 9-layer budget)

| outer_layers | Architecture | model_params | ms/step | Outcome |
|:---:|---|---:|---:|---|
| **1** | 1 enc + 7 main + 1 dec | 19.06M | ~4,267 | Stable convergence to **1.4770 BPB** |
| 2 | 2 enc + 5 main + 2 dec | 20.66M | ~9,248 | **Collapse at step 2000** (1.81→2.53 BPB) |

Adding a second pair of MAMBA outer layers doubled per-step time and caused catastrophic training divergence.

## Architecture

Same 1-stage H-Net layout as [PR #1104](https://github.com/openai/parameter-golf/pull/1104), with the outer layer count reduced to 1:

```
Input -> Embedding -> Encoder (1 MAMBA2 block) -> Routing -> ChunkLayer (L -> C)
      -> Main Transformer (7 attn blocks) -> DeChunkLayer (C -> L)
      -> + Residual Skip -> Decoder (1 MAMBA2 block) -> LM Head
```

- **9 layers total**: 1 encoder + 7 main + 1 decoder (OL1)
- **512 model dim**, 8 heads, 4 KV heads (GQA)
- **19.06M parameters**, `byte260` tokenizer (vocab=260)
- **Chunk target size**: 6, `CHUNK_DIVISOR=4`

## MAMBA Outer-Layer Ablation

We ran two configurations within a fixed 9-layer budget, redistributing capacity between MAMBA outer layers and attention main layers:

**OL2 (2 enc + 5 main + 2 dec):** Training proceeded normally until step ~1400 (best val_bpb: 1.7818), then stagnated and collapsed at step 2000 with a sharp loss spike. Val BPB jumped from 1.8110 at step 1900 to 2.5314 at step 2000 — a 39.8% degradation in a single 100-step window. The H-Net routing statistics show the chunker became erratic (avg_chunk_len drifted from ~6 bytes to ~8–10 bytes) in the steps preceding collapse. The run was aborted. Additionally, OL2 was 2.2× slower per step (~9,248ms vs ~4,267ms), consuming nearly twice the wall-clock time to produce worse outputs.

**OL1 (1 enc + 7 main + 1 dec):** Training remained stable across all 20,000 steps with consistent improvement. The chunker converged to avg_chunk_len ≈ 6.1 bytes, matching the `target_avg_chunk_len=6.0` target throughout.

The mechanism is likely that two sequential MAMBA2 layers in the encoder pathway over-constrain boundary representation early in training, creating gradient interference that destabilizes the routing module. Reducing to one MAMBA layer per side gives the routing module a shorter dependency chain and converges more reliably.

## Training Dynamics (OL1 final run)

| Step | val_bpb |
|-----:|--------:|
| 0 | 7.7850 |
| 500 | 1.9381 |
| 1,000 | 1.8079 |
| 2,000 | 1.7136 |
| 5,000 | 1.6030 |
| 10,000 | 1.5381 |
| 15,000 | 1.4988 |
| 18,000 | 1.4868 |
| 19,000 | 1.4836 |
| **20,000** | **1.4770** |

Val BPB was still decreasing at step 20,000, suggesting headroom remains for longer runs.

## Quantization

INT6 GPTQ with zlib-9 compression, applied after training:

| Metric | Value |
|--------|------:|
| Model artifact (INT6 GPTQ zlib-9) | 13,008,061 bytes |
| Code size | 94,168 bytes |
| **Total** | **13,102,229 bytes** |
| Headroom | 2,897,771 bytes |
| INT6 roundtrip val_bpb | 1.5576 |
| **INT6 sliding window val_bpb** | **1.5194** |
| Quantization gap (roundtrip) | +0.080 |
| Sliding window gain vs roundtrip | −0.038 |

QAT (`LATE_QAT_THRESHOLD=0.15`) was active from early in warmdown. The quantization gap (+0.080 roundtrip) is larger than in the reference Scaled H-Net submission (+0.014), consistent with a smaller, less overparameterized model being harder to quantize.

## Reproduction

```bash
# Final OL1 run (1.4770 BPB float, 1.5194 INT6 sliding)
COMPILE_MODEL=1 OUTER_LAYERS=1 NUM_LAYERS=9 MODEL_DIM=512 \
    NUM_HEADS=8 NUM_KV_HEADS=4 \
    TRAIN_SEQ_LEN=1024 ITERATIONS=20000 \
    LATE_QAT_THRESHOLD=0.15 WARMDOWN_ITERS=1200 \
    TARGET_AVG_CHUNK_LEN=6.0 \
    MODEL_SAVE_PATH=final_model.pt \
    python train_hnet.py
```

> *Note*: All experiments use `seed=1337`.

## Compliance

- [x] Artifact ≤16,000,000 bytes (13,102,229 bytes — 2.9 MB headroom)
- [x] No training on validation data
- [x] No network calls during evaluation
- [x] Non-record: extended run exceeds 10 min wallclock (**20k steps / ~23.7 hours**)
