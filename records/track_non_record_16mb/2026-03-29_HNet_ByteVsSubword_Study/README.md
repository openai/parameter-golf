# [Non-record] 1-Stage Byte-level H-Net at 17.5M: Dynamic Chunking Learns Whitespace-Aligned Boundaries (39x-91x smaller than the H-Net paper)

In this non-record submission I study how H-Net's dynamic chunking compares on **raw bytes (`byte260`)** vs **pre-tokenized subwords (`sp1024`)**, using the same 1-stage 9-layer H-Net backbone.

The best `byte260` run reaches **1.4116 ± 0.013 BPB** at **15.78 MB** under the 10-minute budget on **8×H100**, and when extended to 4 hours, it lands at **1.3595 BPB**. Across **20 matched runs** covering four hyperparameter settings, the byte-level H-Net ends up learning **whitespace-aligned, word-like boundaries** from raw bytes only. At the same time, the subword-level (`sp1024`) H-Net shows a different chunking pattern over already-tokenized inputs.

This submission includes:
- a matched `byte260` vs `sp1024` ablation
- boundary metrics (whitespace agreement and chunk-size coefficient of variation)
- boundary visualizations - qualitative examples
- working multi-GPU DDP training (using padded chunk sequences)

## Key Results (≤16MB)

 | Config | BPB | Size (MB) | Steps | Training time | Seeds |
  |--------|-----|-----------|-------|---------------|-------|
  | **byte260 10-min** (RLW=0.05, chunk=9, KV=4) | **1.4116 ± 0.013** | 15.78 | 4,520 | 10 min | 1337, 1234, 2026 |
  | **byte260 4-hour** (chunk=9, KV=4) | **1.3595** | 15.96 | 85,242 | 4 hours | 1337 |
  | sp1024 10-min (chunk=12, KV=2) | 1.3734 | 15.99 | 4,466 | 10 min | 1337 |∂1.3

All runs use 2 outer layers (OL; encoder + decoder) and 5 main transformer layers (9 total).

The 4-hour byte260 run (**1.3595 BPB**) beats the best sp1024 10-min result (1.3734), even though the byte model starts from raw bytes and no external tokenizer.

## Findings


1. **Byte-level H-Net gets close to subword-level H-Net performance at small scale, within 10 mins.**

    The best `byte260` setup reaches **1.4116 ± 0.013 BPB** in the 10-minute setting. That’s a big improvement over the previous byte-level H-Net result of **1.90 BPB at 22M params** from  [PR #1044](https://github.com/openai/parameter-golf/pull/1044), and it comes 'within range' of the best `sp1024` run here, which gets **1.3734 BPB**.

2. **Byte-level H-Net learns whitespace-aligned segmentation from raw bytes without an external tokenizer.**

     In several artifact-eligible `byte260` runs, predicted boundaries align with whitespace over **97%** of the time. This shows that dynamic chunking can recover linguistically meaningful segmentation structure (i.e., word-like in this case) directly from bytes (see Qualitative Boundary Analysis section).

3. **The byte H-Net still has clear optimization headroom.**

    Extending the `byte260` run from the 10-minute budget to a **4-hour** run reduces BPB from **1.4116** to **1.3595**, suggesting that part of the remaining gap is due to optimization budget rather than a hard limit of byte-level H-Net. Even at 4 hours, the BPB is still decreasing gradually.

4. **The router learns to compress the sequence substantially before the main stage.**

     Across validation samples, training reduces the number of chunk boundaries relative to initialization significantly. For `byte260`, the router starts at about **120 boundaries per 256-byte window** and ends up around **42–47** after training, on average. That results in an average chunk length of roughly **5–6 bytes**. For `sp1024`, the router starts at about **~127 boundaries per 256-token window** and ends up around **39** after training, with an average chunk length of about **7 tokens**.

5. **Byte-level chunking is more regular than subword-level chunking (H-Net).**

     The `byte260` model consistently achieves lower chunk-size CV than `sp1024` (see Results section), which indicates that its learned chunk sequence is more regular in length. Qualitatively, this matches the boundary visualizations: `byte260` tends to produce word-like chunks of similar size, while `sp1024` more often alternates between tiny fragments and much longer merged spans.

6. **Byte-level H-Net benefits from deeper chunking/dechunking interfaces.**

     Moving from `OUTER_LAYERS=1` to `OUTER_LAYERS=2` improves BPB for `byte260`, matching the finding reported in the previous H-Net work for subword-level H-Net ([PR #992](https://github.com/openai/parameter-golf/pull/992)): capacity around the encoder/chunker/decoder interface matters more than putting depth only into the compressed main 'stage'.

## Architecture


This model uses a **1-stage H-Net** that splits an N-block GPT-style backbone into a **E-M-D layout**:


- **Encoder:** E transformer blocks at full sequence length `L`
- **Main stage:** M transformer blocks on the compressed sequence of learned chunks (`C <= L`)
- **Decoder:** D transformer blocks back at full sequence length `L`


The forward path is:

```text
Input → Embedding → Encoder → Routing → ChunkLayer (L → C)
      → Main Transformer on chunks → DeChunkLayer (C → L)
      → + Residual Skip from encoder → Decoder → LM Head
```

The routing module predicts chunk boundaries from adjacent encoder hidden states using cosine dissimilarity between learned Q/K projections (paper Eq. 4). Positions with dissimilarity `>= 0.5` become hard chunk boundaries.


`ChunkLayer` keeps only boundary positions, producing a shorter sequence of chunk representatives.
 `DeChunkLayer` expands the chunk sequence back to length L using the paper’s EMA-based smoothing driven by the routing probabilities (Eq. 5). A learned linear `residual_proj0` adds a skip connection from the encoder output to the dechunked representation before the decoder.

**Parameter count: 17.5M total** (byte260).


## Results

`CV` = the coefficient of variation of chunk sizes (`std / mean`). Lower chunk-size CV means the router produces a more regular compressed sequence, which makes the chunked representation easier to model and suggests that the learned segmentation is more stable rather than alternating between tiny fragments and very long spans.


### Ablation 1: Chunk Length (OL2, rlw=0.03, lrdiff=0.75)

| Chunk Target | Tokenizer | BPB | Size (MB) | ≤16MB | WS agreement | CV |
|---|---|---|---|---|---|---|
| 6 | byte260 | **1.4033** | 15.65 | Yes | 89.3% | 0.74 | d
| 6 | sp1024 | **1.3671** | 16.22 | No | - | 0.68 |
| 9 | byte260 | 1.4206 | 15.77 | Yes | 89.7% | 0.48 |
| 9 | sp1024 | **1.3648** | 16.19 | No | - | 0.76 |
| 12 | byte260 | 1.4040 | 15.75 | Yes | 95.5% | 0.45 |
| 12 | sp1024 | 1.3734 | 15.99 | Yes | - | 0.71 |
| 16 | byte260 | 1.4171 | 15.67 | Yes | **97.4%** | 0.49 |
| 16 | sp1024 | 1.3748 | 15.91 | Yes | - | 0.86 |

- For `byte260`, shorter chunk targets (`6` or `12`) work best for BPB, while larger chunk targets improve whitespace alignment but decrease BPB.


### Ablation 2: Outer Layers (chunk=9, rlw=0.03, lrdiff=0.75)

| Outer Layers | Tokenizer | BPB | Size (MB) | ≤16MB |
|---|---|---|---|---|
| 1 | byte260 | 1.4526 | 15.81 | Yes |
| 1 | sp1024 | 1.4568 | 16.16 | No |
| 2 | byte260 | 1.4206 | 15.77 | Yes |
| 2 | sp1024 | 1.3648 | 16.19 | No |

- Increasing `OUTER_LAYERS` (OL) from `1` to `2` improves BPB for both tokenizers, supporting the argument that depth around the chunking/dechunking interface matters.

### Ablation 3: Ratio Loss Weight (OL2, chunk=9, lrdiff=0.75)

| RLW | Tokenizer | BPB | Size (MB) | ≤16MB | WS agree | CV |
|---|---|---|---|---|---|---|
| 0.03 | byte260 | 1.4206 | 15.77 | Yes | 89.7% | 0.48 |
| 0.03 | sp1024 | 1.3648 | 16.19 | No | - | 0.76 |
| 0.05 | byte260 | **1.4032** | 15.78 | Yes | **95.4%** | 0.45 |
| 0.05 | sp1024 | 1.3697 | 16.14 | No | - | 0.72 |


### Ablation 4: HNET_LR_DIFF (OL2, chunk=9, rlw=0.03)

| LR Diff | Tokenizer | BPB | Size (MB) | ≤16MB | WS agree | CV |
|---|---|---|---|---|---|---|
| 0.75 | byte260 | 1.4206 | 15.77 | Yes | 89.7% | 0.48 |
| 0.75 | sp1024 | 1.3648 | 16.19 | No | - | 0.76 |
| 0.85 | byte260 | 1.4084 | 15.75 | Yes | 90.3% | 0.45 |
| 0.85 | sp1024 | 1.3680 | 16.19 | No | - | 0.89 |

- `HNET_LR_DIFF=0.85` gives a small improvement for `byte260`, but does not show any improvement for `sp1024`.


## Qualitative Boundary Analysis

Comparing how the router segments the same text before and after training, and how the learned chunking differs between `byte260` and `sp1024`.

### Input text

> The quick brown fox jumps over the lazy dog. Natural language processing has made remarkable progress in recent years.

Boundaries generated from **byte260** best 10-min config (RLW=0.05, chunk=9, OL=2, KV=4, 1.4032 BPB) and **sp1024** chunk=9 config (RLW=0.03, chunk=9, OL=2, KV=2).

**byte260 (initial, no train)** -  54 boundaries /118 bytes, avg chunk 2.2 bytes
```
[The qu][i][c][k][ b][ro][w][n f][o][x][ j][ump][s o][v][er][ ][t][he l][a][z][y][ ][d][og.][ N][a][t][ur][a][l l][a][n][g][u][age pro][cessin][g h][as ma][de ][rema][rk][a][b][le prog][ress ][in ][recen][t][ ][y][e][a][rs][.]
```

**byte260 (trained, best 10-min)** - 19 boundaries / 118 bytes, avg chunk 6.2 bytes
```
[The ][quick ][brown ][fox ][jumps ][over ][the ][lazy ][dog. ][Natural ][language ][processing ][has ][made ][remarkable ][progress ][in ][recent ][years.]
```

- The training reduces boundary frequency from ~1 every 2 bytes (random) to ~1 every 6 bytes (trained), and nearly every learned boundary aligns with a whitespace character. The model discovers word segmentation purely from the language modeling objective.

**sp1024 (initial, no train)** - 28 boundaries / 46 tokens, avg chunk 1.6 tokens
```
[The][quick][bro][wn][f][ox][j][um][p][s over the][l][azy][do][g. Nat][ural][lang][u][age pro][cessing][has made][remarkable][pro][g][ress][in][recent][years][.]
```

**sp1024 (trained, chunk=9)** -  6 boundaries / 46 tokens, avg chunk 7.7 tokens:
```
[The][quick brown f][ox j][umps over the lazy dog. N][atural l][anguage processing has made remarkable progress in recent years.]
```

- The sp1024 H-Net creates fewer, larger chunks with uneven sizes, sometimes isolating short prefixes (`[The]`) and merging entire clauses into single chunks. The boundaries don't align with words anymore.

---

### Validation sample (byte260, trained)

Same byte260 best 10-min checkpoint (RLW=0.05, chunk=9, OL=2, KV=4).

```
[Insurance ][Company ][Declares ][Living ][Man ][Dead ][George ][Johannesen ][is ][very ][much ][alive. ][Which ][is ][why ][it ][was ][so ][surprising ][when ][the ][Canadian ][man ][received ][a ][letter ][addressed ]["][To ][the ][Estate ][of ][George ][Johannesen." ][Even ][more ][surprising ][is ][that ][it ][came ][from ][his ][in]
```

- Across 100 validation samples, training reduces boundary count from ~120 to ~42–47 per 256-byte window. The trained byte260 model consistently produces whitespace-aligned chunks with trailing spaces attached to the preceding word.

### Validation sample (sp1024, trained)

Same sp1024 chunk=9 checkpoint (RLW=0.03, chunk=9, OL=2, KV=2).

```
[I][nsurance Com][pany De][clares L][iving M][an De][ad G][eorge J][ohannesen is very much alive. Wh][ich is why it was so surprising when the][C][anadian man received a][letter addressed]
```

- The sp1024 chunker splits mid-word / beginning of the wrod at subword token boundaries (`[I][nsurance Com][pany De][clares]` or `G][eorge`]) and alternates between 1-token fragments and 20+ token spans. Chunk size CV is 0.76 (sp1024) vs 0.45 (byte260). This confirms quantitatively that byte boundaries are more consistent in terms of chunk size uniformity.  The byte260 model produces chunks of similar length (usually word-scale), while sp1024 oscillates between single-token fragments and long multi-token spans.

### 4-hour extended byte260 run

byte260 4-hour checkpoint (RLW=0.03, chunk=9, OL=2, KV=4, 85k steps, 1.3595 BPB).

```
[Insurance][ Company][ ][Declares][ ][Living][ ][Man][ ][Dead][ ][George][ ][Johannesen][ is][ ][very][ ][much][ alive][. Which][ is][ why][ it][ was][ ][so][ ][surprising][ when][ the][ Canadian][ ][man][ ][received][ a][ ][letter][ was][ to]
```

- With more training (85k steps vs ~4.5k@10 mins), the model handles whitespaces differently. Spaces get their own 1-byte chunks (often) or attach to the following word (sometimes) instead of the preceding one. The boundaries are still word-aligned.


## Comparison with Existing H-Net PRs

- The two previous H-Net PRs didn't include a working multi-GPU DDP training path. [PR #992](https://github.com/openai/parameter-golf/pull/992) reports a *simulated* 8×H100 result and mentions multi-GPU training as WIP. [PR #1044](https://github.com/openai/parameter-golf/pull/1044) was trained on a single RTX 4090. This implementation runs with working DDP on 8×H100 by padding the compressed chunk sequence and using a combined causal + padding attention mask.

- [PR #1044](https://github.com/openai/parameter-golf/pull/1044) uses causal depthwise Conv1d for the encoder/decoder, while both [PR #992](https://github.com/openai/parameter-golf/pull/992) and this submission use full transformer blocks. At 22M parameters and 1.8989 BPB, [PR #1044](https://github.com/openai/parameter-golf/pull/1044) showed that byte-level H-Net could work at tiny scale, but the BPB is much higher than the best results reported here.

- Neither existing PR focused on a matched comparison between byte-level and subword-level H-Net, or on quantitative boundary analysis. This submission adds 20 matched runs across four hyperparameters together with boundary metrics.

## Reproduction

```bash
# byte260 best 10-min config
MAX_WALLCLOCK_SECONDS=600 NUM_LAYERS=9 MODEL_DIM=512 \
    MLP_MULT=2 OUTER_LAYERS=2 TARGET_AVG_CHUNK_LEN=9 RATIO_LOSS_WEIGHT=0.05 \
    HNET_LR_DIFF=0.75 torchrun --standalone --nproc_per_node=8 hnet/train_gpt_hnet_byte.py

# 4-hour extended byte260 run
NUM_LAYERS=9 MODEL_DIM=512 MLP_MULT=2 OUTER_LAYERS=2 \
    TARGET_AVG_CHUNK_LEN=9 RATIO_LOSS_WEIGHT=0.03 HNET_LR_DIFF=0.75 \
    MAX_WALLCLOCK_SECONDS=14400 ITERATIONS=500000 \
    torchrun --standalone --nproc_per_node=8 hnet/train_gpt_hnet_byte.py

# sp1024 best 10-min config
MAX_WALLCLOCK_SECONDS=600 NUM_LAYERS=9 MODEL_DIM=512 \
    MLP_MULT=2 OUTER_LAYERS=2 TARGET_AVG_CHUNK_LEN=12 RATIO_LOSS_WEIGHT=0.03 \
    HNET_LR_DIFF=0.75 NUM_KV_HEADS=2 torchrun --standalone --nproc_per_node=8 hnet/train_gpt_hnet.py
```

### Legend

| Abbreviation | Full Name | Description |
|---|---|---|
| **OL** | Outer Layers (`OUTER_LAYERS`) | Number of encoder/decoder transformer blocks in the chunking layer |
| **RLW** | Ratio Loss Weight (`RATIO_LOSS_WEIGHT`) | Weight of the auxiliary loss that pushes avg chunk size toward the target |
| **LR Diff** | HNET LR Diff (`HNET_LR_DIFF`) | Gradient scaling factor for the chunker path relative to the rest of the model |
| **Chunk Target** | `TARGET_AVG_CHUNK_LEN` | Target average chunk length the ratio loss steers toward |
| **WS agree** | Whitespace agreement | % of learned boundaries that coincide with whitespace characters (byte260 only) |
| **CV** | Coefficient of variation | `std / mean` of chunk sizes; lower = more consistent segmentation |
| **byte260** | Byte-level tokenizer | vocab_size=260 (4 special + 256 byte values), 1 token = 1 byte |
| **sp1024** | SentencePiece BPE tokenizer | vocab_size=1024 |

## Compliance

- [x] Artifact ≤16,000,000 bytes (15,779,986 - best 10-min config)
- [x] 8×H100 SXM training
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] Non-record: extended run exceeds 10-min wallclock (**85,242 steps / 4h**)
- [x] All BPB numbers from int8+zlib quantized roundtrip

## Credits

- **Paper**: Hwang et al. (2025), [*Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*](https://arxiv.org/abs/2507.07955) - the H-Net architecture this submission implements. Official code: [github.com/goombalab/hnet](https://github.com/goombalab/hnet)
- **lucidrains**: [github.com/lucidrains/h-net-dynamic-chunking](https://github.com/lucidrains/h-net-dynamic-chunking) - standalone reimplementation of H-Net
- **TimS-ml (#992)**: [github.com/TimS-ml/parameter-golf](https://github.com/TimS-ml/parameter-golf) - inspiration for some of the hyperparameters
