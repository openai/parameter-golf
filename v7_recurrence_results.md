# v7 Recurrence Quantization Fix Results

**Date:** 2026-04-26/27  
**Branch:** shikhar  
**Baseline:** v6 (paired-head Muon, no recurrence) — sliding BPB 1.1029

## Objective

Test two approaches to fix depth recurrence's quantization problem (layers 4,5 replayed twice cause INT6 quantization error compounding):

1. **INT8 for recurred layers** (`RECUR_INT8=1`): Quantize layers 4,5 at INT8 (127 clip range) instead of INT6 (31 clip range)
2. **Skip recurrence at eval** (`RECUR_SKIP_EVAL=1`): Train with 13 virtual layers (recurrence active), evaluate with 11 physical layers (recurrence disabled)

## Configuration

All runs share:
- Seed: 1337
- `RECUR_LAYERS="4,5"` (depth recurrence on layers 4,5, activated at 50% wallclock ~300s)
- Paired-head Muon enabled
- `TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096`
- `SWA_WINDOW_SIZE=256`, `SWA_FULL_ATTN_LAYERS=5`
- `BIGRAM_VOCAB_SIZE=3072`, `BIGRAM_DIM=112`
- `WARMDOWN_ITERS=4000`
- GPTQ with autoregressive calibration (64 seqs x 2048 tokens, temp=0.8)
- Target size: 15.9MB
- Sliding window eval stride: 64

## Results Summary

| Run | Config | Pre-quant BPB | Sliding BPB | Quant Gap | Pruning | File Size | Verdict |
|-----|--------|--------------|-------------|-----------|---------|-----------|---------|
| **v6** | No recurrence (baseline) | 1.0986 | **1.1029** | 0.0043 | ~2% | ~15.5MB | **Best** |
| **v7a** | INT8 recurred layers | 1.0888 | 1.2263 | 0.1375 | 9.3% (2.4M/26M) | 16.54MB | Failed |
| **v7b** | Skip recurrence at eval | 1.0874 | 1.1961 | 0.1087 | None needed | 16.17MB | Failed |
| **v7c** | INT8 + skip recurrence | 1.0886 | 1.2251 | 0.1365 | 9.2% (2.4M/26M) | 16.54MB | Failed |

## Detailed Run Metrics

### v7a — INT8 for Recurred Layers (`RECUR_INT8=1`)

| Phase | Metric | Value |
|-------|--------|-------|
| Training | Steps completed | 5212/20000 |
| Training | Wallclock | 600s (10min cap) |
| Training | Step avg | 115.14ms |
| Training | Recurrence activated | Step 2935 @ 300s |
| Training | SWA start | Step 4450 |
| Training | Late QAT enabled | Step 4628 (scale 0.15) |
| Training | Mid-train val BPB (step 4000) | 1.1325 |
| Training | Final val BPB (step 5212) | 1.0898 |
| Post-EMA | Pre-quant BPB | **1.0888** |
| Quantization | INT8 override | Layers {4, 5} |
| Quantization | Unpruned size | 16.63MB |
| Quantization | Pruning | 2,423,219/25,952,256 weights (9.3%) |
| Quantization | Final compressed size | 16,538,546 bytes |
| Quantization | Total submission size | 16,672,225 bytes |
| Eval | Roundtrip BPB | 1.2339 |
| Eval | **Sliding window BPB** | **1.2263** |
| Eval | Sliding eval time | 151,961ms |

Training curve:
```
step     train_loss  val_bpb   time
0        -           3.4832    0s
500      3.3141      -         51s
1000     3.0185      -         102s
2000     3.0814      -         204s
2935     [recurrence activated]  300s
3000     2.9144      -         342s
4000     2.9303      1.1325    458s
5000     2.7795      -         575s
5212     -           1.0898    600s (stopped)
```

### v7b — Skip Recurrence at Eval (`RECUR_SKIP_EVAL=1`)

| Phase | Metric | Value |
|-------|--------|-------|
| Training | Steps completed | 5406/20000 |
| Training | Wallclock | 600s (10min cap) |
| Training | Step avg | 111.00ms |
| Training | Recurrence activated | Step 2929 @ 300s |
| Training | SWA start | Step 4700 |
| Training | Late QAT enabled | Step 4844 (scale 0.15) |
| Training | Mid-train val BPB (step 4000) | 1.1388 |
| Training | Final val BPB (step 5406) | 1.0886 |
| Post-EMA | Pre-quant BPB | **1.0874** |
| Quantization | All INT6 (no override) | - |
| Quantization | Unpruned size | 15.55MB |
| Quantization | Pruning | **None needed** (already fits) |
| Quantization | Final compressed size | 16,169,729 bytes |
| Quantization | Total submission size | 16,303,408 bytes |
| Eval | Recurrence disabled | Yes (`recur_skip_eval=1`) |
| Eval | Roundtrip BPB | 1.2022 |
| Eval | **Sliding window BPB** | **1.1961** |
| Eval | Sliding eval time | 126,725ms |

Training curve:
```
step     train_loss  val_bpb   time
0        -           3.4832    0s
500      3.3067      -         51s
1000     3.0239      -         102s
2000     3.0806      -         205s
2929     [recurrence activated]  300s
3000     2.9236      -         320s
4000     2.9416      1.1388    436s
5000     2.7964      -         552s
5406     -           1.0886    600s (stopped)
```

### v7c — Both INT8 + Skip Recurrence (`RECUR_INT8=1, RECUR_SKIP_EVAL=1`)

| Phase | Metric | Value |
|-------|--------|-------|
| Training | Steps completed | 5410/20000 |
| Training | Wallclock | 600s (10min cap) |
| Training | Step avg | 110.92ms |
| Training | Recurrence activated | Step 2930 @ 300s |
| Training | SWA start | Step 4700 |
| Training | Late QAT enabled | Step 4848 (scale 0.15) |
| Training | Mid-train val BPB (step 4000) | 1.1398 |
| Training | Final val BPB (step 5410) | 1.0897 |
| Post-EMA | Pre-quant BPB | **1.0886** |
| Quantization | INT8 override | Layers {4, 5} |
| Quantization | Unpruned size | 16.63MB |
| Quantization | Pruning | 2,394,896/25,952,256 weights (9.2%) |
| Quantization | Final compressed size | 16,538,571 bytes |
| Quantization | Total submission size | 16,672,250 bytes |
| Eval | Recurrence disabled | Yes (`recur_skip_eval=1`) |
| Eval | Roundtrip BPB | 1.2332 |
| Eval | **Sliding window BPB** | **1.2251** |
| Eval | Sliding eval time | 126,953ms |

Training curve:
```
step     train_loss  val_bpb   time
0        -           3.4832    0s
500      3.3032      -         51s
1000     3.0218      -         102s
2000     3.0853      -         205s
2930     [recurrence activated]  300s
3000     2.9247      -         319s
4000     2.9484      1.1398    435s
5000     2.7953      -         552s
5410     -           1.0897    600s (stopped)
```

## Analysis

### Recurrence helps pre-quant but destroys post-quant

All three recurrence runs achieved better pre-quant BPB (~1.0874-1.0888) vs v6 (1.0986), confirming recurrence genuinely improves the model by ~0.01 BPB. However, no tested approach could preserve this gain through quantization.

### Why each approach failed

**v7a (INT8 recurred layers):** INT8 uses 8 bits per weight vs INT6's 6 bits. For layers 4,5 (6 weight matrices each = 12 matrices total), this increased total model size from ~15.5MB to ~16.6MB. To fit the 15.9MB budget, 9.3% of all weights had to be pruned (zeroed out), destroying far more quality than the extra 2 bits of precision saved. The quant gap exploded to 0.1375 BPB.

**v7b (Skip recurrence at eval):** The model trained with 13 virtual layers (layers 4,5 replayed) but was evaluated with only 11 physical layers. The model learned features that depend on the second pass through layers 4,5 — removing recurrence at eval is equivalent to removing 2 layers from a trained network. The 0.1087 BPB gap shows the model cannot simply ignore the recurrence it was trained with. Notably, this had no pruning penalty (15.55MB fits under 15.9MB), so the entire gap is from the architectural mismatch.

**v7c (Both combined):** Worst of both worlds. INT8's size penalty forced heavy pruning (9.2%), AND the model lost its recurrence at eval. The results are comparable to v7a alone (1.2251 vs 1.2263), confirming the skip-eval adds negligible benefit on top of the pruning damage.

### Key insight: size budget is the binding constraint

The 15.9MB target is extremely tight. Any approach that increases per-weight bit count (INT8 vs INT6) requires proportionally more pruning, which overwhelms any precision benefit. The only way to use higher-precision quantization for recurred layers would be to:
- Increase the size budget (not allowed by competition rules)
- Reduce model size elsewhere to compensate (e.g., fewer layers or smaller dimensions)
- Find a quantization method that doesn't increase storage size

## Historical Run Comparison

| Run | Description | Pre-quant BPB | Sliding BPB | Quant Gap |
|-----|-------------|--------------|-------------|-----------|
| v1 | Baseline (no recurrence) | 1.0985 | 1.1049 | 0.0064 |
| v3 | Recurrence (unmodified INT6) | 1.0949 | 1.1792 | 0.0843 |
| v4 | TTT (redundant, all-ranks) | 1.0988 | 1.1039 / TTT:1.1932 | - |
| v5 | TTT (distributed fix) | ~same | 1.1041 / TTT:1.1497 | - |
| **v6** | **Paired-head Muon, no recur** | **1.0986** | **1.1029** | **0.0043** |
| v7a | Recur + INT8 recurred layers | 1.0888 | 1.2263 | 0.1375 |
| v7b | Recur + skip recur at eval | 1.0874 | 1.1961 | 0.1087 |
| v7c | Recur + INT8 + skip eval | 1.0886 | 1.2251 | 0.1365 |

## Revised Conclusion (2026-04-27)

The v7 result should **not** be read as "depth recurrence is fundamentally incompatible with INT6." Merged records prove the opposite:

| Record | Stack | Result |
|--------|-------|--------|
| 2026-04-03 `MuonEqR_DepthRecurrence_WD090_AllInt6` | SP4096, layers 4,5 recurrence, all matrix INT6 | 1.0912 sliding BPB |
| 2026-04-05 `SP8192_GPTQ-Embeddings_SDClip_Loop45x2` | SP8192, layers 4,5 looped, matrix INT6 + embedding INT8 | 1.08563 sliding BPB |
| 2026-04-09 `SP8192_3LayerRecur_ParResid_QK525_LegalTTT` | SP8192, layers 3,4,5 recurrence, parallel residuals, QK 5.25, legal TTT | 1.0827 sliding / 1.0810 TTT |

The precise conclusion is:

> Our current v7 recurrence path does not reproduce the merged recurrence path. The likely failure is not recurrence alone, but recurrence combined with mismatched quantization calibration, compromised skip topology, INT8/pruning pressure, and eval-mode variants that do not match the training graph.

**v6 remains our best local artifact at 1.1029 BPB**, but recurrence is still a viable route to pre-Scylla SOTA if implemented like the merged PRs.

## Implementation Plan: Recover Merged-PR Recurrence

This section is intended as the implementation checklist for the remote machine.

### 0. Clarify what v7b did and did not test

v7b used all-INT6 matrix weights:

```bash
RECUR_INT8=0
RECUR_SKIP_EVAL=1
```

But v7b did **not** test all-INT6 recurrent inference. It tested:

```text
train with recurrence -> quantize as INT6 -> disable recurrence at eval
```

That is useful as an architectural mismatch test. It is not the clean SOTA path.

The closest local all-INT6 recurrent-inference result is v3:

```text
v3 recurrence, unmodified INT6:
pre-quant:     1.0949
INT6 sliding:  1.1792
gap:          +0.0843
```

So naive recurrence quantization is bad locally, but merged PRs show it can be made to work.

### 1. Port the merged recurrence topology exactly

Do not keep the current v7 recurrence topology as the reference implementation. The merged PR #1394-style schedule is:

```python
num_loops = 2
loop_start = 4
loop_end = 5

loop_seg = [4, 5]
all_indices = [0, 1, 2, 3]
for _ in range(num_loops + 1):
    all_indices.extend(loop_seg)
all_indices.extend([6, 7, 8, 9, 10])

# all_indices:
# [0,1,2,3,4,5,4,5,4,5,6,7,8,9,10]

encoder_indices = [0,1,2,3,4,5,4]
decoder_indices = [5,4,5,6,7,8,9,10]
```

For PR #1493-style 3-layer recurrence:

```python
num_loops = 2
loop_start = 3
loop_end = 5
enable_looping_at = 0.35

encoder_indices = [0,1,2,3,4,5,3,4]
decoder_indices = [5,3,4,5,6,7,8,9,10]
```

Required model changes:

```python
self.looping_active = False
self.encoder_indices = ...
self.decoder_indices = ...
self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim))
self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim))
```

Forward should use:

```python
enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, num_layers)
```

Critical difference from current v7:

- Do **not** size skip weights from only the physical 11-layer model.
- Do **not** reuse the last physical skip weight with `min(j, self.num_skip_weights - 1)`.
- Use sigmoid skip gates as in merged records.

### 2. Add loop warmup/prewarm

Merged PRs do a normal warmup, then briefly flip recurrence on and warm the compiled graph, then restore the original model and optimizer state.

Implementation shape:

```python
initial_model_state = deepcopy(base_model.state_dict())
initial_optimizer_states = deepcopy(optimizers)

# flat warmup
for step in range(warmup_steps):
    step_fn(...)

# recurrent graph warmup
base_model.looping_active = True
for step in range(warmup_steps):
    step_fn(...)
base_model.looping_active = False

# restore
base_model.load_state_dict(initial_model_state)
optimizers.load_state_dict(initial_optimizer_states)
```

This is not a quality trick; it prevents compile/runtime shape issues from contaminating the timed training section.

### 3. Activate recurrence by training fraction

Use a fraction of effective training time, not a hardcoded step unless reproducing a specific record.

PR #1394-style:

```bash
ENABLE_LOOPING_AT=0.5
LOOP_START=4
LOOP_END=5
NUM_LOOPS=2
```

PR #1493-style:

```bash
ENABLE_LOOPING_AT=0.35
LOOP_START=3
LOOP_END=5
NUM_LOOPS=2
```

Activation log should print the exact encoder/decoder lists:

```text
layer_loop:enabled step:<step> frac:<frac> encoder:[...] decoder:[...]
```

### 4. Make GPTQ recurrence-aware

This is the highest-priority fix.

Merged PR #1394 collects GPTQ Hessians from the actual trained model while recurrence is active. Therefore hooks on layers 4/5 fire multiple times and accumulate all recurrent uses.

Conceptually, for a reused layer:

```text
H_layer4 = X_pass1.T @ X_pass1
         + X_pass2.T @ X_pass2
         + X_pass3.T @ X_pass3
```

Current v7 risk: the separate `_HessianGPT` path is structurally flat unless explicitly given the same recurrence schedule. If GPTQ sees only the flat pass, it optimizes the wrong quantization objective.

Implementation options:

1. Best: collect Hessians directly from the trained recurrent model after EMA, with `looping_active=True`.
2. Acceptable: make `_HessianGPT` implement the same `loop_start/loop_end/num_loops` schedule, set `looping_active=True`, and ensure hooks accumulate into the same physical layer names for repeated uses.

Pass condition:

```text
GPTQ should report Hessians for all matrix params.
Repeated physical layers must receive accumulated stats from every virtual use.
```

If recurrent post-quant still blows up after this, then the failure is real quantization sensitivity. Before this, the measurement is not conclusive.

### 5. Use the merged quantization recipe, not v7 INT8 override

Do **not** use `RECUR_INT8=1` as the next path. v7a/v7c showed whole-layer INT8 creates too much size pressure and forces destructive pruning.

Use:

```text
attention/MLP matrices: INT6, SDClip k=12.85
token embedding:        INT8, SDClip k=20.0
compression:            byte-shuffle + Brotli-11
selective pruning:      off
```

If artifact is too large:

1. Increase `MATRIX_CLIP_SIGMAS` slightly, e.g. `13.5`, then `14.0`.
2. Increase WD if needed, e.g. `MUON_WD=0.090` or `0.095`.
3. Reduce auxiliary tables such as BigramHash only if still over.
4. Do **not** prune millions of weights as the primary fit mechanism.

Target must be decimal:

```text
total submission size < 16,000,000 bytes
```

The v7 artifacts above 16.3M/16.6M bytes should be treated as invalid for final leaderboard purposes.

### 6. Explicitly enable recurrence for quantized eval

After deserializing/dequantizing the eval model:

```python
eval_model.looping_active = h.num_loops > 0
```

or for the current naming:

```python
eval_model.recur_active = bool(args.recur_layers) and not args.recur_skip_eval
```

Log it:

```text
eval: recurrence active=True encoder:[...] decoder:[...]
```

Keep skip-eval as an ablation only. It is not a candidate SOTA path.

### 7. Reproduce merged records in order

#### Stage A: PR #1394-style reproduction

Goal: recover the clean SP8192 recurrence baseline.

Suggested env:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp8192
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
TRAIN_SEQ_LEN=2048
EVAL_SEQ_LEN=2048
MLP_MULT=4.0
MUON_ROW_NORMALIZE=1
QK_GAIN_INIT=4.0
MUON_WD=0.085
EMBED_WD=0.085
EMA_DECAY=0.997
MATRIX_BITS=6
EMBED_BITS=8
MATRIX_CLIP_SIGMAS=12.85
EMBED_CLIP_SIGMAS=20.0
NUM_LOOPS=2
LOOP_START=4
LOOP_END=5
ENABLE_LOOPING_AT=0.5
SWA_WINDOW_SIZE=0
```

Expected target if implementation matches:

```text
pre-quant:           ~1.089-1.091
quantized roundtrip: ~1.101-1.104
sliding BPB:         ~1.084-1.087
artifact:            <16,000,000 bytes
```

If sliding is much worse than `1.09`, do not add more techniques. Debug recurrence/GPTQ/eval parity first.

#### Stage B: Add parallel residual and related zero-byte fixes

Once Stage A is close, add:

```bash
PARALLEL_RESIDUAL_START=7
```

Merged records report parallel residuals mostly tightening the quantization gap rather than massively improving pre-quant.

Optional follow-up:

```bash
HESSIAN_CLIP_LAMBDA=0.175
```

This was non-record but directionally useful in the 2026-04-06 result.

#### Stage C: PR #1493-style final stack

Move from 2-layer recurrence to 3-layer recurrence:

```bash
LOOP_START=3
LOOP_END=5
NUM_LOOPS=2
ENABLE_LOOPING_AT=0.35
PARALLEL_RESIDUAL_START=7
QK_GAIN_INIT=5.25
MUON_WD=0.095
MATRIX_LR=0.022
EMA_DECAY=0.9965
WARMDOWN_FRAC=0.72
```

Targets:

```text
sliding BPB: ~1.0826-1.0830
TTT BPB:     ~1.0808-1.0812
```

#### Stage D: Legal score-first TTT

Only add TTT after sliding is near `1.083`.

Merged PR #1493 TTT:

```bash
TTT_ENABLED=1
TTT_LR=0.005
TTT_EPOCHS=3
TTT_MOMENTUM=0.9
TTT_CHUNK_TOKENS=32768
```

Protocol:

```text
for each 32K-token chunk:
  1. score all sliding windows under no_grad
  2. train on already-scored chunk with SGD
  3. move to next chunk
```

Do not use TTT to hide a bad sliding model. It should be the final ~0.002 BPB improvement, not the main fix.

### 8. Current local techniques to reintroduce only after reproduction

Do not stack local innovations before reproducing Stage A.

Reintroduce one at a time:

1. Paired-head Muon, because v6 improved quant gap by ~0.002 BPB with no pre-quant gain.
2. SWA training attention, only if it does not disrupt recurrence/GPTQ parity.
3. BigramHash sizing changes.
4. Seq length 4096.

Each must be compared against the reproduced PR #1394-style baseline, not against v6.

### 9. Failure diagnostics

If pre-quant is good but post-quant explodes:

1. Verify eval recurrence is active after deserialize.
2. Verify GPTQ Hessian collection ran with recurrence active.
3. Verify skip weights/gates are sized from virtual encoder/decoder lengths.
4. Verify no pruning happened.
5. Verify artifact is under decimal 16,000,000 bytes without pruning.
6. Quantize/evaluate one group at a time:
   - layer 4 Q/K
   - layer 4 V/O
   - layer 4 MLP up/down
   - layer 5 Q/K
   - layer 5 V/O
   - layer 5 MLP up/down

If one group dominates the gap, tune SDClip or QAT only for that group. Do not move whole recurrent layers to INT8 first.

### 10. Key implementation principle

The recurrent model must be internally consistent:

```text
training graph == GPTQ calibration graph == quantized eval graph
```

v7 violated or weakened this principle through skip-eval variants, likely flat/mismatched Hessian collection, INT8/pruning pressure, and non-upstream skip topology.

The merged records succeeded by keeping that graph consistent and making quantization fit naturally under the byte budget.

## Artifacts

- Logs: `logs/v7a_recur_int8.txt`, `logs/v7b_recur_skipeval.txt`, `logs/v7c_recur_both.txt`
- Console logs: `v7a_recur_int8.log`, `v7b_recur_skipeval.log`, `v7c_recur_both.log`
- Run script: `run_v7_recur_tests.sh`
- HuggingFace: `shikhar007/parameter-golf-gram-ns/models/` and `logs/`
