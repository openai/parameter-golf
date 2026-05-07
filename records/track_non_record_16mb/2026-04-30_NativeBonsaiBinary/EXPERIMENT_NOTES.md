# NativeBonsaiBinary Experiment Notes

These notes summarize the H100 handoff lessons that led to the final 4xH100 non-record candidate.

## Workspace State

The RunPod workspace was `/workspace/parameter-golf`, with custom handoff code under `h100_binary_handoff/`.

The official `openai/parameter-golf` repository files were restored around the handoff folder. SP8192 data was downloaded locally from the Hugging Face dataset repo `kevclark/parameter-golf`, not the official default `willdepueoai/parameter-golf` manifest:

- `data/datasets/fineweb10B_sp8192/`
- 80 train shards
- 1 validation shard
- tokenizer files in `data/tokenizers/`

Earlier tests were run on a 1xH100 pod. The final candidate was run on 4xH100. True 8xH100 behavior was not benchmarked for this submission.

## Binary Model Clarification

The model is binary for the main linear weights:

- Training keeps fp32 latent `weight_logits`.
- Forward pass uses `sign(weight_logits) * learned_group_scale`.
- The native packet stores signs as bits plus FP16 group scales.

It is not pure 1-bit for every tensor:

- embeddings remain high precision
- norms, scales, and control tensors remain high precision
- compute uses dense CUDA ops, not bit-packed popcount kernels

## Legal Constraints Considered

- Artifact limit is decimal `16,000,000` bytes, code plus compressed model.
- Use LZMA packet size as reported by `export_native_packet_size.py`.
- No validation data access during training.
- Test-time training, if used, must be score-first.
- Evaluation has its own 10-minute 8xH100 cap.

The final candidate disables TTT.

## Best Confirmed 1xH100 Results

Bad early run:

- `TRAIN_BATCH_TOKENS=524288`
- only 373 updates in 10 minutes
- `val_bpb=1.8968`
- diagnosis: update-starved; binary STE needs many optimizer updates

Corrected 65k run:

- `TRAIN_BATCH_TOKENS=65536`
- `GRAD_ACCUM_STEPS=4`
- `ITERATIONS=2100`
- train time `573.5s`
- `val_bpb=1.4713`
- simple legal TTT: `1.4692`

Best confirmed 32k run:

- `TRAIN_BATCH_TOKENS=32768`
- `GRAD_ACCUM_STEPS=1`
- `ITERATIONS=4200`
- train time `564.1s`
- step time `134ms`
- `val_bpb=1.4302`
- simple legal TTT: `1.4286`
- packet with then-current code: `14,839,523` bytes LZMA plus code

## Smaller Batch Diagnostics

Short speed sweep:

| Batch / accumulation | Approx updates per 20s | Approx tokens/s |
|----------------------|------------------------|-----------------|
| 65k, accum4 | 72 | 239k |
| 65k, accum2 | 80 | 266k |
| 65k, accum1 | 86 | 286k |
| 32k, accum1 | 145 | 245k |
| 16k, accum1 | 228 | 192k |
| 8k, accum1 | 287 | 121k |
| 4k, accum1 | 277 | 57k-60k |

Learning diagnostics:

| Diagnostic | Train time | val_bpb |
|------------|------------|---------|
| 32k, 400 updates | about 54s | 2.0398 |
| 16k, 800 updates, same token budget as 32k diagnostic | about 70s | 1.9469 |
| 8k, 1600 updates, same token budget | about 114s | 1.7830 |
| 16k, 1400 updates, time-matched to 8k | about 120s | 1.7095 |
| 4k, 1600 updates | about 111s | 1.9370 |

Interpretation:

- The 1xH100 small-batch knee appears near 16k.
- 4k is too small; kernel and step overhead dominate.
- 16k may beat 32k in a full 10-minute 1xH100 run, but a full 16k 10-minute run was not completed.

## Size Probes With Current Code

Measured with the current trainer and packet exporter:

| Shape | LZMA plus code |
|-------|----------------|
| 9L x 768 e254 | 14,774,095 bytes |
| 10L x 768 e254 | 15,919,019 bytes |
| 10L x 768 e320 | 17,068,759 bytes |

`10L x 768 e254` is legal but has only about 81KB margin. `10L x 768 e320` is illegal with this packet/export path.

## DDP / 8xH100 Status

DDP means one process per GPU. Each process handles a local slice of the batch, gradients are averaged across GPUs, and then all ranks apply the same optimizer update.

The trainer has an explicit all-reduce DDP path:

- `DDP=1`
- launched via `torchrun --standalone --nproc_per_node=N`
- rank/device from `LOCAL_RANK`
- global `TRAIN_BATCH_TOKENS` split into `local_train_batch_tokens`
- rank 0 logs, validates, and saves

This path was ultimately used on 4xH100 for the final candidate. It was not proven on true 8xH100 for this submission.

Important final fixes:

- DDP rank data offset: ranks train on distinct shards instead of duplicated minibatches.
- Synchronized wallclock stop: all ranks exit the training loop on the same step, avoiding DDP hangs before save.

## Recommended Future Tests

On 1xH100:

- Full 10-minute `16k` run with `TRAIN_BATCH_TOKENS=16384` and `GRAD_ACCUM_STEPS=1`.
- Estimated 6500-6900 updates in 600s.
- Candidate to beat the 32k 1xH100 run based on diagnostics.

On 8xH100:

- Make DDP smoke pass first.
- Compare global batch `32768` and `65536` for step time and short BPB diagnostics.
- Try global batch `16384` only if communication overhead is low.
- Avoid returning to global batch `524288`; it was update-starved.

## Discarded Runs

- `4xh100_5x1024_32k_9800it_20260430_223735`: `val_bpb=1.4736`; legal size and time, but trained before the DDP rank data offset fix.
- 9L x 768 attempts: default 1xH100 learning rates were unstable on this 4x DDP path; loss spiked into double digits early, so these runs were stopped.
