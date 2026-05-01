# SemanticEngine — CareSSM + Live Episodic Memory

**Track:** track_10min_16mb  
**val_bpb:** 1.642868 (3-seed mean, std 0.023340)  
**artifact:** 13,554,222 / 16,000,000 bytes estimated contest-counted int6/LZMA payload, including 500 KB overhead  
**eval:** full 50k FineWeb validation docs, legal prequential packet-online cache

The raw bf16 runtime weight mirror is 44,600,064 bytes. That is not the submitted
artifact size; the submitted artifact uses the same int6/LZMA artifact accounting
used by the dim-384 headroom check.

## Architecture

**SemanticEngine** is a CareSSM trunk with live episodic memory. Unlike the transformer submissions, this is a pure SSM architecture whose memory substrate is active during both training and prequential eval.

### Named Components

| Name | Role |
|---|---|
| **SemanticEngine** | Overall system |
| **CareSSM** | Diagonal recurrent SSM trunk blocks |
| **ChaosSsm** | CPU SSM controller / scheduling plane |
| **Episodic memory** | CRCT evidence substrate + MultiSlotOuterModel + replay eviction pipeline |
| **SemanticOptimizer** | Muon with SSM-channel-coupled momentum beta |

### Dedicated Memory GPUs (8xH100)

On 8xH100, GPU 6 and GPU 7 are not train ranks. They own the memory substrate exclusively:

- **GPU 6 (packet-serving rank):** Builds low-latency episodic residual packets from the pre-recurrence stream and publishes them to train ranks without blocking the trunk step.
- **GPU 7 (maintenance rank):** Owns memory maintenance, slot refresh, and slot commits.

Train ranks never wait on a memory GPU. If no fresh packet is available, the trunk proceeds with a zero-residual failsafe.

### Training vs. Eval

During training, the trunk updates weights while the memory/controller stack generates evidence and maintains the cache.

During eval, the same memory substrate is live, but the run is **prequential**: each chunk is scored under the current memory state first, loss is accumulated, then the cache is updated from the just-scored tokens. The trunk never sees validation tokens before they are scored. The packet-online eval path raises if cache slot count changes before score accumulation.

## Results

| Seed | val_loss | val_bpb | Train steps | Train time | Eval time | Cache slots |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 4.070076 | 1.640762 | 1692 | 596.0s | 347.0s | 93,346 -> 139,998 |
| 1337 | 4.135631 | 1.667189 | 1692 | 594.1s | 349.5s | 89,776 -> 136,428 |
| 294924 | 4.020193 | 1.620653 | 1688 | 594.3s | 364.8s | 93,091 -> 139,743 |
| **Mean** | **4.075300** | **1.642868** | **1690.7** | **594.8s** | **353.8s** | |

All evals scored the full 50,000-doc validation set: 42,216,034 scored tokens and 151,080,645 raw bytes per seed. Each eval performed 3,348 episodic reads and 3,348 score-first episodic writes.

Artifact accounting: the public `artifact_bytes_estimate` is the contest-counted
compressed artifact estimate, `13,554,222` bytes against the decimal `16,000,000`
byte cap. The larger `raw_bf16_weight_bytes` value in `submission.json` is only
the uncompressed runtime state size used by the shared-memory weight mirror.

## Reproduction

```bash
# 1. Clone chaoscontrol and bootstrap the pod
git clone https://github.com/KenMalloy/chaoscontrol.git /workspace/chaoscontrol
HF_TOKEN=<token> bash /workspace/chaoscontrol/scripts/pod_bootstrap.sh

# 2. Run one seed
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# 3. Eval-only from a saved checkpoint
EVAL_ONLY=1 CHECKPOINT_PATH=/path/to/checkpoint.pt \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
