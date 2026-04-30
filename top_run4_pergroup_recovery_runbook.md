# Run 4 Pergroup Recovery Runbook

Date: 2026-05-01

Goal: recover the Run 4 quality (`q_ttt = 1.05950377`) while making the
artifact valid under the 16,000,000 byte cap by compressing the Run 4 model
with PR #1855 / Run 5's `pergroup` compressor.

## Why this matters

Run 4 is the best single-seed result we have from the PR1851 hybrid path:

```text
Run 4 config: PR1851 + PR1855 9 hparams + wd_strong + GPTQ AR
pre   = 1.06330575
q     = 1.07238835
q_ttt = 1.05950377
size  = 16,140,607 B
```

The score is useful, but the artifact is invalid:

```text
cap                         = 16,000,000 B
Run 4 total                 = 16,140,607 B
over cap                    =    140,607 B
Run 4 quantized model blob  = 16,108,157 B
Run 4 compressed code       =     32,450 B
```

The problem is not code size. Even with zero code bytes, Run 4's brotli model
blob is still 108,157 B over cap.

PR #1855's `pergroup` compressor reports about 280 KB savings over plain
brotli on this family of models, so applying that compression mechanism to
Run 4 should make the same trained model fit without changing model quality.

## Important distinction

There are three possible artifact states on the remote machine.

### Case A: Run 4 `final_model.pt` still exists

This is the best case. Do **not** retrain.

Use the Run 4 FP checkpoint and re-run only:

1. GPTQ/LQER with the exact Run 4 hparams
2. PR #1855 per-group compression
3. q / q_ttt evaluation

This should preserve the trained weights. GPTQ calibration can still introduce
tiny variation, but this is much cheaper and cleaner than rerunning training.

### Case B: only Run 4 `final_model.int6.ptz` exists

This is harder. The Run 4 `.ptz` is one brotli-compressed `torch.save` blob.
PR #1855's `pergroup` compressor works best before the final monolithic
compression step because it groups tensor payloads by role and applies row
similarity sorting on hot tensors.

It may still be possible to:

1. decompress the brotli `.ptz`
2. load the quantized state dict
3. feed `quant_result` / `quant_meta` into `_serialize_pergroup`
4. re-evaluate

But this is more fragile than starting from `final_model.pt`. Prefer Case A.

### Case C: neither Run 4 checkpoint nor Run 4 `.ptz` exists

Then Run 4 must be rerun. If rerunning, use `ARTIFACT_DIR` so the files do
not get overwritten by later experiments.

## First check on the remote

Run this from the remote repo root:

```bash
pwd
git rev-parse --short HEAD
git status --short --branch

find . -maxdepth 4 \
  \( -name 'final_model.pt' \
  -o -name 'final_model.int6.ptz' \
  -o -name '*top_pr1855_hparams*' \
  -o -name '*run4*' \
  -o -name '*.pt' \
  -o -name '*.ptz' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
```

The Run 4 log names are:

```text
logs/top_pr1855_hparams_s42.txt
logs/top_pr1855_hparams_s42.stdout
```

The Run 4 default checkpoint paths, if not redirected with `ARTIFACT_DIR`, are:

```text
final_model.pt
final_model.int6.ptz
```

If multiple runs happened after Run 4, those root files may have been
overwritten. Check timestamps and file sizes carefully.

Run 4 logged:

```text
Serialized model: 135417533 bytes
Serialized model quantized+brotli: 16108157 bytes
Total submission size quantized+brotli: 16140607 bytes
```

So a likely Run 4 FP checkpoint should be around:

```text
final_model.pt ~135,417,533 B
```

## Preferred recovery path: reserialize Run 4 FP checkpoint

There is not currently a clean built-in `CKPT_PATH -> GPTQ -> pergroup -> TTT`
entrypoint for `train_top.py`. The existing `TTT_EVAL_ONLY=1` path skips
training and GPTQ, but it expects a pre-existing quantized artifact and does
not recompress an FP checkpoint.

Use one of these two implementation approaches.

### Approach 1: add a small requant entrypoint

Add a `RECOMPRESS_CKPT_PATH` / `OUT_PTZ` path to `train_top.py` or create a
small `requant_top.py` script by copying the structure of `requant_eval.py`.

The script should:

1. construct `GPT(h)`
2. `restore_fp32_params(model)`
3. load Run 4 `final_model.pt`
4. build `ValidationData`
5. call `serialize(h, model, Path("train_top.py").read_text(...))`
6. deserialize and run diagnostic `q`
7. run phased TTT

Use Run 4's model definition and hparams, but use the PR #1855 pergroup
serializer.

Concretely, the safe implementation is:

- start from `train_top.py` because this is the exact Run 4 graph
- port the pergroup compressor block from `train_top_1855.py`
- change `serialize` / `deserialize` to support `COMPRESSOR=pergroup`
- add `RECOMPRESS_CKPT_PATH`

The pergroup code in `train_top_1855.py` is around:

```text
_similarity_sort_l1
_lrzip_compress
_lrzip_decompress
_pack_tensor_payload
_unpack_tensor_payload
_group_key_for_tensor
_serialize_pergroup
_deserialize_pergroup
serialize(... if h.compressor == "pergroup" ...)
deserialize(... if h.compressor == "pergroup" ...)
```

Do not use `train_top_1855.py` blindly unless `load_state_dict(strict=True)`
confirms Run 4's checkpoint is graph-compatible. The Run 5 script is very close
but not guaranteed to be bit-identical to the Run 4 graph.

### Approach 2: rerun Run 4 with pergroup ported into `train_top.py`

If the checkpoint path is too risky, port pergroup into `train_top.py` and rerun
Run 4 training from scratch with `COMPRESSOR=pergroup`. This costs a full run,
but it tests the exact pipeline we would submit.

Use `ARTIFACT_DIR` to preserve files:

```bash
mkdir -p artifacts/run4_pergroup_s42
```

Then run with `ARTIFACT_DIR=artifacts/run4_pergroup_s42`.

## Run 4 hparams to preserve

These are the important Run 4 settings:

```bash
RUN_ID=top_pr1855_hparams_s42_pergroup SEED=42 \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
GPTQ_ALL_REDUCE=1 \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
EMBED_CLIP_SIGMAS=14.0 \
MLP_CLIP_SIGMAS=11.5 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
PHASED_TTT_PREFIX_DOCS=2500 \
COMPRESSOR=pergroup
```

Run 4 used `train_top.py`, not `train_top_1855.py`.

Run 4 used brotli only because `train_top.py` did not yet have pergroup.
The goal is to keep the Run 4 graph and hparams but change the compression
mechanism to PR #1855's pergroup path.

## Required system dependency

PR #1855's pergroup compressor shells out to `lrzip`.

Install it before running:

```bash
apt-get update
apt-get install -y lrzip
which lrzip
lrzip -V || true
```

If `apt-get` needs Ubuntu universe enabled:

```bash
add-apt-repository -y universe
apt-get update
apt-get install -y lrzip
```

## Expected success criteria

A successful recovery should show:

```text
COMPRESSOR=pergroup
Serialized model quantized+pergroup: < about 15.9M
Total submission size quantized+pergroup: < 16,000,000 bytes
quantized_ttt_phased val_bpb: near 1.05950
```

Good outcome:

```text
q_ttt <= 1.0597
total bytes < 16,000,000
```

Excellent outcome:

```text
q_ttt <= 1.0590
total bytes < 16,000,000
```

Record-candidate outcome:

```text
q_ttt <= ~1.0588 on seed 42
```

Even then, we need 3 seeds and the 0.005-nats acceptance evidence.

## Why not just submit Run 4 as-is?

Run 4 is invalid because:

```text
16,140,607 B > 16,000,000 B
```

The overage is in the quantized model blob, not the code:

```text
quantized model blob = 16,108,157 B
compressed code      =     32,450 B
```

Code minification cannot save enough. We need model compression savings.

## Why Run 5 did not solve this

Run 5 used PR #1855's full script and pergroup compressor with our
`wd_strong + AR` additions:

```text
Run 5 q_ttt = 1.06009
```

That is valid-size in expectation, but it is worse than Run 4 by:

```text
1.06009 - 1.05950377 = +0.00058623 BPB
```

So the best next attempt is not "use Run 5 result." It is:

```text
preserve Run 4 trained model quality
replace only Run 4's brotli serialization with PR #1855 pergroup serialization
```

## If rerunning Run 4 from scratch

Use `ARTIFACT_DIR` so the checkpoint survives:

```bash
mkdir -p artifacts/top_pr1855_hparams_s42_pergroup

ARTIFACT_DIR=artifacts/top_pr1855_hparams_s42_pergroup \
RUN_ID=top_pr1855_hparams_s42_pergroup SEED=42 \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
GPTQ_ALL_REDUCE=1 \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
EMBED_CLIP_SIGMAS=14.0 \
MLP_CLIP_SIGMAS=11.5 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
PHASED_TTT_PREFIX_DOCS=2500 \
COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_top.py
```

After the run:

```bash
ls -lh artifacts/top_pr1855_hparams_s42_pergroup/
rg -n "Serialized model|Total submission size|quantized_ttt_phased|val_bpb|pergroup" \
  artifacts/top_pr1855_hparams_s42_pergroup/*.txt logs/*.txt
```

Upload/preserve immediately if the result is good:

```bash
# Example only; adjust repo/path names.
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo_id = "shikhar007/parameter-golf-gram-ns"
run_id = "top_pr1855_hparams_s42_pergroup"
for local, remote in [
    ("artifacts/top_pr1855_hparams_s42_pergroup/final_model.pt", f"models/{run_id}.pt"),
    ("artifacts/top_pr1855_hparams_s42_pergroup/final_model.int6.ptz", f"models/{run_id}.int6.ptz"),
    ("artifacts/top_pr1855_hparams_s42_pergroup/top_pr1855_hparams_s42_pergroup.txt", f"logs/{run_id}.txt"),
]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo_id)
    print("uploaded", local, "->", remote)
PY
```

## Final recommendation

First try to recover from the existing Run 4 FP checkpoint. If that checkpoint
is gone, rerun Run 4 with pergroup ported into `train_top.py`.

Do not spend time on more PR1493-stack ideas until this is settled. Run 4 is
our best known quality point; the next move is compression recovery, not a new
architecture.
