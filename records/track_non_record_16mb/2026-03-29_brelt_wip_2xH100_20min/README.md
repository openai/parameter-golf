# brelt on 2xh100 for 20 minutes

this package contains a non-record wip run of **brelt** and a same-budget run of the original challenge baseline.

the main brelt repository is here:

- https://github.com/guilhhotina/brelt

this folder is the frozen package for this specific run.

## architecture

brelt is a byte-level recurrent latent model.

the visible input is a byte stream.
a local encoder reads bytes and produces local states.
a learned segmentation module groups bytes into patches.
each patch is committed into a latent span state.
a recurrent global latent core mixes those span states.
a bridge projects the global latent states back into local space.
a local decoder produces byte predictions.

the exact `train_gpt.py` snapshot used for this run is included in this folder.

## components used in this run

- raw byte input reconstructed from the challenge dataset shards
- local byte encoder
- learned patch segmentation
- patch commit into span latents
- recurrent global latent mixing with shared depth
- bridge back to byte-local space
- byte-level decoder and byte-level training objective
- int8 + zlib export path
- rotation of weight matrices before quantization

## ideas used in the architecture

### byte-level latent modeling

brelt uses bytes as the causal interface and compresses local spans into latent states.
this part is closest to byte latent transformer style modeling.

reference:

- https://arxiv.org/abs/2412.09871

### shared recurrent depth

the global latent core reuses the same block recurrently instead of using a fully unique deep stack.
this part is closest to universal transformer style shared depth.

reference:

- https://arxiv.org/abs/1807.03819

### rate-distortion / mdl style pressure

the training setup tries to make the latent stream economically useful.
latent states should reduce byte prediction cost enough to justify their own modeling cost.
this is the idea behind the rate controller, latent-rate terms, and segmentation control.

### quantization as part of the training target

the final artifact is expected to survive int8 + zlib compression.
this run used rotation of weight matrices before quantization in the export path.

reference:

- https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

## run setup

- track: `non-record-16mb`
- hardware: `2xh100`
- wallclock: `1200s`
- dataset: `fineweb10B_sp1024`
- train shards: `80`
- validation: `fineweb_val_*`
- world size: `2`
- brelt code snapshot: `train_gpt.py` in this folder
- baseline comparison: original root `train_gpt.py` from the challenge repo

## command used for brelt

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True WARMUP_STEPS=5 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=25 MAX_WALLCLOCK_SECONDS=1200 RUN_ID=brelt_h100_full BRELT_PROFILE=full ENABLE_COMPILE=0 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=2 train_gpt.py
```

## results

### brelt

from `brelt_h100_full.log`:

- stop step: `504`
- wallclock stop: `1200.232s`
- final raw eval: `val_loss=1.3328`, `val_bpb=1.9228`
- final int8+zlib roundtrip: `val_loss=1.36969504`, `val_bpb=1.97605225`
- peak memory allocated: `71650 MiB`
- peak memory reserved: `76786 MiB`
- serialized model: `149377301 bytes`
- serialized model int8+zlib: `13825009 bytes`
- total submission size int8+zlib: `13924361 bytes`

### same-budget original baseline

from `baseline_h100_original.log`:

- stop step: `7056`
- wallclock stop: `1200.022s`
- final raw eval: `val_loss=2.0881`, `val_bpb=1.2367`
- final int8+zlib roundtrip: `val_loss=2.09857836`, `val_bpb=1.24289631`
- peak memory allocated: `10334 MiB`
- peak memory reserved: `10348 MiB`
- serialized model int8+zlib: `15803968 bytes`
- total submission size int8+zlib: `15851654 bytes`

## training signals collected

### brelt training log

these fields were logged during training:

- `train_loss`
- `train_main_loss`
- `train_bpb`
- `active_layers`
- `patches`
- `super`
- `max_patches`
- `max_super`
- `avg_patch_len`
- `boundary_rate`
- `hard_boundary_rate`
- `boundary_bias`
- `latent_rate`
- `dual_lambda`
- `lr_scale`
- `compute_proxy`
- `max_compute_proxy`
- `seen_gb`
- `throughput_mib_s`
- `train_time`
- `step_avg`

selected brelt checkpoints from the log:

- step 50: `train_bpb=3.0702`, `patches=94.2`, `latent_rate=0.0056`, `compute_proxy=7995.9`
- step 100: `train_bpb=2.4546`, `patches=138.7`, `latent_rate=0.0010`, `compute_proxy=17150.0`
- step 150: `train_bpb=2.2533`, `patches=191.5`, `latent_rate=0.0008`, `compute_proxy=29820.5`
- step 200: `train_bpb=2.1672`, `patches=187.8`, `latent_rate=0.0005`, `compute_proxy=28783.9`
- step 225: `train_bpb=2.0566`, `patches=188.7`, `latent_rate=0.0004`, `compute_proxy=29080.9`
- step 300: `train_bpb=2.0200`, `patches=191.3`, `latent_rate=0.0005`, `compute_proxy=29791.3`
- step 400: `train_bpb=1.9712`, `patches=186.5`, `latent_rate=0.0011`, `compute_proxy=28402.8`

### baseline training log

from `baseline_h100_original.log`:

- step 50: `train_loss=4.0518`, `step_avg=169.34ms`
- step 100: `train_loss=3.3492`, `step_avg=169.72ms`
- step 200: `train_loss=2.7883`, `step_avg=169.68ms`
- step 400: `train_loss=2.3708`, `step_avg=169.99ms`
- final int8+zlib roundtrip: `val_bpb=1.24289631`

## files in this package

- `README.md`
- `submission.json`
- `train_gpt.py`
- `brelt_h100_full.log`
- `baseline_h100_original.log`
- `early_baseline.log`
- `requirements.txt`
