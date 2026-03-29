# brelt wip on 2xh100 for 20 minutes

this is a non-record wip package for **brelt**, a byte-level recurrent latent architecture.

the point here is not to claim a leaderboard result yet.
the point is to preserve a real high-compute scaling run, keep the exact code and logs in one place, and show that the architecture already has real signs of life even though it is still unfinished.

the main brelt repo, where the broader development is happening, lives here:

- https://github.com/guilhhotina/brelt

that repo is the living project
this folder is the frozen package for this specific run

## what brelt is trying to do

brelt is built around one core bet:

> a model should not have to think over the full visible byte or token sequence at one flat resolution all the time

instead, it tries to:

- read raw bytes
- compress local spans into patch latents
- run recurrent global mixing over a much shorter internal sequence
- decode back to byte predictions
- stay robust under aggressive quantization and tiny artifact budgets

in other words, brelt is trying to learn a shorter internal sequence and spend most of the expensive computation there

## what this package is

this package captures a **2xh100 / 20-minute** non-record run of brelt, plus a same-budget run of the original challenge baseline for comparison

it is not a record attempt
the official main track is **10 minutes on 8xh100**
this is a scaling and diagnosis package

## setup

- track: `non-record-16mb`
- hardware: `2xh100`
- wallclock: `1200s`
- dataset: `fineweb10B_sp1024`
- train shards: `80`
- validation: full `fineweb_val_*` split
- code snapshot: `train_gpt.py` in this folder
- baseline comparison: original root `train_gpt.py` from the challenge repo, run later on the same machine and budget

## command used for brelt

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True WARMUP_STEPS=5 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=25 MAX_WALLCLOCK_SECONDS=1200 RUN_ID=brelt_h100_full BRELT_PROFILE=full ENABLE_COMPILE=0 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=2 train_gpt.py
```

## brelt result

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

## same-budget baseline comparison

the same machine then ran the **original challenge baseline** with no architectural edits, only the same wallclock budget and logging setup

from `baseline_h100_original.log`:

- stop step: `7056`
- wallclock stop: `1200.022s`
- final raw eval: `val_loss=2.0881`, `val_bpb=1.2367`
- final int8+zlib roundtrip: `val_loss=2.09857836`, `val_bpb=1.24289631`
- peak memory allocated: `10334 MiB`
- peak memory reserved: `10348 MiB`
- serialized model int8+zlib: `15803968 bytes`
- total submission size int8+zlib: `15851654 bytes`

## why this still matters

this run did **not** beat the original baseline on the same 2xh100 budget

but it still proved a few important things:

- recurrent global latent mixing does scale and stays trainable
- brelt can use large h100 compute and memory without conceptually collapsing
- the architecture is no longer just an interesting toy
- the failure mode under scale is much clearer now

that failure mode is basically this:

- segmentation opens too far
- patch count climbs toward ~190
- the latent stream becomes too cheap
- too much compute gets spent on fragmentation instead of useful abstraction

that is bad for this run, but very good for iteration
it turns the next version from guessing into diagnosis-driven work

## why this is being submitted as wip

the main repo is where the day-to-day architecture work is happening

this package exists to make one thing easy:

- review the exact code used
- inspect the exact logs
- understand the current scaling behavior
- keep the experiment archived in the challenge repo

the plan is to come back with a real record attempt later, once the scaling pathology above is addressed properly

## included files

- `README.md` — this summary
- `submission.json` — metadata for the non-record package
- `train_gpt.py` — exact brelt code snapshot used for the run
- `brelt_h100_full.log` — exact brelt training log
- `baseline_h100_original.log` — exact baseline training log on the same machine and budget
- `early_baseline.log` — short early baseline probe captured before reordering the runs to put brelt first
- `requirements.txt` — dependency snapshot used by the challenge repo environment
