# 11L XSA4 Late Shared Workspace Adapter (LSWA-64x4) + MLP2.5

This submission packages a **Late Shared Workspace Adapter** (`LSWA`) graft on top of the public 11-layer March 23 record lineage, while keeping the rest of the stack close to the donor.

- Best included legal seed: **1.08568610** exact post-quant `val_bpb`
- Included companion seeds: `13`, `1313`
- Included top-3 legal mean: **1.10581327**
- Track: `10min / 16MB`
- Hardware: `8xH100`, `598s` train budget

## Result

Only the top 3 legal capped logs are included in this record folder:

| Seed | Steps | final_int6_roundtrip_exact val_bpb | Total bytes |
|------|------:|-----------------------------------:|------------:|
| 2025 | 7197 | 1.08568610 | 15,900,041 |
| 13   | 7212 | 1.11462396 | 15,814,869 |
| 1313 | 7200 | 1.11712974 | 15,895,409 |
| **Mean** | | **1.10581327** | |
| **Std** | | **0.01747561** | |

This is a **record-track submission** centered on the LSWA architecture.

## Core Idea

The new idea is the **Late Shared Workspace Adapter**:

- tokens write into a small shared latent workspace
- the workspace performs one short internal refinement step
- refined workspace state writes back into token states
- the same adapter weights are reused across late decoder sites

In this submission the workspace is:

- `64` latent channels
- `4` workspace slots
- `4` workspace heads
- `1` think iteration
- active from decoder block `5` onward

The important point is that this adds a new computation pattern without replacing the main backbone. It is a **shared late add-on**, not a new transformer trunk.

## Why This Is Interesting

The workspace idea achieves strong scores with **minimal trunk changes**:

- same `11`-layer banked backbone
- same `512` model width
- same `8` attention heads / `4` KV heads
- same late `XSA` on the last `4` layers
- same bigram path
- same VE path on layers `9,10`

The main architectural differences versus the public donor line are:

- add **LSWA-64x4**
- reduce main-trunk `MLP_MULT` from `3.0` to `2.5`
- remove `TTT`
- remove `EMA / SWA / LAWA`
- use exact post-quant eval instead of sliding-window eval

So this is not a “rewrite the whole model” submission. It is a controlled demonstration that a shared workspace writeback path can compete while leaving the underlying record backbone mostly intact.

## Why Sharing Matters

The workspace module is reused across late sites instead of instantiating new full blocks. That matters under a strict artifact cap:

- the model gets multiple late workspace interactions
- the expensive adapter weights are shared
- only the placement in the existing decoder stack changes

This is what makes the idea viable in the 16MB track.

## Reproducing

This folder is self-contained and meant to run directly from inside `records/`.

```bash
cd records/track_10min_16mb/2026-03-27_11L_XSA4_LateSharedWorkspaceAdapter_MLP25
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

It still accepts overrides, for example:

```bash
SEED=13 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Defaults baked into `train_gpt.py`:

- `SEED=2025`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=598`
- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_SEQ_LEN=2048`
- `EVAL_SEQ_LEN=2048`
- `EVAL_STRIDE=0`
- `USE_COMPILE=1`

The packaged trainer writes logs and artifacts into this record folder. The current code size is `87,049` bytes.

## Included Files

- `train_gpt.py`: standalone trainer snapshot with record defaults baked in
- `train.log`: canonical best-seed log (`SEED=2025`)
- `train_seed2025.log`
- `train_seed13.log`
- `train_seed1313.log`
- `submission.json`

## Attribution

This submission is intentionally framed as a derivative work with one main new idea.

Public lineage kept in this snapshot:

- **Base 11-layer banked trunk**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **Parameter Banking + Parallel Muon**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **LeakyReLU(0.5)^2 donor activation line**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee and [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Public March 23 assembled record line**: `LeakyReLU² + Legal Score-First TTT + Parallel Muon` by @abaybektursun

New contribution in this submission:

- **Late Shared Workspace Adapter (LSWA-64x4)**
- `MLP_MULT=2.5` trunk trim to keep the workspace idea legal under the cap
- simplified no-TTT / no-EMA exact-eval deployment recipe
- record-folder packaging with baked-in defaults for a minimal `torchrun` launch
