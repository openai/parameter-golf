# 11L Shared Sparse Sidecar + EMA + AdamW TTT

**3-seed mean val_bpb: 1.0916**  
Best seed: **1.0901**  
Track: `10min / 16MB`  
Hardware: `8xH100 SXM`, `600s` train budget

## Result

| Seed | Steps | val_loss | val_bpb | Total bytes |
|------|------:|---------:|--------:|------------:|
| 13 | 5965 | 1.84057430 | 1.09009466 | 15,973,374 |
| 1337 | 5966 | 1.84364647 | 1.09191418 | 15,986,252 |
| 1111 | 5961 | 1.84521443 | 1.09284281 | 15,868,806 |
| **Mean** | | **1.84314507** | **1.09161722** | |
| **Std** | | **0.00236** | **0.00140** | |

This submission is a derivative of `@sjp611`'s `1.1027` AdamW-TTT 11-layer stack. The main new idea is a **shared sparse sidecar** injected only in the late layers, plus the trimming work required to make that architecture legal under the strict decimal `16,000,000`-byte cap on real `8xH100` runs.

## What Is New Here

The donor already had a strong trunk:
- `11` layers, `512` model dim
- `8` heads / `4` KV heads
- `MLP 3x`
- `SmearGate`
- `BigramHash`
- `EMA`
- `AdamW` TTT
- stride-`64` sliding-window eval
- `int6+zstd` export

The contribution here is not another small optimizer retune. It is a new late-stage refinement path:

- a **single auxiliary sidecar module** is reused across multiple late layers
- each insertion site gets a **learned site embedding**
- each insertion site also gets a **learned residual scale**
- the module itself is lightweight: `gate -> value -> depthwise conv -> proj`

That gives the model extra late local-composition capacity exactly where representations are already semantically rich, without paying for another full transformer block at every site.

## Shared Sparse Sidecar

The sidecar is applied only in the late trunk:
- full local ablation version: `SPARSE_HIDDEN_DIM=64`
- legal cloud submission version: `SPARSE_HIDDEN_DIM=48`
- insertion starts at `SPARSE_START_LAYER=8`
- sidecar weights are **shared across sites**

The sidecar has four pieces:
- gate projection from model dim to sidecar dim
- value projection from model dim to sidecar dim
- depthwise `Conv1d` to add cheap local mixing
- projection back to model dim

The sharing is the important part. The model gets multiple late refinement sites, but the expensive weights are reused. Only the small site embeddings and residual scales are site-specific. That is why this idea is viable under a hard artifact cap.

## Why This Helped

The donor stack is already strong on global sequence modeling and test-time adaptation. The sidecar adds something different:

- **late local refinement** rather than more full-width global attention
- **parameter sharing** rather than duplicating full blocks
- **site-conditioned specialization** rather than forcing one identical late computation everywhere

In practice, this acts like a cheap late feature refiner that complements the existing 11-layer trunk instead of competing with it.

## Matched Sidecar Evidence

Before the final cloud-legal trim, the sidecar idea was validated on the exact donor family in a matched local capped-val experiment:

| Variant | Setting | final int6 roundtrip val_bpb |
|--------|---------|------------------------------:|
| Base donor | no sidecar | 1.09458804 |
| Sidecar | `SPARSE_HIDDEN_DIM=64` | 1.07249022 |

This is not the official leaderboard metric; it is a local capped validation ablation. But it is the cleanest evidence that the sidecar itself was adding value on top of the donor stack before the legal artifact trim.

## Cloud-Legal Trim

The first sidecar-on-donor version was too close to the size cap. The successful legal trim was:

- `SPARSE_HIDDEN_DIM: 64 -> 48`
- `BIGRAM_DIM: 128 -> 96`
- `MAX_WALLCLOCK_SECONDS: 600 -> 596`

This matters because reducing the number of sidecar insertion sites barely saves bytes; the sidecar is shared. The real size knobs were sidecar width and bigram projection width. The final trimmed model kept the architectural idea intact while making all three cloud seeds legal.

## Training / Eval Details

- `ITERATIONS=9000`, early stop on wallclock
- actual stop: about `5960` steps on `8xH100`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786,432`
- `WARMDOWN_ITERS=3000`
- `EMA(0.997)`
- `AdamW` TTT for `10` epochs
- full sliding-window eval with stride `64`
- final artifact uses `int6+zstd`

On the best seed:
- train stop at `596.027s`
- TTT time about `162.4s`
- final sliding eval about `84.8s`

## Comparison

Compared to the donor branch mean:

- donor mean `val_bpb`: `1.1027`
- this submission mean `val_bpb`: `1.0916`
- improvement: **-0.0111 BPB**

So the sidecar was not just legalized successfully; it also moved the final track metric in the right direction on real cloud runs.

## Attribution

This submission is intentionally presented as a derivative work.

- direct donor: `@sjp611`'s `11L EMA + AdamW TTT (10ep)` line
- donor lineage: `PR #398` (`11L EMA + SGD TTT`)
- new contribution here: **shared sparse sidecar architecture**, **late-site conditioning**, and the **cloud-legal trim / 3-seed cloud reproduction**

That is the ownership split I want to make explicit:
- the AdamW-TTT 11-layer trunk is inherited
- the shared late sidecar and its successful cloud-legal deployment are the main new contribution here
