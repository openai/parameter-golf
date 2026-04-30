# Non-Record: State-Space Hybrid with Attention Anchors

This folder records a wishlist-aligned `state-space models` lane for the non-record 16 MB track.

This is **not** a leaderboard record attempt.
This is **not** an official 8xH100 / 10-minute lane run.
This is **not** a full-train-shards claim for the kept run.
This is **not** a claim that the currently promoted public checkpoint is the latest strongest internal finding.

Track label:

- fixed-predictor state-space hybrid
- no adaptive compression
- no eval-time adaptation
- no TTT

## Summary

This PR keeps a conservative, non-record state-space sign-of-life on the public lane. The current public promoted checkpoint remains `STRONGER_VALID_STATE_SPACE_HYBRID_NON_RECORD_V8`, using the `AAAASASSS` fixed-predictor hybrid layout on the standard `train_gpt.py` scorer path.

A later long local campaign refreshed the legal all-attention control frontier and did **not** produce a reviewer-defensible new public promotion. The public checkpoint below should therefore be read as a historical promoted checkpoint for this lane, not as the strongest known internal result after the latest control refresh.

## Current Promoted Public Checkpoint

- Classification: `STRONGER_VALID_STATE_SPACE_HYBRID_NON_RECORD_V8`
- Layout: `AAAASASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: four early exact-attention blocks, one mid attention anchor, and a four-block SSM tail
- SSM core: `s4d`
- SSM kernel size: `96`
- SSM rank: `14`
- Fixed-predictor transfer: `SMEAR_ENABLED=1`
- Export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `4242`
- Training budget: `2200` steps
- Training data actually used by the kept run: the single locally available `fineweb_train_000000.bin` shard
- Primary metric: `final_int8_zlib_roundtrip_exact val_bpb`
- Public promoted score: `1.50126339`
- Public promoted loss: `2.53482035`
- Artifact bytes: `57,941` code + `15,214,485` model = `15,272,426` total bytes
- Wallclock: `1,207,089 ms` training + `124,606 ms` evaluation + about `5,764 ms` export / roundtrip overhead

At the time V8 was promoted, the strongest retained legal all-attention matched control was:

- Control: `full_baseline_1420steps_blackwell_seed2027_top1blockfp16_v7`
- Control score: `1.56658161`
- Control artifact bytes: `15,993,409`
- Historical V8 margin vs that control: `-0.06531822` BPB

That historical control comparison is retained for provenance. It is no longer the strongest known internal legal control after the later local campaign described below.

## Latest Internal Local-Campaign Finding

After V8, a longer local-only Blackwell campaign refreshed the legal all-attention frontier and blocked a new public promotion.

The strongest known internal legal all-attention control from that campaign is:

- Control: `full_baseline_2600steps_blackwell_seed2027_allint8_v9r3`
- Layout: `AAAAAAAAA`
- Score: `1.48142748`
- Artifact bytes: `57,941` code + `15,319,767` model = `15,377,708` total bytes
- Wallclock: `895,800 ms` training + `216,725 ms` evaluation + about `5,176 ms` export / roundtrip overhead

The best unpromoted hybrid candidate from that campaign was:

- Candidate: `full_anchor_s4d_aaaasasss_rank14_k96_corefp16_smear_2600steps_blackwell_seed4242_v9r3`
- Layout: `AAAASASSS`
- Score: `1.48097508`
- Artifact bytes: `57,941` code + `15,471,811` model = `15,529,752` total bytes
- Wallclock: `736,141 ms` training + `124,986 ms` evaluation + about `9,076 ms` export / roundtrip overhead

The candidate was legal and lower on raw BPB than the refreshed control, but only by `0.00045240` BPB. That is too small to satisfy the promotion rule requiring either a large matched-control advantage or a clearly documented control package proving that the hybrid still materially matters.

No new public promotion was made from that campaign.

## Validity / Scope

Passed for the current public non-record checkpoint:

- Same scorer path for control and hybrid: `train_gpt.py`, `final_int8_zlib_roundtrip_exact`
- Full official validation split for the promoted public checkpoint
- Artifact byte audit under the decimal `16,000,000` byte cap
- All counted code for the artifact lives in `train_gpt.py`
- No validation-data training
- No evaluation-time downloads or hidden services
- Fixed-predictor labeling remains explicit
- No eval-time adaptation or TTT
- Recurrent export policy is accounted for separately from the attention / MLP export policy

Main scope limits:

- The kept promoted result is still one-shard local: only `fineweb_train_000000.bin` was locally usable.
- The local dataset manifest reports `195` train shards, so one-shard training remains the biggest realism bottleneck.
- A bounded remote realism package existed earlier through Modal on an 80-shard cached view, but cloud-credit-backed continuation is unavailable now and is not part of any current promotion gate.
- No official-lane H100 feasibility result is claimed.

## Notes

This PR stays intentionally conservative and draft.

The lane remains interesting as a wishlist-aligned, non-record state-space models sign-of-life, but the latest internal evidence says the next public promotion likely needs either stronger realism, a more orthogonal state-space contribution, or a control package that shows the SSM tail matters by more than the tiny refreshed-control margin found in the long local campaign.
