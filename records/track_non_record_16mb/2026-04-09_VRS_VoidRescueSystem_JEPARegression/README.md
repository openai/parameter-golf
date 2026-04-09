# VRS: Void Rescue System for Regression-Based Transformers

This is a non-record 16MB submission for the OpenAI Parameter Golf challenge.

The contribution is architectural rather than leaderboard-oriented: a JEPA-style regression language model plus a small auxiliary "rescuer" decoder that corrects raw regression latents before final token decoding. The official challenge README explicitly asks for JEPA submissions, so this PR is intended as a concrete non-record research contribution in that direction.

## Summary

The model is split into two learned components:

- **Navigator**: a causal transformer trained to predict the next token's embedding with an MSE objective.
- **Rescuer**: a small 524,288-parameter MLP that maps the raw Navigator latent `v_void` to a corrected latent `v_rescued`.

Training is regression-only:

- `loss_A = MSE(v_void, target_embedding)`
- `loss_B = MSE(v_rescued, target_embedding)`
- `loss = 0.5 * loss_A + 0.5 * loss_B`

Evaluation decodes through the shared embedding table:

- raw path: `v_void @ E^T`
- rescued path: `v_rescued @ E^T`

This submission studies a specific failure mode of regression decoding: latents that already contain useful token information but are not directly decodable under the shared embedding geometry. The rescuer is intended to correct that geometric misalignment.

## Why This Is a Non-Record Submission

This run does not compete on absolute BPB against the 10-minute leaderboard. It is being submitted because the idea is distinct and empirically functional under the challenge constraints:

- 10-minute wallclock on 8xH100
- artifact under the 16,000,000 byte cap
- reproducible across 3 seeds
- clear improvement over the raw regression path
- clear improvement over standalone regression-only baselines

## Three-Seed 10-Minute Results

All runs used the same architecture and 600-second wallclock cap on 8xH100.

| Seed | Best `val_bpb` | Best raw `val_bpb_A` | Peak `nn_acc` | Stop step | Artifact bytes | Total bytes |
|------|---------------:|---------------------:|--------------:|----------:|---------------:|------------:|
| 7    | 1.8658 | 1.9550 | 0.5032 | 13432 | 15,939,206 | 15,980,840 |
| 42   | 1.8664 | 1.9269 | 0.5053 | 13469 | 15,941,691 | 15,983,325 |
| 1337 | 1.8679 | 1.9383 | 0.5067 | 13430 | 15,934,171 | 15,975,805 |
| mean | 1.8667 | 1.9436 | 0.5051 | — | 15,938,356 | 15,979,990 |

Interpretation:

- The rescuer consistently improves over the raw regression decode path by roughly `0.06` to `0.09` BPB.
- The effect is stable across seeds under the 10-minute budget.
- The best seed (`7`) is used for `submission.json`, while the additional logs are included to show consistency.

## Comparison Against Standalone Regression Baselines

I also trained three separate regression-only baselines with no rescuer and the same 10-minute budget.

| Model | Best `val_bpb` range | Peak `nn_acc` range |
|-------|---------------------:|--------------------:|
| Regression-only baseline | 2.0941 - 2.1301 | 0.5005 - 0.5018 |
| VRS (this submission) | 1.8658 - 1.8679 | 0.5032 - 0.5067 |

So the rescuer is not just improving an internal probe. It improves over standalone regression runs trained separately under the same wallclock regime.

## Submission Metadata Notes

This submission is based on the exact 10-minute submitable VRS script and its three original training logs.

One caveat: this script predates the newer `final_int8_zlib_roundtrip_exact` logging format used by some later Parameter Golf submissions. It does print:

- in-training validation metrics (`val_loss`, `val_bpb`, `val_bpb_A`, `nn_acc`)
- exact compressed artifact bytes
- successful roundtrip load (`roundtrip_validation:passed`)

Accordingly:

- `submission.json` reports the best logged validation numbers from the best seed run
- `bytes_model_int8_zlib`, `bytes_code`, and `bytes_total` are exact
- the included three logs show that the result is stable, not a single-seed outlier

## Reproducing the Run

From the repository root:

```bash
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
WARMDOWN_ITERS=20000 \
RESCUER_LR=0.04 \
SEED=7 \
torchrun --standalone --nproc_per_node=8 records/track_non_record_16mb/2026-04-09_VRS_VoidRescueSystem_JEPARegression/train_gpt.py
```

The script defaults were adjusted only for reproducibility inside the `records/...` folder:

- default dataset/tokenizer paths resolve relative to the repository root
- default `MAX_WALLCLOCK_SECONDS` is `600`

The modeling logic and training recipe are otherwise the original submitable VRS configuration.

## Included Files

- `train_gpt.py`: self-contained VRS training script snapshot
- `train.log`: best-seed log (seed 7)
- `train_seed42.log`
- `train_seed1337.log`
- `submission.json`: metadata for this non-record submission
- `results.tsv`: compact seed summary
- `vrs-spec.txt`: full technical spec for the method

## External Reference

- Zenodo record: https://zenodo.org/records/19477224
