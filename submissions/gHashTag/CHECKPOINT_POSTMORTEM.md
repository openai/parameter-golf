# Checkpoint Infrastructure Post-Mortem

**Date:** 2026-04-30 · **Auditor:** perplexity-computer-grandmaster (R5-honest lane)
**Anchor:** `phi^2 + phi^-2 = 3`

## Summary

**We have no `model.bin` to submit.** 1,851 experiments in
`experiment_queue`, 5,243 mid-training BPB rows in `bpb_samples`, 307
runs marked `done` — zero weight tensors saved to persistent storage.

This post-mortem documents the three failure points and the fix path
for Gate-3.

## Failure 1 — `record_checkpoint()` is a stub

In [`trios-trainer-igla/src/train_loop.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/train_loop.rs),
the trainer's inner loop logs intent (`println!("seed=... step=... val_bpb=...")`)
and writes telemetry to `bpb_samples` via
[`trios_trainer::neon_writer::bpb_sample(canon, seed, step, bpb)`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/neon_writer.rs)
— but there is no call site that serialises the actual `HybridModel`
struct (the 196K–2M parameter tensors) to disk.

`grep -rn "safetensors\|serialize\|save_model\|checkpoint" src/` in the
trainer repo returns zero matches for the weight-serialization intent.
The ledger has a triplet contract for **evidence** (BPB, step, seed,
sha) but the evidence does not include the weights themselves.

## Failure 2 — Railway containers use ephemeral storage

The seed-agent Dockerfile (`Dockerfile.real-seed-agent`) does not mount
a persistent volume. The trainer's working directory is
`/work`, which is the container's writeable layer — wiped on every
redeploy. We ran 32 redeploys today (acc1+acc2 batch ops), so every
in-progress run lost its state.

Railway's free tier does not ship native scheduled-volume mounts; the
project would need to either (a) upgrade to a paid plan with volumes,
or (b) stream checkpoints to object storage (S3, R2, or GitHub Releases
on a schedule).

## Failure 3 — No local mirror

Over the 24-hour sprint, no operator and no cron job copied weight
tensors out of the Railway containers. The only locally-present
weights in the workspace (`/home/user/workspace/trios-trainer-igla/target/release/`)
are binaries, not trained weights. There is no `.safetensors`, `.pt`,
`.bin`, or `.ckpt` file anywhere under `trios-*` or `t27`.

A search confirms the negative result:

```
$ find /home/user/workspace -type f \
      \( -name '*.safetensors' -o -name '*.pt' -o -name '*.ckpt' \
         -o -name '*weights*' -o -name '*model.bin' \) 2>/dev/null
(no matches outside placeholder model.bin which is 59 bytes of ASCII)
```

## Why synthetic weights are refused

From the R5-honest standing orders and the reviewer's explicit warning
earlier in this sprint:

> "Не пытайся подделать checkpoint — не генерируй synthetic random
> weights даже с disclaimer. Ratification bot прогонит на eval,
> получит BPB~8, ты попадёшь в black list submitter'ов."

Submitting a random-initialized tensor disguised as a trained artifact
would:

1. Produce BPB ≈ 8 on the reviewer's eval (random over 256-vocab).
2. Contradict the explicit claim on this PR.
3. Violate the `NO-DONE-WITHOUT-EVIDENCE` rule
   ([SOURCE_OF_TRUTH.md](https://github.com/gHashTag/trios-trainer-igla/blob/main/SOURCE_OF_TRUTH.md))
   that every DONE claim must be backed by a reproducible ledger row +
   green CI + merged PR.
4. Compromise the `gHashTag` GitHub identity in the eyes of the
   Parameter Golf maintainers permanently.

We accept the deadline cost and submit no model.

## Fix path for Gate-3

### Fix 1 — add safetensors writer to trainer

Rust `safetensors` crate ships a `serialize_to_file(tensors, path)`
helper. Hook into `train_loop::run_single` at `step % 200 == 0`:

```rust
if step.is_multiple_of(cfg.checkpoint_interval()) {
    let tensors: HashMap<String, TensorView> = model.state_dict();
    safetensors::serialize_to_file(
        tensors,
        None,
        format!("/data/ckpts/{canon}_step{step}.safetensors"),
    )?;
}
```

`checkpoint_interval()` is already in the trainer (default 200) via
[trios-trainer-igla#56](https://github.com/gHashTag/trios-trainer-igla/pull/56).
The missing piece is the `serialize_to_file` call.

### Fix 2 — Railway volume or S3 streamer

Two options:
- **Paid-tier volume:** Railway volumes at `/data` at ~ $0.25/GB-mo.
  Simplest; mount on every seed-agent service.
- **R2 streamer:** add a `bin/weight-streamer` Rust service that
  watches `/data/ckpts/*.safetensors`, uploads to Cloudflare R2 under
  `trios/ckpts/{exp_id}/step{N}.safetensors`, records the URL in
  `bpb_samples.ckpt_url` (new column). $0/mo up to 10GB/mo egress.

Recommend R2 streamer for cost and cross-worker accessibility.

### Fix 3 — Khepri-3 checkpoint audit

New ledger-daemon Job 5 (PR-6 of the Scarabaeus
[umbrella #101](https://github.com/gHashTag/trios-railway/issues/101)):
every 5 min, assert `bpb_samples.ckpt_url IS NOT NULL` for every row
with `step >= 1000`. Any row without a checkpoint URL flips the
experiment to `last_error='SCARABAEUS-NO-CKPT'` and is excluded from
ratification. This prevents Gate-3 from repeating Gate-2's mistake.

## Acceptance for Gate-3

- [ ] `record_checkpoint()` writes a real `.safetensors` to `/data/ckpts/`
- [ ] A local `cargo run --bin smoke_train -- --canon CKPT-SMOKE-1`
      produces a readable 59 KB+ safetensors file
- [ ] R2 streamer uploads at least one file successfully in CI
- [ ] Khepri-3 Job 5 flags a synthetic no-checkpoint run within 5 min
- [ ] Next-sprint submission PR includes the actual weight file

phi² + phi⁻² = 3 · R5-honest · NEVER STOP.
