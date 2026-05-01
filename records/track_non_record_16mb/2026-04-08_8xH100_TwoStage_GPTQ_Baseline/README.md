# Non-record: Checkpointed AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112

**Author:** Jaksen ([@jaksen](https://github.com/jaksen))
**Base:** [2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072](../../track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md)
**Contribution:** A validated checkpointed `8xH100 -> 1xH100` execution path for the public AR self-gen GPTQ + XSA-all + BigramHash `3072 x 112` stack
**Best saved result:** `1.13071788 val_bpb`, `15,651,808` bytes, `seed=314`

## The Short Version

This folder documents my strongest saved `8xH100`-backed Parameter Golf result
to date. I am not presenting it as a leaderboard claim. The useful contribution
here is the execution path: Stage 1 trains and checkpoints on `8xH100`, and
Stage 2 runs GPTQ, artifact packing, and final evaluation on `1xH100`.

That two-stage split matters because it proves that GPTQ and final evaluation do
not need to remain on the expensive `8xH100` box for this stack. The strongest
saved path in this folder is `seed=314`, ending at
`final_int6_sliding_window_exact val_bpb: 1.13071788` with a final artifact
size of `15,651,808` bytes.

## Result

| Item | Value |
| --- | --- |
| Seed | `314` |
| Stage 1 hardware | `8xH100 80GB` |
| Stage 2 hardware | `1xH100 80GB` |
| Post-EMA diagnostic | `1.1501` |
| Final roundtrip `val_bpb` | `1.15442828` |
| Final sliding `val_bpb` | `1.13071788` |
| Final artifact bytes | `15,651,808` |
| Stage 1 steps | `4783` |
| Stage 1 avg step time | `~124 ms` |

## What This Contributes

- A working `8xH100 -> 1xH100` two-stage pipeline for this exact stack.
- A saved result under the decimal `16,000,000` byte cap.
- A clean proof that checkpoint save plus single-GPU GPTQ replay works end to
  end.
- A baseline I can reuse for future compliant reruns without keeping the full
  post-train path on `8xH100`.

## Lineage / Credit

This submission is directly based on the public record lineage in
[2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072](../../track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md).

I am not claiming the following techniques as new in this folder:

- AR self-generated GPTQ calibration
- XSA on all 11 layers
- BigramHash `3072 x 112`
- selective pruning
- `lzma` preset 9 packaging

The specific local contribution here is narrower:

- checkpointing the trained model on `8xH100`
- moving GPTQ, artifact packing, and final eval to `1xH100`
- preserving the strongest saved result from that path
- preserving the recovered raw Stage 1 and Stage 2 Modal logs

## Canonical Evidence

The canonical saved evidence for this submission lives in
[proxy_results.md](proxy_results.md):

- `record_seed314_stage1_jaksencharles`
- `record_seed314_stage2_jaksencharles`

The corresponding implementation and metadata live in:

- [submission.json](submission.json)
- [train_gpt.py](train_gpt.py)
- [run_gptq.py](run_gptq.py)
- [stock.env](stock.env)
- [stage1_modal_seed314.log](stage1_modal_seed314.log)
- [stage2_modal_seed314.log](stage2_modal_seed314.log)

## What Changed From The Base Stack

This line preserves the public `2026-03-25` modeling base:

- the same 11-layer architecture
- XSA on all 11 layers
- BigramHash `3072 x 112`
- AR self-generated GPTQ calibration
- selective pruning
- `lzma` preset 9

The key local additions are execution-focused rather than modeling-focused:

- [train_gpt.py](train_gpt.py) supports `SKIP_QUANTIZE=1` and checkpoint export
  so Stage 1 can stop after training and save `final_model.pt`
- [run_gptq.py](run_gptq.py) runs the GPTQ, pruning, roundtrip eval, and
  sliding-window eval path on `1xH100`
- [proxy_results.md](proxy_results.md) preserves the promoted smoke, proxy, and
  funded run summaries that led to the saved baseline
- [stage1_modal_seed314.log](stage1_modal_seed314.log) preserves the recovered
  raw Stage 1 `8xH100` training log
- [stage2_modal_seed314.log](stage2_modal_seed314.log) preserves the recovered
  raw Stage 2 `1xH100` GPTQ/eval log

## Operational Findings

The important operational lesson from this work is that stage splitting is
worth keeping.

- I validated the inherited stack on `1xH100` smoke runs before spending on
  `8xH100`.
- I ran multiple real `8xH100` attempts and identified the main budget sinks:
  launcher/runtime confusion, FA3/runtime mismatch, and keeping GPTQ on the
  expensive box.
- The two-stage path removed the biggest avoidable post-train cost by moving
  GPTQ and final evaluation to `1xH100`.

That is the main reason this folder is useful even though the saved score is
not a record.

## Historical Run Sequence

The campaign history for this fork was:

1. `smoke.env`
2. `stock.env`
3. `warmdown4500.env`
4. `arcalib96.env`
5. funded `8xH100` attempts
6. two-stage validation on `1xH100`
7. strongest saved `8xH100 -> 1xH100` baseline on `jaksencharles`

The strongest saved funded baseline was:

- Stage 1 on `8xH100`
- Stage 2 on `1xH100`
- final saved sliding score `1.13071788`

## Why This Is Non-record

As of `2026-04-08`, the merged rank-1 leaderboard entry in the repository
README is `1.1147`, while this saved result is `1.13071788`. I am also not
making any 3-seed significance claim from this folder.

So the right way to present this work is as a non-record technical submission:
useful because it proves an execution path and preserves a real saved result,
not because it already wins the challenge.

## Files In Scope

For review, the important files in this folder are:

- [README.md](README.md)
- [submission.json](submission.json)
- [proxy_results.md](proxy_results.md)
- [train_gpt.py](train_gpt.py)
- [run_gptq.py](run_gptq.py)
- [stock.env](stock.env)
- [requirements.txt](requirements.txt)
- [stage1_modal_seed314.log](stage1_modal_seed314.log)
- [stage2_modal_seed314.log](stage2_modal_seed314.log)

One small logging quirk is preserved in the Stage 2 log: after the exact
sliding-window result, the script also prints
`final_int8_zlib_roundtrip_exact` with the same final `1.13071788` value. That
label comes directly from the inherited logging path in
[run_gptq.py](run_gptq.py); the actual artifact bytes and GPTQ path above it
show that this submission is the recovered Stage 2 int6+lzma run.
