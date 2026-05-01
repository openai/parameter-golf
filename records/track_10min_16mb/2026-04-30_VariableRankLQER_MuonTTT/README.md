# Non-record: Variable-Rank LQER + Muon-TTT experiment

**Status:** non-record negative result. This submission documents an attempted variable-rank LQER and Muon-as-TTT stack on a strong CaseOps/LQER/SparseGate baseline. The attempted stack did not improve seed 42 and exceeded the 600s evaluation limit, so it is not submitted as a leaderboard record.

## Summary

This experiment added three env-gated changes:

- `LQER_VARIABLE_RANK=1`: redistribute LQER correction rank by GPTQ Hessian trace while staying under the original raw LQER correction storage budget.
- `TTT_OPTIMIZER=muon`: use a Newton-Schulz/Muon-style update direction for LoRA TTT parameters.
- `PHASED_TTT_PREFIX_DOCS=3000`: evaluate a longer phased TTT prefix.

The baseline reproduction passed on seed 42, but the full stack regressed and exceeded eval time. No further paid runs were performed.

## Results

| Run | Seed | Config | Post-TTT BPB | Artifact bytes | Train time | Eval time | Outcome |
|---|---:|---|---:|---:|---:|---:|---|
| Reproduction | 42 | baseline, new switches off | 1.06005735 | 15,907,913 | 599.601s | 527.2s | valid reproduction |
| Stack | 42 | variable-rank LQER + Muon-TTT + prefix 3000 | 1.06203230 | 15,922,566 | 599.546s | 627.3s | regressed; eval non-compliant |

Reference copied from the source baseline for seed 42: post-TTT BPB `1.05989454`. The reproduction run landed within the planned reproduction window `[1.0585, 1.0615]`.

## What Failed

The variable-rank LQER allocator selected high-trace tensors and stayed within its storage budget:

```text
[lqer-var] selected=[('blocks.0.attn.c_q.weight', 8), ('blocks.0.attn.c_k.weight', 8), ('blocks.0.attn.c_v.weight', 8), ('blocks.3.attn.c_k.weight', 8), ('blocks.3.attn.c_v.weight', 8), ('blocks.4.attn.c_k.weight', 1), ('blocks.0.mlp.proj.weight', 8)] storage_units=55198/55494
```

However, the stack degraded the quantized diagnostic BPB before final TTT:

| Run | Diagnostic quantized BPB |
|---|---:|
| Reproduction | 1.07275682 |
| Stack | 1.07326567 |

The longer prefix also pushed final TTT eval over the 600s limit:

```text
quantized_ttt_phased val_loss:2.32412035 val_bpb:1.06203230 eval_time:627269ms
total_eval_time:627.3s
```

## Compliance Notes

- No root files are touched.
- The stack run is **not compliant** for leaderboard purposes because eval time was `627.3s`.
- Artifact size stayed under 16,000,000 bytes in both observed runs.
- The byte-accounting audit passed on both observed runs:

```text
[byte-audit:eval_val] caseops sidecar tokens=100000 bytes=312067 mean_bytes=3.1207 OK
```

## Files

- `train_gpt.py` — env-gated implementation of byte audit, variable-rank LQER, and Muon-TTT.
- `terminal_seed42_runs.md` — pasted terminal evidence for the seed 42 reproduction and stack run.
- `reference/train_seed{42,0,1234}.log` — copied baseline reference logs for comparison.
- `submission.json` — metadata for this non-record experimental result.
- `lossless_caps.py`, `prepare_caseops_data.py`, tokenizer model — inherited CaseOps support files.

## Reproduction Command

The valid reproduction uses the baseline switches:

```bash
LQER_VARIABLE_RANK=0 TTT_OPTIMIZER=adam PHASED_TTT_PREFIX_DOCS=2500 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The failed stack uses:

```bash
LQER_VARIABLE_RANK=1 LQER_VARIABLE_RANK_CAP=8 \
TTT_OPTIMIZER=muon TTT_MUON_NS_STEPS=3 \
PHASED_TTT_PREFIX_DOCS=3000 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

This experiment builds on the published CaseOps/LQER/SparseGate/SmearGate lineage and tests whether Hessian-allocated LQER plus Muon-style TTT composes on top of it. It did not compose in this seed-42 run, but the negative result may help narrow the search space.
