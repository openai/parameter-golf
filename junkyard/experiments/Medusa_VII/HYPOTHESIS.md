# Medusa_VII — Causality Fix + DeltaNet Ablation

## What changed vs Medusa_IV

**Fix 1: DeltaNet cross-loop state carry (causality violation)**

Medusa_IV carried the DeltaNet final state from loop N into loop N+1 as `initial_state`.
After loop 1, that state encodes all positions 0..T-1. In loop 2, position `t` initializes
from this state, which already contains writes from positions `t+1..T-1` — future tokens
leak into every prediction. This explains the anomaly where sliding window BPB was *worse*
than roundtrip BPB: the look-ahead disproportionately helped early positions (large fraction
of the roundtrip average), but sliding window only scores late-in-window positions where
genuine causal context already rich.

Fix: state is NOT carried between loops. Each loop calls `chunk_delta_rule` with
`initial_state=None`. Within a single call the kernel is causal (left-to-right).

**Fix 2: prefill_shard header offset (oracle/mixer — inactive)**

Both `TrainNgramOracle.prefill_shard` and GPU oracle `prefill_shard` read from byte 0,
ingesting the 1024-byte shard header as 512 garbage uint16 tokens into hash tables.
Fixed to skip the 256×int32 header, matching `load_data_shard`. Inactive in current
runs (ARTIFACT_NGRAM=0, MIXER_ENABLED=0), but correct for future use.

## Ablation design

| Run | DELTA_NET_HEADS | Config | Expected |
|-----|-----------------|--------|----------|
| Medusa_IV s300 | 4 (causal violation) | baseline | 0.9578 (known) |
| Medusa_VII s300 DN=0 | 0 | EMA+GPTQ, no DeltaNet | ? — isolates EMA+GPTQ contribution |
| Medusa_VII s300 DN=4 | 4 (fixed, causal) | EMA+GPTQ + legal DeltaNet | ? — isolates DeltaNet value |

The gap (DN=0) → (DN=4 fixed) = real DeltaNet contribution, no causal cheat.
The gap (DN=4 fixed) → Medusa_IV (DN=4 broken) = how much was the causality violation worth.

## Run commands

```bash
# Ablation: no DeltaNet (clean EMA+GPTQ baseline)
DELTA_NET_HEADS=0 SEED=300 bash experiments/Medusa_VII/run.sh

# Fixed DeltaNet
SEED=300 bash experiments/Medusa_VII/run.sh
```

## Results

| Run | Seed | Post-EMA | RT Int6 | SW Int6 | SW<RT? | EMA steps | Size | Notes |
|-----|------|----------|---------|---------|--------|-----------|------|-------|
| CC_II | 1337 | 0.7278 | ~0.934 | 1.0427 | ❌ | ~full | ~9.8MB | DN=4 violation, no late EMA |
| Medusa_IV | 300 | 0.3882 | ~0.85 | **0.9578** | ❌ | ~3000 | ~10.1MB | DN=4 violation — score was the bug |
| M_VII DN=0 | 300 | 1.2002 | 1.2065 | **1.1823** | ✅ | 3088 | 9.08MB | Honest baseline |
| M_VII DN=4 fixed | 300 | 1.2124 | 1.2204 | **1.1958** | ✅ | 494 | 9.60MB | EMA-starved (55% slower train) |

## Verdict

- SW < RT restored on both honest runs — causality fix confirmed working.
- Causal DeltaNet (per-loop reset) provides no measurable benefit at this wall-clock budget.
  55% compute overhead → fewer training steps → EMA starved (494 vs 3088 steps).
  Even accounting for this, result is ~identical to DN=0.
- The entire 0.9578 margin over baseline came from the DeltaNet look-ahead (causality violation).
- Honest architecture score: **~1.18 BPB** (Medusa_VII DN=0).
