# Ghost v7C Micro-Simulation Report

This is a CPU-only tiny-model simulation of the GPT-patched Ghost v7C branch.
It is intended to catch code-path failures before burning RunPod credits. It does not prove real BPB.

## v7C additions tested

- LeakyReLU(0.5)^2 MLP activation
- ResFormer sparse/value-residual code path, using learned softmax gates in the tiny sim
- Existing patched mechanics: quant/dequant roundtrip, corrected sliding eval, score-first LaCT TTT, no-QV freeze mask

## Raw sim output

```text
import_ok
micro_model_params:16682
forward_loss:3.456668 sec:0.009
per_token_shape:(1, 8) mean_matches:True
quant_done sec:0.001 raw_bytes:33364
roundtrip_loss:3.456674 delta:0.000006 missing:0 unexpected:0
sliding_eval_loss:3.491724 bpb:4.029994 sec:0.007
ttt:causal_online lr=0.001 freeze_layers=0 no_qv=True
ttt:no_qv — Q+V projections frozen, updating K+MLP+norms only
ttt:lact chunk_size=2 n_seqs=2 grad_steps=1
ttt:seq:2/2 running_bpb:4.2202
ttt:complete causal_online_bpb:4.2202
ttt_loss:3.473671 ttt_bpb:4.220167 sec:0.019
ttt_changed_params:12 frozen_qv_changed:0
zlib_compressed_bytes:21783 decompress_ok:True```

## Notes

The sandbox's torch.optim.AdamW initialization path hung during local CPU testing, so `micro_sim.py` patches in a tiny SGD-like optimizer for the simulation only. The real `train_gpt.py` still uses AdamW for RunPod/GPU execution. This means the micro-sim validates score-first TTT control flow and freeze masks, not AdamW performance.

## What passed

- Import and model construction work.
- Forward loss and per-token loss are consistent.
- Quantize/dequantize roundtrip works.
- Sliding-window eval runs with corrected per-token scoring.
- Score-first LaCT TTT code path executes: score chunk first, then adapt chunk.
- no-QV TTT mask holds: frozen Q/V parameters did not change.
- ResFormer added only tiny parameter overhead in the toy model: parameter count rose from 16680 to 16682.
- Compression/decompression roundtrip works.

## What this does not prove

- Real FineWeb BPB.
- Full 11L/512d stability.
- Real 8×H100 throughput.
- Real AdamW behavior in the cloud environment.
- Whether ResFormer or LeakyReLU improves the actual submission.

## Next step

Run a tiny RunPod smoke test before any full 8×H100 run. If smoke passes, run a single seed with:

```bash
LEAKY_RELU_SLOPE=0.5 RESFORMER_ENABLED=1 RESFORMER_MODE=sparse RESFORMER_LEARNED=1 RESFORMER_DETACH_V0=1
```

Keep a control run with `RESFORMER_ENABLED=0` so we can attribute gains or damage.
