# PR 363 — Depth Recurrence in Parameter-Constrained Transformers: What Works, What Doesn't, and Why

**Author:** Evangeline Kamin (evangelinehelsinki)
**Branch created:** 2026-03-21 (research writeup updated 2026-03-24/25)
**Claimed BPB:** 1.1787 (best looped, sliding window); controlled flat comparison 1.1648; submission.json val_bpb=2.3876 (non-record entry)
**Artifact size:** 1,461,542 bytes (submission.json); looped best ~15.6 MB per README
**Seeds:** not stated for primary runs; SEED=42 for controlled comparison

## Files retrieved
- `records__track_non_record_16mb__2026-03-21_DepthRecurrence_MixedPrecisionQuant__README.md`
- `records__track_non_record_16mb__2026-03-21_DepthRecurrence_MixedPrecisionQuant__submission.json`
- `records__track_non_record_16mb__2026-03-21_DepthRecurrence_MixedPrecisionQuant__train_gpt.py`
- `records__track_non_record_16mb__2026-03-21_DepthRecurrence_MixedPrecisionQuant__requirements.txt`

## Claimed changes (from README, verbatim)

> I spent four days trying to make depth-recurrent transformers competitive in Parameter Golf. They aren't. A flat 11-layer model beats a looped 3x3 model by 0.025 bpb on identical hardware with identical tricks.
>
> Two findings survived: Noisy QAT (a training technique that collapses quantization error amplification through recurrence from 0.37 bpb to 0.002 bpb) and the 3x3 > 2x5 loop configuration (more unique blocks with fewer repeats beats fewer blocks with more repeats).
>
> Noisy QAT: Instead of STE fake-quantization, inject differentiable uniform noise calibrated to match the magnitude of int8 per-row quantization error:
> ```
> with torch.no_grad():
>     amax = self.weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
>     step_size = amax / 127.0
> noise = (torch.rand_like(w) - 0.5) * step_size.to(w.dtype)
> w = w + noise
> ```
> Applied only to core (shared) blocks. Gap collapsed from 0.37 to 0.002 bpb.
>
> Controlled comparison: Flat 11L 512d = 1.1648 bpb / 15.3 MB / 5375 steps / 112 ms/step. Looped 3x3 640d = 1.1894 bpb / 14.5 MB / 4175 steps / 144 ms/step.
>
> SmearGate hurt recurrence (gating mechanism incompatible with shared weights). MTP broke badly (auxiliary gradients corrupted shared recurrent weights). Value Residual +0.14 worse. XSA all-layers on looped +0.001 worse. Cyclic Muon momentum +0.058 worse. Factored embeddings (192/256) +0.053 to +0.063 worse. Late QAT + int5 +0.006 worse. BigramHash(10240) no improvement on looped.

Non-record research writeup. PR also contains `pr325_train_gpt.py` script (not the records train_gpt.py).
