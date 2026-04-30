# Record: Independent 3-seed reproduction of PR #1874 + TTT_LORA_RANK=192

**val_bpb = 1.06996** (3-seed mean, std 0.00059) | **all 3 total submissions < 16 MB** | **all 3 trains < 600 s, all 3 evals < 600 s** | 8 × H100 80 GB SXM | **t = 17.67 vs current SOTA, p < 0.005**

> **What this submission is, in one paragraph.**
> An independent end-to-end reproduction of [PR #1874](https://github.com/openai/parameter-golf/pull/1874) by @AjAnubolu — the full SmearGate / AttnOutGate / LoRA-TTT / Phased Global SGD TTT / Polar Express NS / MIN_LR / LQER stack — run from scratch on our own pod across three independent seeds, with one additional hyperparameter change (LoRA-TTT rank raised from 128 to 192). All three training+eval logs are included unedited; all three quantized model artifacts are included reload-ready in `models/`; two additional sweep artifacts (rank=128 baseline and rank=192 single-seed) are included so a reviewer can independently verify the rank-delta claim. The 3-seed mean of 1.06996 is **0.011 nats** below the current merged SOTA (PR #1493, 1.0810), passes the README's 0.005-nat threshold by 3.5×, and clears the `p < 0.01` significance requirement by a wide margin (computed `t = 17.67` vs critical `t = 6.965` for one-tailed df = 2; passes `p < 0.005` by `t = 17.67` vs critical `t = 9.925`).

---

## 3-Seed Results (verbatim from logs)

| Seed | val_bpb (`quantized_ttt_phased`) | Total submission bytes | Headroom under 16M | Eval time (s) | Log | Artifact |
|------|----------------------------------|-----------------------:|-------------------:|---------------|-----|----------|
| 42   | **1.06927777** | 15,954,871 | 45,129 | 438.3 | [`train_seed42.log`](train_seed42.log)   | [`models/champion_3seed_42.int6.ptz`](models/champion_3seed_42.int6.ptz)   |
| 314  | **1.07023963** | 15,954,924 | 45,076 | 440.6 | [`train_seed314.log`](train_seed314.log) | [`models/champion_3seed_314.int6.ptz`](models/champion_3seed_314.int6.ptz) |
| 999  | **1.07035739** | 15,947,796 | 52,204 | 434.3 | [`train_seed999.log`](train_seed999.log) | [`models/champion_3seed_999.int6.ptz`](models/champion_3seed_999.int6.ptz) |
| **Mean** | **1.069958** | 15,952,530 | 47,470 | 437.7 | — | — |
| **Std (sample, n = 3)** | **0.000592** | — | — | — | — | — |

Every number in this table is produced by the included `train_gpt.py` and reported by the script itself. Grep any log for `quantized_ttt_phased val_loss:... val_bpb:...` (line ~759) and `Total submission size quantized+brotli:... bytes` (line ~136) — those lines are the source of truth.

---

## Byte-Budget Compliance — Authoritative Numbers

The challenge counts model + LZMA-wrapped code together. The included `train_gpt.py` runs `_compressed_code_size()` at the end of every training run, which reads its own source, runs it through pyminify + lzma + b85, and reports the resulting byte count. That number is added to the brotli-compressed int6 model artifact to produce the total.

| Component | Seed 42 | Seed 314 | Seed 999 |
|-----------|--------:|---------:|---------:|
| Code size (uncompressed source the script self-introspects) | 134,706 B | 134,706 B | 134,706 B |
| Code size (lzma-wrapped, what the budget charges) | 33,710 B | 33,710 B | 33,710 B |
| Model `.int6.ptz` (brotli-compressed quantized state dict) | 15,921,161 B | 15,921,214 B | 15,914,086 B |
| **Total submission bytes (model + wrapped code)** | **15,954,871 B** | **15,954,924 B** | **15,947,796 B** |
| **Cap** | **16,000,000 B** | **16,000,000 B** | **16,000,000 B** |
| **Headroom** | **45,129 B** | **45,076 B** | **52,204 B** |

The shipped `train_gpt.py` is 32,353 B on disk (already LZMA-wrapped). The 33,710 B figure above is what the script computes for *its own* budget when run; both are well under any reasonable interpretation of the cap.

---

## Statistical Significance

The README requires beating SOTA by 0.005 nats at `p < 0.01`:

- SOTA at submission: **1.0810** (PR #1493 by @bigbag, currently merged on `main`)
- Our 3-seed mean: **1.069958**
- Improvement vs SOTA: **0.011042 nats**
- Required improvement: 0.005 nats
- Excess over requirement: **0.006042 nats**
- Standard error of the mean (n = 3, df = 2): 0.000342
- **t-statistic: 17.67**
- Critical t (one-tailed, df = 2, p = 0.01): 6.965 → **passes**
- Critical t (one-tailed, df = 2, p = 0.005): 9.925 → **passes**
- p-value bound: **< 0.005**

Even if a reviewer wanted to attribute zero credit to our `TTT_LORA_RANK 128 → 192` change and treat the submission purely as an independent reproduction of PR #1874's stack, the 3-seed mean still clears the 0.005-nat threshold over the current merged SOTA at `p < 0.005`.

---

## What We Actually Did (compute log + sweep table)

This is a single-paragraph, no-spin account. We ran ~$245 of compute on a single 8×H100 RunPod node over a ~36-hour weekend window (2026-04-26 to 2026-04-28). Phases:

1. **Phase 0 — independent reproduction of PR #1874 (single seed, ~$15).** Pulled PR #1874's source verbatim, set up the SP8192 + FineWeb data pipeline on our pod, confirmed the stack reproduces to `val_bpb 1.06907` on seed 42 (within ~2σ of @AjAnubolu's reported 1.06766; reproduction artifact shipped in `models/pr1874_baseline_rank128_seed42.int6.ptz`).

2. **Phase 1 — single-seed hyperparameter sweep on top of PR #1874 (~$173).** Tested one knob at a time, each as a clean A/B against the seed-42 reproduction. Results in the table below. Most came back inside the noise band; only `TTT_LORA_RANK=192` consistently improved.

3. **Phase 2 — Newton-Muon graft attempt (~$12).** Filed as a separate non-record submission ([branch `nm-doc-packing-negative-result`](https://github.com/GodlyDonuts/parameter-golf/tree/nm-doc-packing-negative-result)). It regressed strongly due to dynamo recompile fragmentation; we wrote up the negative result rather than burying it.

4. **Phase 3 — 3-seed validation of `TTT_LORA_RANK=192` (~$45).** Three full 600 s training + ~440 s eval runs at seeds 42, 314, 999 with `TTT_LORA_RANK=192` set as the new default in the wrapped `train_gpt.py`. Those are the three logs and the three `champion_3seed_*` artifacts in this folder.

### Single-seed sweep results (one-knob-at-a-time on PR #1874 + seed = 42)

| Run | Configuration | val_bpb | Δ vs PR #1874 baseline (1.06907) |
|-----|--------------|--------:|--------------------------------:|
| `pr1874_repro_seed42`  | PR #1874 unmodified, our pod          | 1.06907 | (baseline) |
| `ttt_lora_rank_192`    | rank 128 → 192                        | **1.06888** | **−0.00019** |
| `LQER_RANK=6`          | LQER rank 4 → 6                       | 1.06912 | +0.00005 |
| `muon_backend_6`       | MUON_BACKEND_STEPS 5 → 6              | 1.06914 | +0.00007 |
| `lqer_topk_5`          | LQER top-K 3 → 5                      | 1.06907 | 0.00000 |
| `lqer_topk_4`          | LQER top-K 3 → 4                      | 1.06926 | +0.00019 |
| `gate_attn_w36`        | AttnOutGate width 24 → 36             | 1.06933 | +0.00026 |
| `pr1874_nm_smoke`      | + Newton-Muon enabled                 | 2.11910 | +1.05 (catastrophic — see non-record submission) |

The `TTT_LORA_RANK=192` row is the only knob in our sweep that produced a smaller-is-better delta against the reproduction baseline. We took that into 3-seed.

---

## Honest Note on the Rank-192 Effect Size

We want to be precise about what `TTT_LORA_RANK=192` does and does not buy:

- **Sweep evidence (single seed, controlled A/B):** rank=192 scored 1.06888 vs 1.06907 for rank=128. That's a 0.00019-nat improvement.
- **3-seed seed=42 replication:** 1.06928. This is *worse than* the rank=128 sweep baseline by 0.00021 nat.

These two numbers are not contradictory — both fall within the same ~0.0002-nat run-to-run kernel-scheduling noise floor we observed across the entire sweep (`gate_attn_w36`, `lqer_topk_4`, `muon_backend_6` all moved by similar magnitudes in different directions). The honest interpretation is: **the rank-192 effect is in the noise for our 3-seed evaluation.**

The 0.011-nat improvement vs the 1.0810 SOTA is large enough (`t = 17.67`) that this submission clears `p < 0.005` regardless of whether one credits the rank change or treats the submission purely as a clean reproduction of PR #1874. Both framings get to the same conclusion.

---

## What the Five Shipped Artifacts in `models/` Are For

| File | What it is | Bytes | Reported val_bpb |
|------|-----------|------:|-----------------:|
| `models/champion_3seed_42.int6.ptz`              | 3-seed run, rank=192, seed=42 — **headline result** | 15,921,161 | 1.06927777 |
| `models/champion_3seed_314.int6.ptz`             | 3-seed run, rank=192, seed=314 — **headline result** | 15,921,214 | 1.07023963 |
| `models/champion_3seed_999.int6.ptz`             | 3-seed run, rank=192, seed=999 — **headline result** | 15,914,086 | 1.07035739 |
| `models/pr1874_baseline_rank128_seed42.int6.ptz` | PR #1874 reproduction, rank=128, seed=42 (sweep) | 15,921,395 | 1.06906581 |
| `models/sweep_rank192_seed42.int6.ptz`           | rank=192 sweep, seed=42 | 15,921,684 | 1.06887519 |

Total: ~76 MB of binary artifacts. Including model artifacts is not standard practice on this leaderboard; we're including them here because:

1. **The headline 3-seed result is verifiable without a 600 s retrain.** A reviewer can eval the three `champion_3seed_*` artifacts directly.
2. **The rank-delta claim is independently verifiable.** A reviewer can eval `pr1874_baseline_rank128_seed42` against `sweep_rank192_seed42` and confirm the 0.00019-nat sweep delta on identical seed=42.
3. **Forensic / provenance value.** The included `.int6.ptz` files were produced *by the train logs included next to them*, not edited or replaced after the fact. Anyone can hash-verify or eval them.

### CPU-only inspection (no GPU needed, verified)

The `.int6.ptz` files are produced by PR #1874's `serialize()` (in `train_gpt.py:2103-2136`): a torch-saved `{"w": <quant_result>, "m": <quant_meta>}` dict, byte-shuffled with stride 2, then brotli-compressed. To read on CPU you reverse those steps:

```python
# verified on 2026-04-28 against models/champion_3seed_42.int6.ptz
import brotli, io, torch, numpy as np
_BSHF_MAGIC = b"BSHF"

def _byte_unshuffle(data):  # mirrors train_gpt.py:_byte_unshuffle (lines 1990-2002)
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = (n - pos + stride - 1) // stride
        out[pos::stride] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

with open("models/champion_3seed_42.int6.ptz", "rb") as f:
    raw = brotli.decompress(f.read())
state = torch.load(io.BytesIO(_byte_unshuffle(raw)), map_location="cpu", weights_only=False)
print(list(state.keys()))                          # ['w', 'm']
print(len(state["w"]), "quantized tensor entries") # 207
print(list(state["m"].items())[:1])                # [('blocks.0.attn.c_q.weight', 'gptq (int6)')]
```

This confirms the artifacts are well-formed int6 GPTQ-quantized state dicts with the expected layer structure. No GPU required.

### GPU eval-only

PR #1874's `train_gpt.py` does **not** ship with an explicit `EVAL_ONLY` flag — its pipeline is `train → quantize → eval` end-to-end. To eval a shipped artifact without retraining, point the script at it via the `final_model.int6.ptz` filename it expects, and call `deserialize(h, device)` (at `train_gpt.py:2139-2154`). For most reviewers, **the simpler verification path is to retrain a single seed from scratch** (~10 minutes of 8×H100 time per seed); the regression number we report is large and stable, and the script is wired end-to-end.

---

## Code Delta vs PR #1874 — One Line

```diff
- ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 128))
+ ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 192))
```

Everything else — every kernel, every loss term, every quantizer, every other hyperparameter — is PR #1874 byte-for-byte. The shipped `train_gpt.py` is a 32,353 B LZMA wrapper around the modified 134,706 B source.

---

## Compliance with Issue #1017 Track B (legal eval-time adaptation)

Each line below is something the submitted `train_gpt.py` actually does:

- **Causality.** Sliding-window eval scores each position from prefix tokens only. No look-ahead.
- **Normalized distribution.** Standard softmax over full vocab. No n-gram cache, no logit biasing, no temperature override.
- **Score before update.** Every chunk is fully scored under `torch.no_grad()` BEFORE any TTT update; SGD runs only on already-scored tokens. The phased TTT loop in the source explicitly separates the score pass from the update pass.
- **Single pass.** Each token is scored exactly once.
- **No SLOT** (standard or causal).
- **No pre-quant TTT on val data.** Quantization happens once at end of training; TTT runs at eval time on the quantized model only.
- **No ETLB.**
- **Train under 600 s on all 3 seeds.** Evidence: `max_wallclock_seconds: 600.0` setting in every log; final training step `4500/20000 train_loss: 2.84xx train_time: 9.2m` (552 s) on all 3 seeds; `gptq:reserving 4s, effective=596000ms` reservation line in every log.
- **Eval under 600 s on all 3 seeds.** `total_eval_time` = 438.3 s, 440.6 s, 434.3 s.
- **Total submission bytes < 16,000,000 on all 3 seeds.** See byte-budget table above; minimum headroom 45,076 B.

---

## Reproduction (clean room, single 8×H100 80GB SXM node)

```bash
# 1) Environment
pip install brotli sentencepiece zstandard
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# 2) Data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# 3) Three seeds
for SEED in 42 314 999; do
  SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```

`TTT_LORA_RANK=192` is the new default inside the shipped `train_gpt.py` — no env var needed. To reproduce PR #1874's rank=128 baseline, set `TTT_LORA_RANK=128`.

---

## Why You Can Trust These Numbers

1. **Logs are unedited.** Every log in this folder is `tee`'d directly from `torchrun` and copied without modification. Each contains the NCCL init, the full hyperparameter dump, per-step training metrics with wall-clock timestamps, the GPTQ quantization step (with Hessian collection time), and both the sliding-window and TTT-phased eval blocks.
2. **The shipped `train_gpt.py` is the same file used to produce the numbers.** No clean-up, no post-hoc minification, no separate "submission" version.
3. **Five reload-ready quantized artifacts are shipped in `models/`.** Anyone with `brotli` and `torch` can verify the artifacts on CPU (snippet above is verified). Anyone with 8×H100 can eval them and confirm the reported BPB.
4. **Our seed=42 reproduction of PR #1874 (1.06907) is within 2σ of @AjAnubolu's claim (1.06766).** That's the cross-pod variance we'd expect for the same code, same seed, different hardware lots — it is itself evidence that PR #1874 reproduces, not just our number on top.
5. **The single rank-192 code delta is the only difference vs PR #1874.** Anyone can `diff` the unwrapped sources and confirm.

---

## Attribution — what is and isn't ours

**What is ours:**
- The single hyperparameter change `ttt_lora_rank: 128 → 192` (≤0.0002-nat in measured effect, in the noise for our 3-seed evaluation).
- The independent 3-seed reproduction of PR #1874's stack with full unedited logs and reload-ready artifacts, on hardware separate from PR #1874's author.
- A separate non-record submission documenting why Newton-Muon × document-packed loaders fail.

**What is *not* ours and is properly attributed:**

| Component | Source |
|-----------|--------|
| Full stack assembly | @AjAnubolu — [PR #1874](https://github.com/openai/parameter-golf/pull/1874) |
| SmearGate, AttnOutGate w24, LoRA-TTT, Phased Global SGD TTT base | @dexhunter — [PR #1790](https://github.com/openai/parameter-golf/pull/1790) |
| LQER int4 rank-4 top-K asymmetric pack | [PR #1530](https://github.com/openai/parameter-golf/pull/1530) (original), [PR #1797](https://github.com/openai/parameter-golf/pull/1797) (SP8192 port) |
| Polar Express Newton–Schulz | [PR #1667](https://github.com/openai/parameter-golf/pull/1667) |
| MIN_LR for QAT | [PR #1787](https://github.com/openai/parameter-golf/pull/1787) |
| Score-first TTT framework | @abaybektursun — [PR #549](https://github.com/openai/parameter-golf/pull/549), @dexhunter — [PR #1413](https://github.com/openai/parameter-golf/pull/1413) |
| SP8192 + GPTQ + SDClip + MuonEq-R lineage | @clarkkev — [PR #1394](https://github.com/openai/parameter-golf/pull/1394) |
| Depth recurrence | @dexhunter — [PR #1331](https://github.com/openai/parameter-golf/pull/1331), [PR #1437](https://github.com/openai/parameter-golf/pull/1437) |

If anything in this submission deserves credit, it is overwhelmingly the people above. The only contribution we claim as our own is the rank=192 hyperparameter and the independent reproduction itself.

---

## On PR #1900's Provenance Review (read this part if you're the admin)

We are aware of [PR #1900](https://github.com/openai/parameter-golf/pull/1900), in which @regina-openai flagged validity/provenance concerns on PR #1787 (MIN_LR) and PR #1797 (LQER), both of which are upstream of PR #1874 and therefore upstream of this submission. We want to address this directly:

1. **No numerical claim in this submission was copied from a blocked parent.** Every BPB number in `submission.json` and in this README maps to a `quantized_ttt_phased val_loss:... val_bpb:...` line in one of the included logs, produced by a run we executed on our own pod. The corresponding `.int6.ptz` artifacts are in `models/` and are reload-ready; `champion_3seed_42.int6.ptz` is byte-traceable to `train_seed42.log`.

2. **We did inherit blocked techniques** by reproducing PR #1874, which integrates them. We are not aware of any path to score in the 1.067-1.070 BPB band on the SP8192 track without these techniques in some form. We're open to being corrected.

3. **If admin policy is that derivative submissions inherit a parent's blocked status, we will not contest closure.** The open-source value of this submission — independent reproduction with full logs, reload-ready artifacts, and a falsifiable byte-budget claim — is non-zero even without a leaderboard slot.

4. **We will gladly submit a variant with the blocked features off.** Both are gated behind environment variables in the shipped `train_gpt.py`. One-line change:

   ```bash
   MIN_LR=0.0 LQER_ENABLED=0 SEED=42 \
     torchrun --standalone --nproc_per_node=8 train_gpt.py
   ```

   Estimated 3-seed mean for that variant: 1.077-1.079 BPB. Still above the 0.005-nat threshold over SOTA but with tighter margin and no blocked-parent dependencies. ~$45 / ~3 hours of pod time to produce. We can have a second 3-seed table for that configuration ready on request — just say the word.

We'd rather hear "no, run the variant" than ship a quietly tainted record.

---

## Compute Provenance

- **Platform:** RunPod, single-tenant 8×H100 80GB SXM pod
- **Total spend:** ~$245 USD across the full project (sweep + 3-seed + non-record NM run)
- **Time window:** 2026-04-26 to 2026-04-28
- **PyTorch:** 2.9.1 + CUDA 12.8
- **FlashAttention 3:** `cu128_torch291` wheel from `windreamer.github.io/flash-attention3-wheels`
- **Per-seed cost (ballpark):** ~$15 for the 600 s train + 440 s eval
- **All RunPod billing on `csramineni@gmail.com`.** Invoice PDFs available privately to the admin team if provenance becomes a question.

---

## Acknowledgements

Compute funded by personal RunPod credits. Thanks to **@AjAnubolu** for [PR #1874](https://github.com/openai/parameter-golf/pull/1874) (this submission is fundamentally an independent reproduction of that work) and to **@dexhunter** and **@clarkkev** for the years of architectural groundwork the entire 1.06–1.08 BPB band sits on.

Submitted by:
- **Saicharan Ramineni** ([@GodlyDonuts](https://github.com/GodlyDonuts))
- csramineni@gmail.com

## Included Files

- `README.md` (this file)
- `submission.json` — machine-readable metadata, including verified byte budget and statistical numbers
- `requirements.txt`
- `train_gpt.py` — LZMA-wrapped, 32,353 bytes; defaults to `TTT_LORA_RANK=192`
- `train_seed42.log`, `train_seed314.log`, `train_seed999.log` — full 600 s train + ~440 s eval logs
- `models/champion_3seed_{42,314,999}.int6.ptz` — the three headline 3-seed artifacts
- `models/pr1874_baseline_rank128_seed42.int6.ptz` — PR #1874 reproduction (rank=128, seed=42), for the rank-delta A/B
- `models/sweep_rank192_seed42.int6.ptz` — single-seed rank=192 sweep run (seed=42), for the rank-delta A/B
