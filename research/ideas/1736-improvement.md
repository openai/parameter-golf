# 1736-improvement — rebase baseline onto #1736 and stack levers on top

**Status:** 🟢 ACTIVE — adopted as our new baseline lineage on 2026-04-20, replacing the direct-from-#1493 approach of spec-000. See `diary/2026-04-19-frontier-scan.md` and `diary/2026-04-19-frontier-map.md` for the evidence base behind this switch.
**Expected Δ (cumulative on top of our spec-000 baseline 1.0810):** ~–0.015 to –0.025 bpb if the full stack lands.
**Source:** modal-scenario analysis of the 2026-04-19 frontier snapshot. #1736 is the credible open-PR frontier under the "lossy casefold illegal, CaseOps legal, pre-quant TTT illegal" ruling — roughly the 50% outcome.

## Context

Up to spec-007 we were iterating levers against the merged-SOTA #1493 recipe (1.0810). In the 10 days between 04-09 and 04-19 the frontier moved substantially past that on unmerged branches. After filtering out:

- **BANNED** — Trinity / SLOT / N-gram stacking (#1246, #1722, #1723)
- **BROKEN** — GDN-family byte-accounting bug (#1698, #1711, #1712, #1734; see Issue #1719)
- **DISPUTED-unlikely** — pre-quant TTT (#1735, #1738) — physically implausible vs Issue #1017 corpus-level TTT ceiling (~0.0003 bits); likely DQ
- **DISPUTED-lossy** — casefold (#1578, #1693) — Issue #1604 leans toward lossless-only

… the credible ceiling for our push is **#1736 at 1.0655** (dexhunter, 2026-04-19): SP8192 + CaseOps + attn-out gate + quant-gate + Loop45 + phased TTT.

Sibling reference: **#1729 at 1.0678** (romeerp) — same tokenizer, simpler stack (CaseOps + tapered WD, no gates). Useful as the "tokenizer-only" checkpoint in our reproduction.

## Plan — reproduce #1736 directly, then iterate

### Spec 008 — #1736 full reproduction (new baseline)

Reproduce #1736 end-to-end. Single spec, full stack — no intermediate #1729 step, because we're committing to #1736 as the base regardless of whether individual levers attribute cleanly.

- Data: pull `romeerp/parameter-golf-caseops-v1` from HuggingFace (~20–30 GB).
- Code: port #1736's pinned commit — CaseOps operator tokens (`\uE001`–`\uE003`) + byte-sidecar eval plumbing, attn-out gate (per-head, zero-init, 1,056 params), SmearGate (13 params), quant-gate, Loop45 depth recurrence, multi-phase SGD phased TTT.
- Hardware: 2×H100 mini first → 8×H100 3-seed if mini lands within ±0.003 of target.
- **Target:** val_bpb ≈ 1.0655. This becomes our new local baseline; subsequent specs report Δ vs it.
- If mini misses badly: debug as one integration (tokenizer, gates, or TTT path) rather than splitting into separate specs. Most likely failure modes are data-shard or CaseOps-plumbing mismatches.

### Spec 009 — first novel lever on top of #1736

Candidates from our existing research, roughly ranked by expected Δ × safety:

1. **SpinQuant V1 (Hadamard rotation of weights before GPTQ)** — from #1695, quant-orthogonal, claimed ~–0.005. Lowest risk: a witnessed independent lever we just haven't tried yet.
2. **`per-group-bit-allocation`** — our own idea, quant-orthogonal, could be ~–0.002.
3. **`ar-selfgen-gptq-calib`** — our own idea, quantization calibration via self-generated AR samples.
4. **`layerwise-lr-decay`** — optimization-side, small but clean.

SpinQuant is the obvious first pick — it's the safe stack that nobody has combined with #1736 yet.

### Spec 010+ — our own novel levers

Reserve these for things that aren't already on the frontier. Candidates TBD from spec 006/007 dynamics findings and fresh ideas.

## Why this switch is defensible

- **Every piece of #1736 has an independent witness.** CaseOps from #1729 (romeerp), attn-out gate from #1667 (MarioPaerle), multi-phase TTT from #1626/#1700, base arch from #1530/#1523. Not one PR's claim, many.
- **dexhunter is credible.** Same person who found the GDN byte-accounting bug (#1719); reputational cost of posting a fake is high.
- **Physical-plausibility check passes.** Each lever's claimed Δ is within magnitude other PRs have measured independently. No free lunches.
- **Legality tail risk is small.** Lossless CaseOps survives #1604's modal ruling; no pre-quant TTT, no SLOT, no banned eval tricks.

## Risks

- **Tokenizer DQ tail (~10–15%).** If #1604 rules against *all* non-identity normalizations (including CaseOps operator tokens), the base dies. Mitigation: our own ideas in specs 010+ are mostly quant/optimizer-side, so they'd port to a non-CaseOps base with modest rework.
- **Reproduction miss.** #1729's dataset is retokenized FineWeb; if we can't reproduce the number within ±0.003 on mini, something is off (tokenizer version, data shard, hparam) and we'd need to debug rather than push forward.
- **Integration cost.** Spec 008 is ~5–6 h execution (full #1736 port + mini + 3-seed). One execution session to stand up the new baseline before our own levers start.

## Cumulative landing target

| Stack | Expected val_bpb | Vs merged SOTA | Vs spec-000 |
|---|---|---|---|
| spec-000 (merged SOTA replica) | 1.0810 | 0 | 0 |
| spec 008 (#1736 reproduction) | ~1.0655 | −0.0155 | −0.015 |
| spec 009 (+SpinQuant) | ~1.0605 | −0.020 | −0.020 |
| spec 010 (+novel lever) | ~1.055–1.060 | −0.025 | −0.025 |

Lands us credibly mid-pack on a legal submission under the modal ruling. Matching or beating #1736 requires the spec-010 lever to land; otherwise we're tied with the current frontier, which is still meaningful if the disputed-frontier PRs get DQ'd.
