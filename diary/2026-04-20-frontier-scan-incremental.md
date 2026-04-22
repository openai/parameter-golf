# 2026-04-20 — frontier scan (incremental since pt4)

**Baseline scan:** `diary/2026-04-20-pr-scan-pt4.md` (through PR #1747, finished ~05:47 UTC / 13:47 Asia/Shanghai).
**This scan:** open PRs on `openai/parameter-golf` updated after that cutoff.
**Scope:** record track only (non-record submissions noted but not deep-dived).

## New PRs since pt4

| # | Author | Title | Track | Claimed bpb |
|---|---|---|---|---|
| 1748 | elad-simbalista | basic submission improving baseline | **non-record** | — (empty body) |
| 1749 | gracebml | GDN-Hybrid + Legal Score-First TTT + Full-Hessian GPTQ Int6 | **non-record** | 1.0996 (1×H100, 28% budget) |

No new record-track PRs. No updates to previously-scanned record-track PRs.

## Per-PR verdicts

### #1748 elad-simbalista — skip
- **Track:** `records/track_non_record_16mb/2026-04-20_EMA_Muon_row_GPTQ/`. Not our track.
- **Body:** empty. No claimed bpb, no description.
- **Files:** 3075-line train_gpt.py + log. No signal without author context.
- **Verdict:** `other` / not-relevant. Nothing to port.

### #1749 gracebml — note, don't port
- **Track:** `records/track_non_record_16mb/`. Not our track.
- **Claim:** 1.0996 val_bpb on 1×H100 wallclock-capped at 4800s, only 5610/20000 steps. Author extrapolates "meaningfully below 1.08" at 8×H100 — still worse than #1736's 1.06549, so no record-track threat.
- **Novel levers worth logging for idea backlog:**
  - **"Legal Score-First TTT"** — explicit `inference_mode()` eval pass, then AdamW step on the just-scored chunk. This is the score-before-update pattern pt4 flagged as *legal and under-used* (sits downstream of TTT, cannot be absorbed). #1749's implementation is cleaner than #1555's Tap-In variant.
  - **Full-Hessian GPTQ with Cholesky compensation** — claims +0.013 bpb int6 roundtrip (vs typical +0.05–0.10). Composes with SpinQuant/rotation. Could stack with the #1695/#1732 quantization cluster we already deemed clean.
- **Caveats:**
  - Uses GatedDeltaNet. pt4 flagged the FLA/GDN cluster for byte-counting bugs (#1698/#1711/#1712/#1734). Need to verify #1749's byte accounting before trusting any sub-claim, even though it's non-record.
  - Score-First TTT legitimacy: pt4 already noted score-before-update is legal. No dispute here.
- **Verdict:** `clean` in principle (non-record track, score-first TTT is legal). Low priority — worse than our baseline, different track.

## Leaderboard deltas vs pt4

**None.** Record-track frontier unchanged:
- Clean floor: ~1.071 (#1667, #1727, #1695)
- Tokenizer-disputed best: **#1736 @ 1.06549** (our baseline)
- Pre-quant-TTT-disputed best: #1738 @ 1.0354 (likely illegal)

## Action items

- **Idea backlog:** add "Full-Hessian GPTQ + Cholesky compensation (from #1749)" as a potential compose-lever for the quant stack. Low priority vs current spec 011/012/013 sequencing.
- **No changes** to spec queue or budget plan.

## Meta

- Incremental scan cost: ~2 `gh pr view` calls + classification. Would have been wasteful to rescan 200 PRs.
- Cadence feel: if scanning daily, most days will look like this (0–3 new PRs, usually non-record noise). Rare days will surface a real record-track threat. Daily scan still cheap enough to be worth it.
