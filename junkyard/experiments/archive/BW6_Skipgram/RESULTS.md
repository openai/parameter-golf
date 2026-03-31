# BW6_Skipgram — Gate Results

## Architecture

BW5 + `TRIGRAM=1` — trigram hash `(t-2, t-1, t)` added into existing bigram embedding table.
Zero extra parameters. Same 2048-slot table, same projection, same scale.

Parent: `crawler/2026-03-29_BW5/` (champion: 1.18672385 BPB, 8.61MB)

---

## Gate: 8×H100, 2000 steps, seed=444

| ARM | TRIGRAM | step_avg | raw_bpb | int6_sw_bpb | size_bytes |
|-----|---------|----------|---------|-------------|------------|
| BW6SK-00 | 0 (bigram only) | 74.53ms | 1.3083 | 1.28951966 | 9,482,608 |
| BW6SK-01 | 1 (bigram+trigram) | **74.47ms** | **1.3088** | **1.28965847** | **9,342,986** |
| delta | — | **−0.06ms** | **+0.0005** | **+0.00014** | **−139,622** |

### Speed: PASSES
- −0.06ms. Zero overhead. As expected — one extra embed lookup.

### Quality: FAILS (null result)
- raw_bpb: +0.0005 worse
- int6_sw_bpb: +0.00014 worse
- Both deltas are within cross-run variance (~0.0003 BPB). This is not a hard failure — it is a **null result**. Trigram neither helps nor hurts.

### Size: Interesting — −140KB
- Trigram arm is 139KB smaller despite identical parameter count.
- Hypothesis: the additional trigram hashing signal produces slightly different weight distributions that compress more efficiently under int6+zstd. Quant_gap may be marginally tighter.

---

## Verdict: DOES NOT PROMOTE — Null result

**The trigram signal is noise at 2000 steps on the crawler.** Delta is well inside variance. This is meaningfully different from pyramid's hard failure (+0.034) — trigram is neutral, not harmful.

**Why trigram may not signal here:**
- The crawler already loops 3× over the same weights — the recurrent structure may already implicitly capture (t-2, t-1, t) context via accumulated hidden state across loops. Trigram adds a static lookup that the recurrent path has already approximated.
- The bigram embedding table (2048 slots) may already be saturated — the trigram hash collides heavily into the same 2047 slots, diluting rather than enriching the signal.
- The neural SOTA (Rascal II) is a standard transformer where trigram provides raw context not otherwise available. The crawler's recurrent loops partially substitute for this.

**Concept notes:**
- A larger vocab table (`BIGRAM_VOCAB_SIZE=4096+`) might reduce collision and let trigram signal through
- Dedicated trigram table (separate params, ~128K) would eliminate collision entirely but adds size
- The size benefit (−140KB) is worth noting if we ever need artifact compression tricks

**Note on gate script:** The `int6_sw_bpb:` grep pattern is broken — the log format is `final_int6_sliding_window_exact val_loss:X val_bpb:Y`. Values above were extracted manually from the raw log output.
