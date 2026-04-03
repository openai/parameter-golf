# Lab Protocol — Parameter Golf

_We are competing for #1. Every pod dollar is a decision._

---

## The One Rule

**ONE variable changes per test. If you change two, the result is meaningless and the money is gone.**

Before committing any gate script: diff it against the baseline. Count the differences. If it's more than one, stop.

---

## Pipeline: Gate → Full → Submit

```
Hypothesis
    ↓
Single GPU gate (2000 steps)
    ↓ passes?
8×H100 full run (600s, seed=444)
    ↓ beats baseline?
8×H100 confirmation (seed=300)
    ↓ both seeds confirm?
Submission branch → PR
```

**Never skip the gate.** A 2000-step single GPU run costs ~$0.50. A full 8×H100 run costs ~$3-4. Skipping the gate to save 10 minutes has cost us runs.

**Never submit on one seed.** Seed variance is real. Two seeds confirming = it's real.

---

## Cost Discipline

- 8×H100 SXM: ~$1.67/hr per GPU = **$13.36/hr for 8×**
- Full 10-min run (with pod overhead): **~$3-4**
- Per-race budget: **~$15**
- Do not suggest a run without a validated gate result or a clear hypothesis

**Reproducing a score we already own = no.** Never re-run a baseline we control unless the architecture changed.

---

## Checkpoints

After every full run, `final_model.pt` gets copied to a unique name immediately:

```bash
cp final_model.pt checkpoints/EXP_s${SEED}_$(date +%Y%m%d_%H%M%S)_bpb${BPB}.pt
```

The pod gets destroyed. If the checkpoint isn't saved before that, it's gone.

---

## Script Standards

- Every experiment lives in `experiments/<Name>/`
- Every experiment has: `run.sh`, `gate.sh` or `gate_1gpu.sh`, `RESULTS.md`
- `run.sh` uses `train_gpt.py` from the same directory (symlink or copy)
- Scripts are committed and pushed before the pod fires
- Never paste raw commands. Always a `.sh` file.
- Log files go to `experiments/<Name>/results/` or `logs/`

---

## Naming

- Confirm experiment names before creating directories
- Active series: `Bandit_Wagon_V`, `Bandit_Wagon_V_Cannon`, etc.
- Superseded experiments → `experiments/archive/`
- Never reuse a name from a previous run

---

## SOTA Garage

Three active models:

| Track | Model | BPB | Size |
|-------|-------|-----|------|
| Neural | Rascal II | 1.10987 | 15.44MB |
| Crawler | BW5 seed=444 | 1.18672 | 8.61MB |
| Compression | FX_WING_DELTA | 0.2233 | — (model lost) |

**Submission branch protocol:**
1. Never submit from TEST_LAB
2. Create dedicated branch → push to Open-parameter-golf-1 fork → PR to openai/parameter-golf
3. Every PR needs: `submission.json`, logs, README with reproduce instructions

---

## Experimental Design

- Proxy deltas (500 steps, 1 GPU) inflate **5–15×** vs full run. Never promote from proxy alone.
- Gate (2000 steps, 1 GPU) is the minimum signal to trust.
- SWA kicks in at step ~7650. Results before that step are pre-SWA.
- Wallclock budget is 600s. Extra parameters cost convergence speed — account for this.
- `COMPILE_FULLGRAPH=1` is now baseline for all BW5+ experiments.

---

## Seeds

- Primary: **444**
- Confirmation: **300**
- Never use 1337 for new experiments.

---

## Submission Checklist

- [ ] Two seeds confirmed, both beat baseline
- [ ] `submission.json` present
- [ ] Logs committed
- [ ] README with reproduce instructions
- [ ] File size ≤ 16MB
- [ ] Score-first always (no training on val before scoring)
- [ ] Branch is NOT TEST_LAB
