# Parameter Golf — Blog Post v2 Outline

## Angle and Tone

- **Reporter, not insider.** Tell the story from the outside looking in. The reader doesn't need to care about BPB or quantization to stay engaged — techniques are plot devices, not the subject.
- **Exciting, not technical.** Less mechanism, more drama. The score numbers, the leaderboard moves, the rulings — these carry the narrative.
- **Dynamics, not just explanation.** There's tension (who's winning?), mystery (how did they score 0.9?), revelation (it was a bug), and vindication (the careful engineers win anyway).

---

## Structure

### 0. The Hook

OpenAI released a public competition: train the best language model that fits in 16 MB, in 10 minutes, on 8 H100s. Small stakes on paper. But what unfolded over six weeks had everything — cutting-edge technique, controversy, and a lesson about what it actually takes to make progress at the frontier.

*Today we give an overview of what happened.*

---

### 1. The Competition

Dense, no C1–C4 rule taxonomy. Two things to convey:

- **What is being scored.** A language model is a probability distribution over text. Better model = closer to the true distribution = lower bits-per-byte. Analogy: imagine you're trying to guess the next word in a sentence — a good model is one that's rarely surprised.
- **What makes it hard.** You have 16 MB and 10 minutes. Every byte spent on model weights is a byte not spent elsewhere. You have to be clever.

No more than 2–3 short paragraphs.

---

### 2. The Model

Three beats:

1. **What an LLM actually is.** Attention + MLP layers stacked on top of each other. One sentence on each. Keep it concrete — attention lets each token look at all other tokens; MLP is where the model "thinks" about what it saw.

2. **The baseline.** OpenAI's starting model was already not a simple transformer. Walk through the key baseline features briefly. Note the starting score: 1.2244 BPB.

3. **The evolution.** Spotlight: **depth recurrence** — run layers 3–5 in a loop, multiple passes per forward step. Free test-time compute within the 16 MB budget. Activates at 35% of training wallclock for throughput reasons. Intuitive analogy: instead of reading a sentence once, you read it again with what you just learned.

   Everything else (XSA, GPTQ, CaseOps, phased TTT, parallel residuals, SmearGate, etc.) demoted to a **footnote** or compact table. The point is the stacking story, not an exhaustive list. Final SOTA: 1.0565 BPB.

---

### 3. Too Good to Be True

**Hook:** The leaderboard wasn't just inching forward. Periodically, a submission would appear claiming a score far below anything else — something in the 0.9x range, shattering the rest of the field. Everyone noticed.

Two main drivers:

- **N-gram tilt.** A small external model that nudged the probability distribution toward likely next tokens, based on n-gram statistics. In theory legal. In practice: the implementation contained a bug — the boundary lookup used the same token it was conditioning on (C1 violation). The gain evaporated once the bug was identified and fixed. A clean version survived but contributed only modestly.

- **PPM-D.** A classical byte-level compression algorithm bolted on as a second opinion. PPM-D tracks context patterns and makes predictions; the idea was to blend its distribution with the model's. Initial results were dramatic. On inspection: the blending formula didn't produce a valid probability distribution (C2 violation) — it didn't sum to 1. The gain was an artifact of the broken math. A corrected submission showed almost no improvement over the base model.[^ppmd]

**The lesson.** These weren't acts of bad faith — they were honest mistakes caught by careful peer review. But they point to something deeper. A well-trained language model is already a calibrated entropy estimator: where it predicts a flat distribution, the text really is hard to predict. The correlation between the model's uncertainty and the true information content is tight. PPM-D and n-gram statistics track exactly the same "easy" tokens the NN already handles well. For an external signal to help, its errors would have to be *uncorrelated* with the model's — it would need to be uncertain where the NN is confident. That turns out to be extremely hard to achieve. There is no silver bullet. The progress in this competition was incremental, compounding, and hard-won.

[^ppmd]: The PPM-D case is more technically interesting — a longer discussion of why the blending formula fails and what a correct version would need to look like is left for a future post / appendix.

---

### 4. Drama on the Last Day

Set the scene: the field heading into the final days, front-runners clustered around 1.05x BPB.

Then: someone noticed something odd. A submission that had quietly dominated the leaderboard for weeks was retested — and the score looked too clean. Investigation followed.

**The analogy:** training the model on the exam itself. The validation set used to score submissions had a default flag — `--val-docs=10000` — that caused an 80% overlap with the training data. Anyone who had used the standard CaseOps setup since PR #1729 was, without knowing it, training on the held-out test documents. A good score, but not a fair one.

Once the overlap was quantified, a wave of front-runners were disqualified. Almost everything submitted since the CaseOps era was tainted. The leaderboard reshuffled.

What remained: a small set of clean submissions, and an open door.

**Ending with flair.** The competition closed with a picture-perfect finish — the final accepted submission (PR #2135) was clean, incrementally better than anything else that remained standing, and arrived courtesy of a single well-placed hyperparameter change on top of six weeks of careful engineering. Exactly the kind of finish the competition deserved.

---

## Thematic Arc

- People tried shortcuts (n-gram exploits, broken compression hybrids, tokenizer tricks).
- The shortcuts that looked too good were either bugs or violations.
- The gains that held came from making the transformer itself more expressive.
- The competition was a proof-by-exhaustion that there is no clever hack that replaces careful engineering at the frontier.
- The final word belongs to the model.
