# Parameter Golf: Six Weeks of Compressing Language Models to Their Limits

---

> **What does a 14% compression improvement actually look like?**
> Both models below were trained on the same dataset, in the same 10 minutes,
> on the same hardware. The only difference is what the community built in six weeks.

*From the FineWeb validation set, same 150-token context fed to both models:*

| | SP1024 Baseline · 1.22 BPB | Near-SOTA Model · 1.06 BPB |
|---|---|---|
| *"…his driver license and auto insurance had been cancelled in October. This poses a problem for"* | the estate, though internalieness to gets the pass. It is a fat check that's gives ges auto insurance company… that's a **fairyyyyyyyyyyyyyyyyyyyyy** | the estate's creditors who need to work out a payment, especially with the new insurance regulations in october… |
| *"…New host Steve Byrnes was feeling his way through this"* | season. He had just had a short, **bad weekkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk** | season's show, and while he played one of the best nights of his career at the same time, he hadn't been able to really shine since the first Nashville game… |
| *"…Prosecutors now have until"* | Jan. 2, the **Supreen House** prosecutor sentences on a **murded** case… **Mazzzzaglia**… **charies's''''''** | Dec. 18 to sign off on a case after the new hampshire attorney general's office says they have more time… |

*Prompted with "You are a helpful AI assistant. Please answer the following question." (not very good at general conversation):*

| | SP1024 Baseline · 1.22 BPB | Near-SOTA Model · 1.06 BPB |
|---|---|---|
| *"The history of the internet began in the 1960s when"* | the Netherlands had established a strong supply of information about the **Canadian government and its supply chains**. The German government, in its own right, is commonly established as a matter of legal… | the internet became the **first electronic commerce tool** that was used by many people, including children, to connect online. the internet is now used by more than 3 billion users worldwide… |
| *"What is 15 multiplied by 13? Let me work it out:"* | **1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 14. 13.** 14. 15. 16. 17. 18. 19. **19.** 20.1 | 15 multiples 13 have 15, for example, 15, for **1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000** |
| *"To bake a chocolate cake, you need flour, eggs, butter, and"* | sugar… you need to have **an infinite wooden cake**. Otherwise, you need to be taken as close to an… | salt… Add 1 tablespoon of flour to 1/4 cup of milk and pour over your cake. Add 1/4 teaspoon of sugar… Cook a lightly toasted chocolate cake. Make cupcakes Ã 3 |

---

## Part 0: The Competition

In March 2026, OpenAI announced Parameter Golf: a community competition to build
the best language model that fits inside 16,000,000 bytes, trained in exactly
10 minutes on 8 H100 GPUs. The competition ran for six weeks, attracted over
2000 PR submissions, and ended with the community pushing a baseline model from
1.22 bits-per-byte down to 1.056, a compression improvement of roughly 14%
through pure algorithmic ingenuity, with no change to the hardware budget or
the data.

Parameter Golf sits within a family of challenges sometimes called L(N)
optimization. Sister challenges include the NanoGPT Speedrun (fixed time,
minimize loss) and the NanoGPT Flowrun (fixed data, minimize loss). Here
the constraint is different: fixed size, with both training time and model
capacity tightly capped.

Before we look at the models, let us first understand how the competition works.

**The setup.** The dataset is a slice of FineWeb, a large corpus of filtered
web text. Competitors train on a fixed training split and are scored on a
held-out validation set they cannot access during training. The training window
is 10 minutes of wall-clock time on 8 H100 GPUs. Triton kernel compilation
happens beforehand (competitors pre-warm their kernel caches as a separate
step and it does not count against the clock), and GPTQ quantization runs after
the clock stops as a post-processing step. The final artifact (which must fit
inside 16,000,000 bytes) is the compressed model weights plus the
`train_gpt.py` code that runs them.¹ Evaluation gets a separate 10-minute
window; test-time training (TTT) happens within that window, consuming some of
the budget before the scoring pass begins.

**How scoring works.** The task is next-word prediction. The model sees text
one token at a time, left to right, and at each position must output a
probability distribution over all possible next tokens. If the next token is t,
the model earns log₂ p(t) bits, close to zero when confident and right,
increasingly negative when wrong. Over a full document t₁, t₂, …, tₙ, the
total score is:

  −∑ₖ log₂ p(tₖ | t₁, …, tₖ₋₁)

Dividing by the total number of bytes in the document gives **bits-per-byte**
(BPB). The byte-level denominator matters: a token that encodes two bytes
counts as two bytes in the denominator, so longer tokens are neither rewarded
nor penalized just for being large, something that turns out to be important
later. Lower BPB is better.

A useful analogy: imagine a multiple-choice exam with 256 options at every
question. Instead of circling one answer, you must assign a probability to each
option, with all 256 summing to 1. Your score on each question is −log₂ of the
probability you assigned to the correct answer. Assign 100% to the right answer:
score 0 (perfect). Assign equal weight to all 256 options: score log₂(256) = 8.
A model with no knowledge of text does exactly that (uniform over 256 bytes),
scoring 8 BPB. A perfect oracle scores 0 BPB. The baseline transformer the
competition started from scored **1.2244 BPB**.

**The rules: C1–C4.** The competition launched without a complete ruleset. As
participants found increasingly creative and occasionally questionable ways
to improve their scores, four constraints were codified mid-competition through
community discussion in Issue #1017:²

- **C1: Causal eval:** the probability assigned to token tₖ must depend only
  on the tokens before it. No peeking at what comes next. In the multiple-choice
  analogy: you must commit to your probability distribution before seeing which
  answer is correct. What happened in practice: certain submissions fed the
  answer token into the model as part of the context while scoring that very
  token, effectively choosing all options, seeing which one is right, then
  going back and erasing the wrong ones.

- **C2: Normalized distribution:** the output must be a valid probability
  distribution summing to exactly 1. You cannot manufacture artificially low
  entropy by assigning more than 100% total mass across tokens. In the analogy:
  you cannot assign 90% probability to each of four options.

- **C3: Score before update:** in TTT, a chunk of validation text must be
  fully scored *before* any gradient step is applied to it. Training on a chunk
  and then scoring it means the model has already adapted to those exact tokens.
  In the analogy: you cannot read the exam questions, go study them, then come
  back and sit the exam.

- **C4: Single pass:** each validation token is scored exactly once. No
  re-scoring after adaptation. In the analogy: you cannot retake the same
  question after seeing the answer.

In the six weeks the competition ran, the score fell from 1.2244 to 1.0565, a
drop of 0.168 BPB. [TODO: score progression graph]

In the next section we trace the key techniques that drove this improvement.

---
¹ The artifact is the compressed model weights plus all code in `train_gpt.py`.
The cap is 16,000,000 bytes (decimal), not 16 MiB (16,777,216 bytes), a
distinction that matters: the actual budget is about 4.5% smaller than a
"16 megabyte" headline implies. No external downloads or network calls are
permitted during evaluation.
² Issue and PR numbers refer to the openai/parameter-golf GitHub repository:
`https://github.com/openai/parameter-golf/issues/{number}` or `/pull/{number}`.

---

## Part 1: Model Comparison and Techniques

There are five components to a Parameter Golf submission: the tokenizer,
the model architecture, the training setup, the quantization pipeline, and
post-training adaptation. Let's start with the base model that OpenAI
provided at the start of the competition.

*(This section assumes basic familiarity with transformer architecture at
the level of NanoGPT (attention, residual stream, MLP layers). If that's
new territory, Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
is the right starting point.)*

---

### The Baseline

OpenAI's starting model was already not a simple transformer, not a plain
NanoGPT-style stack of attention and MLP layers. The baseline was more
carefully engineered than that. Let's break it down.

**Tokenizer.** The baseline used SentencePiece with a 1024-token vocabulary
(SP1024), trained on the same FineWeb corpus used for scoring. A 1024-token
vocabulary is quite small by modern standards; GPT-2 uses 50,000 tokens.
The small vocabulary keeps the embedding table compact, which matters when
your entire model has to fit in 16 MB.

**Model architecture.** The baseline was a 9-layer, 512-dimensional transformer
with three structural additions worth calling out:

- **U-Net skip connections:** in a standard transformer, each layer feeds
  only into the next. The U-Net pattern adds direct connections that skip the
  middle of the network. The 9 layers are split into an encoder half (layers
  1–4), a bottleneck (layer 5), and a decoder half (layers 6–9). Each decoder
  layer receives the output of its mirror encoder layer as an additional
  residual, weighted by a learned scalar *w*:

  ```
  Standard:  h_l = Block_l(h_{l-1})

  U-Net:     h_l = Block_l(h_{l-1})               for l ≤ 5
             h_l = Block_l(h_{l-1} + w · h_{10-l}) for l ∈ {6,7,8,9}
  ```

  Concretely: layer 6 gets a skip from layer 4, layer 7 from layer 3, layer 8
  from layer 2, layer 9 from layer 1. The early-layer representations, which
  tend to capture local, surface-level patterns, are fed directly into the
  late layers alongside the deep representations. This is the same idea that
  made U-Net famous in image segmentation.
- **Grouped-query attention (GQA)** and **rotary positional embeddings (RoPE)**,
  both standard in modern LLMs, just not in NanoGPT.
  GQA shares key-value heads across query heads to cut parameter count; RoPE
  encodes position by rotating query and key vectors rather than adding learned
  position embeddings.

**Training.** The baseline used the **Muon optimizer**, a popular choice
over AdamW. The learning rate follows a
warmup-then-warmdown schedule, rising over the first few steps then decaying
to zero by the end of the 10-minute window.

**Quantization.** Every weight in the model is a number stored with some
number of bits: more bits means more precision, but also a larger file.
After training at bfloat16 (16 bits per weight), the baseline rounded its
MLP weights down to int6 (6 bits), shrinking what would otherwise be a
~70 MB model into something that fits the 16 MB budget. The game is
managing the tradeoff: aggressive enough to fit, not so aggressive that
predictions degrade. The baseline's approach was simple: round the biggest
chunk of parameters and leave everything else alone.

**Post-training adaptation.** The baseline did none. During the 10-minute
evaluation window, it simply ran the model forward on the validation text
and recorded the scores. Later models would use this window to actively update
their weights in response to what they were seeing, a technique called
test-time training (TTT). The baseline serves as the clean reference point
before that complication enters.

This baseline scored **1.2244 BPB**. By the end of the competition, the best
submission had reached **1.0565 BPB**, the same hardware, the same data, the
same 10 minutes, and a model that had been rebuilt almost from scratch across
every one of those five components. The rest of this section traces how.

---

### The Evolution

#### Tokenizer

The vocabulary grew in two steps: SP1024 → SP4096 (PR #1218) → SP8192
(PR #1394). Counterintuitive, since a larger vocabulary means a larger
embedding table and therefore more parameters. The payoff comes from the
BPB denominator: it counts *bytes*, not tokens. A token that encodes two
bytes contributes two bytes to the denominator, so a model that packs more
bytes per token earns a lower BPB for the same prediction quality. The
embedding cost is real, but it pays for itself.

The more interesting tokenizer story is about capitalization. An SP8192
vocabulary wastes slots on case variants: "the", "The", and "THE" are
three separate entries for the same word. PR #1578 introduced casefold:
lowercase everything before tokenizing, freeing up hundreds of duplicate
slots. The problem: casefold permanently destroys information, and a model that
cannot recover the original casing cannot correctly score it. Ruled illegal
in Issue #1604.

CaseOps (PR #1729) solved this losslessly. Four control tokens are reserved
in the vocabulary (TITLE, ALLCAPS, CAPNEXT, ESC) and capitalization is
encoded inline before tokenizing:

```
"The NASA launched."  →  "TITLE the ALLCAPS nasa launched."
```

The original text is fully recoverable. Nothing is discarded. The ~8188
remaining vocabulary slots are now entirely free of case duplication, and
the control tokens are cheap to predict (capitalization follows clear
patterns: sentence starts, acronyms, proper nouns) so the model pays
very little BPB on them.

SP8192+CaseOps remained the tokenizer frontier for the rest of the
competition. CaseOps also turned out to be the center of an unexpected
controversy, not about the tokenizer itself. A significant fraction of
CaseOps submissions were later disqualified, for reasons we'll return to
in Part 3.

---

#### Model Architecture

Here we walk through the most significant architectural changes from the
baseline to the final model.

- **Bigger and deeper (PR #86, PR #1218).** The final model is 11 layers
  with a 4× MLP width, up from the baseline's 9 layers and 2× MLP width.
  Wider MLPs give each layer more capacity; more layers give the network
  more processing steps. Both changes came with higher weight decay to keep
  the weights compressible under quantization.

- **Depth recurrence (PR #1344).** The most structurally novel change. In a
  standard transformer, each layer runs exactly once per token. The final
  model loops layers 3–5 three times per forward pass, the same three
  layers, the same weights, applied three times in sequence:

  ```
  standard:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

  with loop: 1 → 2 → 3 → 4 → 5 → 3 → 4 → 5 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
                       └──────────────── ×3 ────────────────┘
  ```

  17 effective processing steps from 11 physical layers, at zero additional
  parameter cost. PR #1344 introduced the structure with 2 passes over
  layers 3–5; the final configuration of 3 passes was established in later
  records. The model does not start training with the loop active; we cover
  the training curriculum in the next section.

- **Parallel residuals (PR #1204, from layer 8 in PR #1529).** In a
  standard transformer layer, attention and MLP run sequentially. From
  layer 8 onward, the final model runs them in parallel: both branches
  receive the same input x, and their outputs are added together:

  ```
  standard:   h = x + Attn(x),  then  h = h + MLP(h)

  parallel:   h = x + Attn(x) + MLP(x)
  ```

  This squeezes more computation out of each layer without adding parameters.

- **XSA (PR #287).** XSA is gaining traction in the community as a simple
  attention improvement. In standard attention, each token strongly attends
  to itself, its own value vector dominating the output and acting as a
  near-identity shortcut. XSA removes this self-contribution by projecting
  it out of the attention output:

  ```
  standard:   y  = Σⱼ αⱼ vⱼ

  XSA:        y  = y − (y · v̂) v̂       where v̂ = v / ‖v‖
  ```

  The component of y along the current token's own value vector is
  subtracted out, forcing the model to actually use context from other
  tokens. Applied to all layers, it was one of the larger single
  architectural improvements in the competition.

- **Smaller additions.** A learned SmearGate blending each token with its
  neighbor (PR #162, reintroduced in PR #1667), a multiplicative attention
  output gate first in PR #1667 then narrowed in PR #1787, a LeakyReLU²
  MLP activation replacing relu² (PR #493), partial RoPE with layer-norm
  scaling (PR #315), and sigmoid-gated U-Net skip connections (PR #289).

---

#### Training

Several training hyperparameters were refined from the baseline: the LR
warmdown schedule gained a minimum floor of 0.1× rather than decaying all
the way to zero (PR #1787), and the Muon optimizer's internal Newton-Schulz
iteration was retuned with better polynomial coefficients (PR #1344/#1787).
The more structural training changes are EMA, the recurrence
curriculum, and progressive context lengthening.

- **EMA (PR #287).** One of the most impactful single changes in the
  competition. Instead of evaluating the final checkpoint's weights
  directly, the model maintains an exponential moving average of all past
  weight iterates throughout training. The eval model is this running
  average, not the last step. Because SGD iterates are noisy, the average
  sits in a flatter, more stable region of the loss landscape and
  generalizes significantly better. EMA replaced stochastic weight
  averaging early in the competition; the decay value introduced in PR #287
  was never revisited through the final SOTA.

- **Depth recurrence curriculum (PR #1420).** The loop over layers 3–5
  does not activate from the start of training. For the first 35% of
  wallclock (~3.5 minutes), the model trains as a standard 11-layer
  network. The loop then switches on and runs for the remainder. The reason
  is throughput: 17 effective layers is significantly slower per step than
  11. By training as a standard 11-layer network first, the model gets more
  gradient steps within the 10-minute budget before paying the cost of the
  loop.

- **Progressive context (PR #2014).** Rather than training at a fixed
  sequence length, the final model progressively lengthens context during
  training: starting at 1024 tokens, moving to 2048 for the bulk of
  training, and finishing at 3072. This gives the model long-context
  representations by the end without paying the throughput cost of 3k
  sequences from step one.

Taken together, the training loop is a carefully choreographed 10
minutes: fast 11-layer passes early, the recurrence loop switching on at the 35% mark, context growing longer as the clock runs down. Every
decision is timed to extract the most signal from a budget that ends the
moment the last second expires.

---

#### Quantization

Quantization and compression happen after the 10-minute training clock
stops, a separate post-processing step before the artifact is sealed.
Not everything gets quantized equally. The bulky matrix weights (
attention projections, MLP weights, and embeddings) dominate the artifact
size and get quantized aggressively (int6 or int7). The small scalar and
1D parameters like gains, skip weights, and mixing coefficients are kept
in full precision: they are too sensitive to round safely and too small to
matter for the size budget. The baseline rounded MLP weights to int6 using
simple nearest-neighbour rounding. The final model does something
considerably more sophisticated.

- **GPTQ (PR #535).** The core insight of GPTQ: when you round a weight,
  you introduce an error. Instead of ignoring that error, you can
  compensate for it by adjusting the remaining unquantized weights in the
  same layer. GPTQ uses the Hessian of the loss (second-order information
  about how sensitive the output is to each weight) to compute these
  compensating adjustments. The result is a quantized model that stays
  much closer to the original's predictions than naive rounding would
  achieve. This evolved to all weights including attention (PR #1285) and
  embeddings quantized at int7 (PR #1394 → PR #1586).

- **LQER (PR #1797).** After GPTQ, some quantization error remains.
  LQER stores a correction: compute the residual between the original and
  quantized weights, take a rank-4 low-rank approximation of it, and pack
  those correction factors into the artifact alongside the quantized
  weights. The model reconstructs a better approximation of the original
  weights at inference time. The correction costs ~30 KB of artifact space
  and recovers a meaningful fraction of the remaining quantization damage.

- **Further refinements.** AWQ-lite (PR #1908) identifies the most
  activation-sensitive weight columns and quantizes those at int8 rather
  than int6; Calib32 (PR #2135) doubles the calibration batches for a
  better Hessian estimate. The quantized weights also go through a
  compression pipeline, evolving from LZMA to Brotli-11 (PR #1179) and
  finally to a per-group lrzip+ZPAQ pipeline with L1 row reordering
  (PR #1855), saving ~280 KB over Brotli-only and fitting meaningfully more
  model into the 16 MB cap.

---

#### Post-Training / TTT

The 10-minute eval window is not just for scoring. The final model uses
most of it to actively adapt its weights to the validation text it is
about to score. This is test-time training (TTT): gradient descent at
eval time, within the same budget as scoring. The first legal TTT
appeared in PR #549; the structure that made it to the final SOTA is
considerably more sophisticated.

The TTT procedure has two nested levels of adaptation.

- **Per-document LoRA (PR #1530).** The base model weights are frozen. For
  each validation document, a set of low-rank adapter matrices (LoRA) are
  attached to the key, output, and MLP projections of every layer. The
  document is processed in small chunks: 8 tokens for short documents, 24
  for medium, 48 for long (PR #2014). For each chunk: score it first under
  the current LoRA, then take a gradient step on that chunk to update the
  LoRA weights. By the time the final chunk is scored, the LoRA has already
  adapted to the document's style, vocabulary, and content. After the
  document is done, the LoRA resets; nothing carries over to the next
  document.

- **Global SGD phase (PR #1610/#1626).** On top of the per-document LoRA,
  there is a global pause after an initial batch of documents have been
  scored (2000 in PR #1610, refined to 2500 in PR #1626). At that point, a
  full SGD pass runs on the base model weights themselves, not just the
  LoRA, using all the already-scored documents as training data. The base
  model is then updated, the LoRA resets, and the remaining documents are
  scored on top of this improved base. The LoRA handles fast local
  adaptation per document; the global SGD step shifts the base model toward
  the distribution of the validation set as a whole.

The eval budget splits roughly as ~120 seconds for a standard baseline
scoring pass and ~480 seconds for the TTT loop, the whole thing just
fitting within the 600-second cap.

Many other submissions attempted additional eval-time interventions beyond
TTT: n-gram tilts, retrieval augmentation, ensemble methods. Several were
ruled illegal. We will examine a few of these in the next section.

---

The final model is a messy, sophisticated combination of all of the above:
significant structural innovations layered on top of dozens of smaller
refinements, each contributing a fraction of the 0.17 BPB gap to the
baseline.

---

## Part 2: The Disqualifications

At various points in the competition, the open PR list showed submissions
claiming BPB in the 0.8–1.0 range, well below what the merged leaderboard
reflected. These numbers were real in the sense that the code produced
them, but they were not valid scores: the techniques behind them violated
one of the competition's core conditions. We discuss two such techniques
and one data leak here.

A pretrained model is static. It knows nothing about what has already
appeared in the document it is currently scoring. If a document mentions
"San Francisco" ten times, the model treats the eleventh occurrence the
same as the first. Several submissions tried to close this gap by running
lightweight online statistics alongside the model: track what has appeared
so far in this document, and use that to sharpen the model's predictions.
Two approaches in particular (n-gram tilting and PPM-D) attracted
significant attention, and both ultimately ran into legality problems.

### N-gram Tilt

An online n-gram tilt (PR #1145) maintains a prefix-keyed hash table of
token co-occurrence statistics, updated as each token is scored. At each
position, if the recent token history strongly predicts a specific next
token, that token's probability gets a boost before the loss is computed.
No extra artifact bytes, no parameters: just a running table built from
the document itself.

The implementation used three expert channels (token n-gram, within-word
continuation, and word-start). The token expert was clean. The within-word
and word-start experts had a classic C1 violation (PR #1420):

```c
const uint16_t tok = tokens[i];  // the target token being predicted
const uint8_t is_boundary = boundary_lut[tok];
if (!is_boundary && st->within_len > 0U)
    within_valid[i] = 1U;        // fire the hint at position i
```

The gate reads `tokens[i]` before scoring position i, the label, not the
prefix. In the shipped code, 100% of positions where the within-word
expert fired were continuation tokens; no causal system can achieve that.
The fix (PR #1514): disable within-word and word-start entirely, keep only
the token-order-16 expert. That clean expert survived into the final SOTA.

### PPM-D

PPM-D (Prediction by Partial Matching) is a classical lossless compression
algorithm. It maintains a trie of byte n-gram counts and at each position
predicts the next byte by looking up the longest matching context, falling
back to shorter contexts when the full history hasn't been seen before. It
was state-of-the-art for text compression before neural networks and is
particularly effective at within-document repetition: once "San Francisco"
has appeared several times, the byte sequence becomes highly predictable.

The appeal for parameter golf was direct: BPB is charged at the byte level,
and PPM-D operates at the byte level. A cluster of submissions (starting
with PR #1785) mixed PPM-D with the neural net by spreading each token's
probability uniformly across its bytes, an n-byte token with probability p
contributing p^(1/n) to each of its byte positions, then taking a convex
combination with PPM-D's byte predictions. Claimed scores dropped into the
0.8–1.0 range.

The problem, identified in Issue #1872, is that the spread does not produce
a valid probability distribution. For any multi-byte token with p < 1,
p^(1/n) > p: the per-byte contributions are inflated. Summing across all
tokens that start with a given byte gives more than 1.0. The mixture is not
normalized, violating C2.

PR #1905 ran the decisive experiment: using the same PPM configuration but
with a correct byte marginal (the proper way to convert token probabilities
to byte probabilities), PPM-D was *worse* than the baseline by 0.038 BPB.
The entire apparent gain came from the invalid spread inflating the NN's
apparent uncertainty on multi-byte tokens, then giving PPM spurious credit
for resolving it. The 0.8x BPB figures were an artifact of the scoring
construction, not a real compression improvement.

## Drama on the Last Day

On April 30th (the day before the competition closed), PR #2014 dropped
at **1.0576 BPB**, a clean record built on progressive context scheduling.
Then in the final hours, a flurry of PRs appeared beating it: **1.047**,
**1.043**, numbers that seemed implausibly good. The field looked wide open.

People started looking more closely.

It turned out that `prepare_caseops_data.py` (the script everyone had
been copying to build CaseOps datasets since PR #1736) defaulted to
`--val-docs=10000`. Training started at document 10,000; validation
covered documents 0–50,000. Eighty percent of the validation set had been
in training the whole time.

Fifteen PRs were eventually classified as leaky (Issue #2127), most of
their authors having inherited the data setup from earlier submissions
without knowing. The fix had actually been discovered and quietly deployed
eight days earlier, then re-introduced independently on the same day. The
best detail: the author who first caught and fixed the bug also submitted
PR #2118 (the most egregiously leaked PR of the competition) whose own
`submission.json` admitted the overlap in its technique summary field.

When the dust settled, one PR stood as the clean improvement over #2014:
**PR #2135** at **1.0565 BPB**, by a narrow margin of 0.001. The final
SOTA of the competition.

Six weeks, two thousand pull requests, a 0.168 BPB drop from a number
that already looked hard to beat. This competition had everything:
techniques stacking on each other in ways nobody planned, ingenious
innovations that shouldn't have worked but did, controversial methods that
looked like miracles, mayhem on the last day, and a picture-perfect finish.
One clean submission, one narrow margin, one number that stood.
