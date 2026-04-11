# Parameter Golf Baseline Architecture Analysis

This document explains the baseline `train_gpt.py` in plain English, then places the current leading 16MB record in context.

The audience is assumed to understand systems engineering, budgets, tradeoffs, and profiling, but not machine-learning jargon. Every specialized term is defined the first time it appears.

## Executive Summary

The baseline is a deliberately simple small language model trainer:

- It trains a 9-layer transformer, which is the now-standard neural network design used in GPT-style models.
- It uses a very small vocabulary of 1024 tokens, meaning its built-in word dictionary is tiny by modern LLM standards.
- It ties the input and output embedding table, meaning the same lookup table is reused both to read tokens in and predict tokens out. This is a major byte saver.
- It spends most of its parameter budget inside the per-layer MLPs, the feed-forward sub-networks inside each transformer block.
- It spends most of its compute budget on repeated matrix multiplies in the MLP and attention layers, not on data loading or logging.
- It fits under the 16MB limit mainly by exporting weights as int8, meaning each weight is rounded to an 8-bit integer, then compressing the result with `zlib`.

The baseline is useful because it isolates the real competition levers:

- Better byte allocation
- Better post-training compression
- Better use of the 10-minute training window
- Better use of the separate 10-minute evaluation window

The current best results did not come from one exotic idea. They came from repeatedly moving bytes out of low-value places and into high-value places, then reducing the quality loss from compression.

## A Few Terms Up Front

- Token: a small text unit from the tokenizer. It is not always a word. It can be part of a word, a whole word, punctuation, or a byte-like fragment.
- Tokenizer: the component that turns text into token IDs and back.
- Embedding: the learned numeric lookup table that turns each token ID into a vector the model can process.
- Transformer block: one repeated processing unit in the model, containing attention plus an MLP.
- Attention: the part that lets each token look at earlier tokens in the sequence.
- MLP: multi-layer perceptron; here it means the per-token feed-forward sub-network inside each block.
- Quantization: storing weights with fewer bits, such as int8 or int6, instead of high-precision floating point.
- QAT: quantization-aware training. The model is trained while simulating future rounding/compression damage.
- BPB: bits per byte. This competition’s score. Lower is better.
- RoPE: rotary position embedding, a way to tell the model where each token sits in the sequence.
- GQA: grouped-query attention. Multiple query heads share fewer key/value heads to save parameters and memory.
- SWA: stochastic weight averaging. Average several late checkpoints.
- EMA: exponential moving average. A rolling smoothed copy of the weights.
- GPTQ-lite: a post-training quantization trick that tests a few clipping settings and keeps the one that best reconstructs the original weights.

## 1. Baseline Architecture

### What the baseline is trying to do

The baseline trains the best small GPT-style language model it can in about 10 minutes on 8 H100 GPUs, then compresses the model plus the code into less than 16,000,000 bytes.

Its priorities are:

1. Stay understandable.
2. Stay reproducible.
3. Use straightforward engineering choices that new competitors can improve.

It is not designed to win. It is designed to be a stable launch platform.

### The full pipeline at a glance

The baseline has five major stages:

1. Load hyperparameters from environment variables.
2. Load tokenized training and validation shards from disk.
3. Train a 9-layer GPT-style transformer with distributed data parallelism across GPUs.
4. Quantize the trained weights to int8 and compress them with `zlib`.
5. Decompress the artifact, load it back, and re-run validation so the printed score reflects the actual submission bytes, not the uncompressed model.

That last step matters. In this competition, the compressed round-trip score is what counts.

### Hyperparameters: the global control panel

The `Hyperparameters` class is just a namespace full of defaults read from environment variables.

This includes:

- Data paths
- Random seed
- Validation cadence
- Training length and wall-clock cap
- Model shape
- Learning rates
- Optimizer settings

Why it matters:

- It makes the script easy to sweep without editing code.
- It keeps the baseline generic enough to run on different machines and datasets.
- It lets competitors search the space quickly.

Effect on the 16MB budget:

- Indirect only. Hyperparameters do not consume much artifact space.
- They strongly affect whether the learned weights compress well.

### Muon optimizer: why the baseline does not just use Adam everywhere

An optimizer is the rule that updates weights after each batch.

The baseline uses three optimizer buckets:

- Adam for token embeddings
- Muon for large matrix weights inside transformer blocks
- Adam for small scalar and vector control parameters

Muon is unusual. It takes a gradient update for each 2D matrix, then orthogonalizes that update with a Newton-Schulz iteration. In plain English: it reshapes updates so they behave more like clean rotations and less like badly conditioned stretching.

Why it matters:

- The competition budget is short. Fast convergence matters more than asymptotic elegance.
- Muon often improves early training efficiency on matrix-heavy transformer weights.
- The script uses separate optimizers because embeddings and tiny control tensors behave differently from large block matrices.

Effect on the 16MB budget:

- No direct byte cost in the final artifact, because optimizer state is not shipped.
- Strong indirect effect: the optimizer changes the weight distribution. Cleaner, smaller, less spiky weights survive quantization better and compress better.

### Tokenizer-aware but tokenizer-agnostic scoring

The competition does not score plain loss alone. It scores compression quality in bits per byte.

Because different tokenizers can split text differently, the baseline builds lookup tables from the SentencePiece tokenizer:

- Base byte count for each token
- Whether the token starts with a leading space marker
- Whether the token is a special boundary/control token

Then validation converts token-level loss into bytes-based scoring.

Why it matters:

- It prevents a tokenizer change from getting a free score boost just because it changed token count.
- It aligns model quality with real compression of the original text stream.

Effect on the 16MB budget:

- None in the submission artifact.
- It affects what kinds of tokenizer changes are actually worth trying.

### Validation path

Validation loads the entire `fineweb_val_*` split, slices it into fixed-length sequences, runs the model, and computes:

- `val_loss`: normal token prediction error
- `val_bpb`: the competition metric

Important engineering detail:

- The training wall-clock cap in this script excludes validation time. Training time is paused during eval and resumed afterward.

Why it matters:

- The script can spend some time validating without eating into the 10-minute training budget.
- Separate record submissions later exploit the separate evaluation budget much more aggressively with sliding-window scoring.

Effect on the 16MB budget:

- None directly.
- But evaluation strategy changes can improve the reported score without changing model bytes at all.

### Post-training quantization: how the baseline gets under 16MB

The trained model lives in bf16 and fp32 during training. That is far too large to submit.

The baseline export path does this:

1. Small float tensors are kept in float, usually downcast to fp16 if possible.
2. Large 2D tensors are quantized per row to int8.
3. Non-floating tensors are kept exactly.
4. The resulting object is serialized with `torch.save`.
5. That serialized blob is compressed with `zlib` level 9.

Per-row quantization means each row gets its own scale factor instead of using one scale for the whole matrix. This usually preserves quality better because different output channels have different numeric ranges.

The baseline also clips outlier values before quantizing, using a very high percentile rather than the absolute max. That avoids wasting range on a few extreme values.

Why it matters:

- This export path is the whole reason the baseline can submit at all.
- The competition is as much a weight-compression problem as a model-design problem.

Effect on the 16MB budget:

- Massive. This is the central byte-saving mechanism.
- The baseline artifact was reported at about 15.82MB for weights and about 47.6KB for code, for a total of about 15.86MB.

### Data loading: simple streaming instead of fancy sampling

Training data lives in binary shard files. The loader:

- Verifies a simple shard header
- Streams shards sequentially
- Wraps around forever
- Splits one contiguous chunk across ranks so each GPU gets a disjoint span

Why it matters:

- Minimal Python overhead
- Minimal moving parts
- Deterministic behavior

Effect on the 16MB budget:

- None directly.
- Good throughput matters because more steps in 10 minutes means better model quality.

### RMSNorm

Normalization layers keep activations numerically stable. The baseline uses RMSNorm, which normalizes by root-mean-square magnitude.

Why it matters:

- It is simpler and lighter than LayerNorm.
- It tends to work well in transformer stacks.

Effect on the 16MB budget:

- Almost free. In this baseline, RMSNorm has essentially no learned weight payload because it uses the functional form directly.

### CastedLinear

`CastedLinear` keeps large weight matrices in fp32 for optimizer quality but casts them to the current compute dtype during the matrix multiply.

Why it matters:

- Training quality of fp32 master weights
- Speed of bf16 compute

Effect on the 16MB budget:

- None directly, since only final exported weights matter.
- Indirect quality benefit because cleaner training produces weights that quantize better.

### Rotary position encoding

The model needs to know token order. The baseline uses RoPE, which rotates query and key vectors by position-dependent angles.

Why it matters:

- It is parameter-free.
- It generalizes reasonably well to sequence handling.
- It is now a standard low-overhead positional encoding.

Effect on the 16MB budget:

- Zero learned parameters.
- Good positional handling without spending bytes.

### Attention block

Each transformer block contains causal self-attention.

The baseline attention has:

- 8 query heads
- 4 key/value heads
- Head dimension 64
- Query, key, value, and output projection matrices
- Per-head learned query gain
- Causal masking so tokens only see the past
- GQA so keys and values are shared across pairs of query heads

Why it matters:

- Attention is the mechanism that lets the model use context.
- GQA reduces parameter count and memory traffic without throwing away the transformer structure.

Effect on the 16MB budget:

- Attention is a large fraction of the model, but not the largest.
- In the baseline, all attention matrices across all 9 layers add up to about 7.08 million parameters, around 41% of total parameters.

### MLP block

After attention, each block runs an MLP. Here it is a two-layer feed-forward network with ReLU squared activation:

- Expand from 512 to 1024
- Apply ReLU
- Square it
- Project back to 512

Why it matters:

- MLPs are where much of the model’s memorization capacity lives.
- In small transformers, widening the MLP is often one of the highest-return places to spend extra bytes.

Effect on the 16MB budget:

- This is the single largest byte consumer.
- In the baseline, MLP weights across all 9 layers total about 9.44 million parameters, around 55% of total parameters.

That number explains a large part of the later leaderboard evolution: most successful submissions learned how to compress MLP weights harder than attention weights.

### Residual and control parameters

Each block has a few small learned control vectors:

- `attn_scale`
- `mlp_scale`
- `resid_mix`

The full model also has `skip_weights` for U-Net-like skip reuse between encoder-style and decoder-style halves of the stack.

Why it matters:

- These are tiny but high leverage.
- They let the model tune how much each branch contributes.

Effect on the 16MB budget:

- Negligible byte cost.
- The baseline treats these as special control tensors and keeps them in float because their size is tiny and their sensitivity is relatively high.

### GPT model structure

The top-level `GPT` class is a 9-layer stack with:

- Token embedding table
- First half of layers acting like an encoder side that stores skip states
- Second half acting like a decoder side that reuses those skips in reverse order
- Final normalization
- Output logits computed either with the tied embedding table or a separate head

This is not a full encoder-decoder model. It is still a GPT-like causal model. The “encoder/decoder” language here just describes how skip connections are arranged.

Why it matters:

- The skip structure gives later layers a short path back to earlier representations.
- Tied embeddings save bytes twice: no separate output projection matrix, and the embedding becomes more valuable per byte because it serves two jobs.

Effect on the 16MB budget:

- Tied embeddings are a major budget win.
- The embedding table is only about 524,288 parameters, around 3% of the total, but it is unusually important because it is used both at input and output.

### Training loop

The training loop is optimized around the wall-clock limit:

- Distributed training across GPUs
- Gradient accumulation so the code works for fewer than 8 ranks while preserving the same global batch
- Warmup steps to trigger compilation and kernel specialization before timing begins
- Per-step learning-rate scaling with warmup and warmdown
- Periodic validation
- Early stop at the time cap

Two small but important systems choices:

- The warmup steps are thrown away and the model state is restored, so compilation overhead does not contaminate measured training.
- The stop logic synchronizes across ranks so one GPU cannot keep going after another hits the cap.

Why it matters:

- This is a race against wall-clock, not epochs.
- Throughput engineering directly becomes model quality.

Effect on the 16MB budget:

- None directly.
- Huge indirect effect because better scheduling and faster kernels determine how much useful learning happens before export.

## Baseline Byte Reality

The baseline’s parameter distribution is approximately:

| Component | Approx params | Share of params | Why it matters |
|---|---:|---:|---|
| MLP weights across 9 layers | 9,437,184 | 55.1% | Largest byte sink; best place to compress harder |
| Attention weights across 9 layers | 7,077,888 | 41.3% | Second-largest byte sink; more sensitive than MLP |
| Token embedding / tied output table | 524,288 | 3.1% | Small by count, high sensitivity |
| Skip weights + controls + q gains | tiny | under 1% | Cheap to keep high precision |

The baseline model has about 17.1 million parameters total.

A useful mental model:

- The MLP owns most of the bytes.
- Attention owns most of the context-useful compute.
- The embedding owns a disproportionate amount of score sensitivity relative to its size.

## 2. Current #1 Submission Analysis

Target file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

### What this submission is, in one sentence

It is an 11-layer, widened, heavily compressed transformer that keeps the strongest ideas from the prior leaderboard run, then improves the export path and late-stage training smoothness enough to squeeze out another 0.0013 BPB.

### What it inherited from the earlier record line

The current #1 is not a fresh architecture. It sits on top of a mature stack that had already accumulated:

- 11 layers instead of 9
- 3x MLP expansion instead of 2x
- Int6 mixed quantization for most block weights
- Int8 embeddings
- `zstd` instead of `zlib`
- U-Net skip connections
- Efficient Partial XSA on the last 4 layers
- Partial RoPE
- LN Scale
- Shared value embedding on the top layers
- SmearGate
- BigramHash
- Orthogonal initialization
- Muon + AdamW with weight decay
- Sliding-window evaluation

So the leading run is best understood as a mature systems stack plus final-mile compression refinements.

### The four named innovations in the README

#### 1. GPTQ-lite

Normal post-training quantization picks one clipping rule, usually based on the row max.

GPTQ-lite in this record tries five clipping percentiles per matrix row:

- 0.999
- 0.9995
- 0.9999
- 0.99999
- 1.0

Then it picks the one with the lowest reconstruction mean squared error, meaning the quantized row stays closest to the original row.

Why it matters:

- It improves quantization quality without retraining.
- It costs export-time compute, not training-time compute.
- That is exactly the kind of trade the competition rewards.

Practical meaning:

- The model learns once.
- The exporter becomes smarter about how to spend the fixed int6 grid.

The README attributes about `-0.0006 BPB` to this alone.

#### 2. EMA weight averaging

EMA keeps a continuously updated smoothed copy of the weights:

`ema = 0.997 * ema + 0.003 * current`

Why it matters:

- It smooths out noisy late-training fluctuations.
- Smoother weights often generalize better.
- Smoother weights also quantize and compress better because they tend to have fewer sharp outliers.

This is a systems-engineering move as much as an ML move: better final weight geometry makes the byte-constrained export less destructive.

The README attributes about `-0.0006 BPB` to EMA.

#### 3. Longer warmdown: 3500 instead of 3000

Warmdown is the period where the learning rate decreases toward zero.

Why it matters:

- In a short-run training regime, late steps are expensive and precious.
- Extending warmdown gives the model more low-learning-rate cleanup time.
- That tends to improve both final quality and quantization robustness.

This is a small gain, about `-0.0002 BPB`, but consistent with the broader pattern seen in earlier records: compression-friendly weights often come from gentler late training.

#### 4. Earlier late-QAT threshold: 0.15 instead of 0.1

Late QAT means fake quantization only turns on once the learning rate has already dropped.

Raising the threshold from 0.1 to 0.15 activates fake quantization earlier in the run.

Why it matters:

- The model gets more time to adapt to future rounding damage.
- The quantization gap gets a little smaller.

The README attributes about `-0.0001 BPB` to this change.

### The real story of the current #1

The leading submission’s real innovation is not “invent a brand-new network.” It is:

- Build a deeper and wider model than the baseline.
- Compress MLPs harder than attention.
- Spend evaluation budget more intelligently.
- Make late-stage weights smoother.
- Make export smarter row by row.

That is why the technique stack compounds.

### Why these changes make sense under a 16MB hard cap

The #1 submission is still under 16MB because it follows a clear byte policy:

- Keep high-value, quantization-sensitive components gentler.
- Crush the cheap, repetitive matrices harder.
- Use a stronger compressor than the baseline.
- Treat quantization as part of the architecture, not just the last line of the script.

In plain English: it stops pretending every weight deserves the same precision.

## 3. Byte Budget Breakdown

### Baseline artifact numbers

The naive baseline README reports:

- Compressed model bytes: 15,815,847
- Code bytes: 47,642
- Total submission bytes: 15,863,489

That means the baseline is already close to the ceiling. There is very little unused slack.

### Where the bytes really are

#### 1. MLP weights are the main budget

The baseline MLP stack is about 55% of all parameters.

This is why later records aggressively moved MLP weights to int6 and then int5:

- MLP weights are numerous.
- They are comparatively compressible.
- The bytes freed there can buy another layer, a wider MLP, or auxiliary features like BigramHash.

If you remember one byte-budget fact, remember this:

The MLP is the bank account.

#### 2. Attention weights are the second biggest budget line

Attention matrices are about 41% of the baseline parameters.

They are usually more sensitive than MLPs, especially in later layers and especially for key/value projections that control context retrieval.

That is why strong submissions often use:

- Int6 for attention
- Int5 only for MLP
- FP16 exceptions for a few especially sensitive tensors

#### 3. The embedding table is small but fragile

The tied embedding is only about 3% of baseline parameters, but it is unusually important:

- It is the input dictionary.
- It is also the output prediction table.

Quantization damage hits both roles at once.

This is why one of the first successful improvements was keeping the tied embedding in fp16 instead of int8. It cost around 500KB, which is large in this competition, but it removed a much larger quality loss than its parameter share would suggest.

#### 4. Control tensors are almost free

The various scales, mixes, gains, and skip weights are tiny.

They should generally be kept high precision unless proven otherwise. Compressing them harder saves almost nothing and can easily damage quality.

### What the baseline export is doing inefficiently

The baseline already uses sensible per-row int8, but compared with later records it leaves a lot on the table:

- It stores all quantized values in full int8.
- It uses `zlib`, not `zstd`.
- It does not differentiate MLP from attention.
- It does not use learned or searched clipping beyond a single percentile rule.
- It does not use QAT, so the model is not trained to survive the export.

### Where there is still room to compress further

Based on the record history, the biggest remaining compression opportunities are:

#### 1. Bit-packing below 8 bits

Most record submissions still store int5 and int6 values inside int8 containers and rely on compression to exploit unused high bits.

That works, but it is not the theoretical limit.

Directly packing 5-bit and 6-bit values would save raw bytes before compression. The tradeoff is code complexity and possibly worse secondary compression if packing destroys repetitive structure.

This is still a meaningful unexplored axis in the record writeups.

#### 2. Better per-component precision maps

The leaderboard already learned:

- MLP can go lower precision than attention.
- Embeddings need gentler handling.
- Some late-layer tensors deserve exceptions.

There is probably still room for more granular sensitivity maps:

- Per-layer precision
- Per-submodule precision
- Different rules for query, key, value, output, and MLP branches

#### 3. Better post-training quantization search

GPTQ-lite is a simple row-wise clip search, not the end state.

There is room for:

- More candidate clip rules
- Better error metrics than plain row MSE
- Group-wise or block-wise scale search
- Quantization rules that explicitly optimize compressed size, not just reconstruction error

#### 4. Codebook or product quantization

None of the record READMEs claim a true codebook-based weight representation, where small learned dictionaries replace raw weights.

That is still a major unclaimed space.

#### 5. Structured pruning plus byte reallocation

One top record used 3% magnitude pruning, but this is still lightly explored.

There may be room to:

- Remove predictable low-value neurons or heads
- Spend those bytes on extra depth, wider MLPs, or higher-quality embeddings

## 4. Compute Budget

There are really two compute budgets:

- Training budget: 10 minutes
- Evaluation budget: 10 minutes

The best submissions use both aggressively.

### What takes the most time during training

#### 1. MLP matrix multiplies

The MLP is the largest parameter block and a major compute sink. Every layer expands from model width to hidden width and then projects back.

When records widened MLP from 2x to 3x, they improved model quality but also increased per-step cost.

#### 2. Attention, especially at long sequence lengths

Attention cost grows roughly with sequence length squared. Moving from 1024 to 2048 or 4096 context dramatically increases work per step.

The records show the real trade:

- Longer context can improve quality.
- But if it cuts total step count too much, it can lose overall.

This is why later winners often settled on 2048, not 4096, and invested the saved bytes elsewhere.

#### 3. Quantization-aware training overhead

QAT adds extra fake-quantization work during forward passes.

It often paid off when it dramatically reduced quantization loss, but it is not free. One record explicitly reports meaningful step-time overhead from QAT.

The winning pattern became “use QAT late or selectively,” not “quantize-aware everything from step one at any cost.”

#### 4. Weight averaging overhead

SWA and EMA both add small overhead:

- SWA costs extra checkpoint storage and averaging
- EMA costs extra copy/update work every step

But compared with the cost of another full layer or longer context, they are cheap, which is why EMA became attractive late in the record line.

### What takes the most time during evaluation

#### 1. Sliding window evaluation

This is the biggest evaluation-time cost increase in the record history.

Instead of scoring disjoint chunks, sliding window re-runs overlapping windows and only scores the newest tokens in each window. It is much more expensive, but it gives almost every scored token near-maximum context.

Why it kept winning:

- Zero artifact cost
- Large BPB gain
- Fits inside the separate 10-minute evaluation budget

This is probably the single highest-return “free” idea discovered in the repo history.

#### 2. Test-time training

LoRA TTT was explored very early. It uses a tiny low-rank adapter trained on each validation document as evaluation proceeds.

It worked, but most of the gain in that early record came from:

- Document isolation
- Strided/sliding evaluation

TTT itself added only a small increment. It also burns evaluation budget.

That likely explains why the main record line did not keep building on it.

### Where compute time could still be saved

#### 1. Reduce attention cost without giving up context quality

The record line added more architecture and compression tricks, but there is still limited exploration of ways to keep long effective context with lower attention cost.

#### 2. Make quantization smarter at export time, not during training

GPTQ-lite is a good example: spend more exporter compute to avoid spending more training compute.

This is a strong direction because the competition rewards final score, not training elegance.

#### 3. Be selective with expensive features

The best records already moved toward this:

- XSA only on the deepest layers
- Late QAT instead of full-run QAT
- Mixed precision by component

That pattern likely still has room.

## 5. Opportunity Map

This section summarizes the full record evolution in `records/track_10min_16mb/`.

### The progression, in plain English

#### Phase 1: Easy wins on schedule and evaluation

Records from March 17-19 established that the baseline was leaving obvious gains on the table:

- Lower learning rates helped.
- Longer or better-shaped warmdown helped.
- Sliding window evaluation was a huge free gain.
- Keeping the tied embedding in fp16 prevented disproportionate export damage.
- Longer context helped when not overdone.

This phase taught the field that score was being lost in the export and eval path, not just in model quality.

#### Phase 2: Spend freed bytes on more model

Once compression got better, submissions started buying:

- 10 layers instead of 9
- 11 layers instead of 10
- MLP 3x instead of 2x

This was the decisive shift. Better compression was not the goal by itself. It was a funding source for a better architecture.

#### Phase 3: Add lightweight structural features

The next strong ideas were small, targeted add-ons:

- SmearGate
- BigramHash
- U-Net skip refinements
- Orthogonal initialization
- Weight decay tuned for quantization robustness

These worked because they added useful inductive bias without blowing the byte budget.

#### Phase 4: Smooth the final weights and refine export

Late records focused on polish:

- SWA
- EMA
- Partial RoPE
- LN Scale
- XSA only on deep layers
- GPTQ-lite
- Better late-stage warmdown and QAT timing

By then the architecture was already strong. The remaining gains came from making the final compressed artifact less lossy.

### Record-by-record technique inventory

| Record | Main idea |
|---|---|
| 2026-03-17 NaiveBaseline | Starting point |
| 2026-03-17 LoRA_TTT | Document-isolated eval, LoRA test-time training |
| 2026-03-18 LowerLR | Lower learning rates |
| 2026-03-18 FP16Embed_WD3600 | FP16 embedding export, longer warmdown |
| 2026-03-18 LongContextSeq2048 | Longer context training |
| 2026-03-19 TrainingOptSeq4096 | Even longer context plus better optimizer schedule |
| 2026-03-19 WarmdownQuantization | Train specifically for compressibility |
| 2026-03-19 SlidingWindowEval | Spend eval budget for a large score gain |
| 2026-03-19 10L MixedPrecision | 10 layers funded by mixed int8/int6 export |
| 2026-03-19 MixedQuant Int6Int8 SlidingWindow | QAT-style robustness plus sliding eval |
| 2026-03-19 Seq2048 FP16Emb TunedLR | 10 layers, int6 QAT, zstd, wider MLP |
| 2026-03-19 SlidingWindow FP16Emb 10L MuonWD OvertoneInit | Weight decay, 10 layers, better init |
| 2026-03-19 smeargate_orthoinit_muonwd | SmearGate, BigramHash, OrthoInit, MLP 3x |
| 2026-03-19 MLP3x_QAT_Int6_SlidingWindow | 11 layers, 3x MLP, weight decay, full stack maturation |
| 2026-03-20 Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA | Adds SWA on top of the stack |
| 2026-03-20 10L Int5MLP MuonWD04 SWA50 | First major int5 MLP win; bigram bucket count increase |
| 2026-03-20 11L EfficientPartialXSA FA3 SWA120 | Efficient partial XSA on deep layers |
| 2026-03-20 11L XSA4 EMA Int6 MLP3x WD04 | EMA replaces SWA, XSA extended to last 4 layers |
| 2026-03-21 11L XSA4 EMA PartialRoPE LateQAT | Partial RoPE and LN scaling |
| 2026-03-22 11L EMA GPTQ-lite warmdown3500 QAT015 | Smarter export and late-stage polish |

### Techniques clearly tried already

These have already been explored in the records and are not “white space” anymore:

- Lower learning rates
- Longer warmdown
- FP16 embedding passthrough
- Sequence-length tuning at 1024, 1408, 2048, and 4096
- Sliding window evaluation
- Mixed int8/int6 export
- Int5 for MLP
- `zstd` replacing `zlib`
- QAT and late-QAT
- 10-layer and 11-layer stacks
- MLP widening
- SmearGate
- BigramHash
- Orthogonal initialization
- Weight decay for quantization robustness
- SWA
- EMA
- XSA on top layers
- Partial RoPE
- LN scaling
- Light pruning
- LoRA TTT

### Techniques that appear tried and likely weak or at least non-dominant

Based on the READMEs:

- SwiGLU: better per step, too slow overall
- Depth recurrence: promising in theory, but too slow or too undertrained in current attempts
- LZMA: worse than `zlib`
- Overly high embedding LR: harmful
- Full-sequence 4096 context: useful but expensive; not the eventual winning path
- Early LoRA TTT as a main strategy: did not become part of the record line

### Techniques I did not see claimed in the record READMEs

These are the biggest remaining gaps visible from the record history:

#### 1. True sub-byte packing

The records use int5/int6 conceptually, but the stored tensors are still generally int8 containers plus compression. I did not see a README claiming true packed 5-bit or 6-bit storage.

That is important because the competition is byte-constrained first.

#### 2. Codebook-based weight compression

I did not see product quantization, vector quantization, or small learned codebooks used for weights.

This is one of the most obvious remaining compression families.

#### 3. Structured sparsity as a primary byte strategy

Light magnitude pruning appears, but not a full structured-sparsity program where removed heads or neurons directly fund added depth or wider high-value layers.

#### 4. Parameter sharing done seriously in the modern stack

Some early notes mention recurrence or looping, but I did not see a mature record combining:

- modern mixed precision
- modern eval tricks
- modern byte-aware architecture
- cross-layer sharing or recurrence

That might deserve another look now that the rest of the stack is stronger.

#### 5. Revisit test-time adaptation on top of the modern stack

LoRA TTT was tested early on a much weaker model. I did not see it revisited on top of:

- sliding-window eval
- mixed int5/int6 quantization
- EMA/SWA
- XSA/Partial RoPE stack

It may still be too expensive, but it has not been ruled out in the current stronger regime.

#### 6. Tokenizer redesign as a primary lever

The framework supports custom tokenizers, but the record line mostly treated the 1024-token SentencePiece tokenizer as fixed.

That is understandable because tokenizer submissions get more scrutiny, but it remains a meaningful design space.

#### 7. Compression-aware objective functions

The records increasingly train for quantization robustness, but I did not see a README claiming they explicitly optimize for final compressed byte count or compressed-roundtrip score during training.

That could be powerful if done cheaply.

### Biggest remaining gains, ranked

If the goal is “highest upside that still matches what the repo history suggests,” the most interesting directions look like this:

1. Better weight representation than int8 containers for int5/int6 values.
2. Codebook-style or blockwise quantization for MLP-heavy sections.
3. Finer-grained sensitivity mapping so each submodule gets the cheapest precision it can survive.
4. Revisit recurrence or sharing now that export and eval tricks are much better.
5. Revisit test-time adaptation on top of the modern stack, but only if eval budget still fits.

## Practical Takeaways

If you want the baseline to become competitive, the clearest lessons from the full record line are:

1. Do not spend equal precision on every tensor.
2. Treat MLP bytes as the main funding source.
3. Protect the tied embedding even if it looks small on paper.
4. Spend evaluation budget aggressively when it buys score without artifact bytes.
5. Optimize the exporter almost as hard as the model.
6. Smooth late-stage weights because smooth weights compress better.

## Bottom Line

The baseline is a clean, sensible, well-engineered small GPT trainer. But under Parameter Golf rules, it is leaving performance on the table in three places:

- It is too uniform in how it compresses weights.
- It is too conservative in how it spends the separate evaluation budget.
- It does not yet treat compression as a first-class architectural constraint.

The record line shows the winning pattern clearly:

- Compress MLPs harder
- Reinvest bytes into depth and width
- Add a few cheap structural features
- Use evaluation context more intelligently
- Polish the final weight geometry and export path

That is the foundation this repo evolved toward, and it is the right foundation for further work.
