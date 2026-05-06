# Asymmetric Two-Lane Parallel Routing + Tap-In V6 `cross_w=0.12` + MUON_WD=0.12 + Trimmed GPTQ + Wider Loop + Per-Pass Embeddings + Muon 0.98 + Legal TTT

## Results

3-seed mean **+V6+TTT**: **1.073938** sliding-window BPB. All seeds comfortably under 16 MB.

| Seed | Steps | Pre-quant BPB | Quant BPB | Raw SW BPB | + V6 + TTT | Artifact bytes |
|---|---|---|---|---|---|---|
| 2025 | 4955 | 1.082623 | 1.093401 | 1.076417 | **1.073313** | 15,944,620 |
| 1234 | 4960 | 1.083167 | 1.094769 | 1.077604 | **1.073801** | **15,942,701** |
| 42   | 4951 | 1.083938 | 1.094446 | 1.077692 | **1.074701** | 15,943,555 |
| **Mean** | 4955 | 1.083243 | 1.094205 | 1.077238 | **1.073938** | 15,943,625 |

Against the current leaderboard #1 (`1.0810`, PR #1493, bigbag — *SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT*): **−0.007062 nats cleared**, comfortably above the challenge's 0.005-nat SOTA improvement bar.

### Budget (recommended seed 2025)

| Component | Bytes |
|---|---|
| Produced model (quantized + brotli, `.int6.ptz`) | 15,864,198 |
| Code (`train_gpt.py` source, counted at runtime via `len(Path(__file__).read_text().encode("utf-8"))`) | 80,422 |
| **Total submission size** | **15,944,620** |
| **Headroom under 16 MB** | **+55,380** |

*(The committed `train_gpt.py` in this folder is a 33,409-byte LZMA stub that carries the 80,422-byte source code as compressed data. The stub extracts its payload to a temp directory at run time and `Path(__file__)` then points to the extracted, uncompressed source, which is what `code_bytes` counts against the 16 MB budget in the training log.)*

## Key Techniques

### Asymmetric lane initialization

The two-lane routing matrix is initialized asymmetrically instead of as all-ones. Attention output starts strongly routed to lane 0 ($1.3$) and weakly to lane 1 ($0.7$); MLP output starts strongly routed to lane 1 ($1.3$) and weakly to lane 0 ($0.7$). Zero additional parameters — this is a pure initialization change to the existing $11 \times 2 \times 2$ `parallel_post_lambdas` tensor.

<details>
<summary><i>Why symmetry breaking helps</i></summary>

With all-ones init, both lanes are identical functions of the input at step 0. The optimization landscape has a continuous symmetry between them: any solution is equivalent to a permuted solution where lanes are swapped. Training has no preferred direction in this symmetry group and the two lanes remain near-identical throughout, collapsing the effective dimension of the routing matrix.

Asymmetric init breaks the symmetry at step 0 and the two lanes specialize: lane 0 becomes the "attn-heavy" path, lane 1 the "mlp-heavy" path. The learned routing matrix at the end of training still prefers the asymmetric pattern (we verified by inspecting `parallel_post_lambdas` in the trained checkpoint), confirming the optimization does land in a non-symmetric solution that the symmetric init was failing to reach in 10 minutes.

Given the null cost (no new parameters, no new code paths, no extra training time), this is free.

The $1.3 / 0.7$ split was chosen to keep the initial lane outputs at roughly the same magnitude as the all-ones baseline while forcing specialization. We did not sweep this — a tighter split ($1.2 / 0.8$ or $1.1 / 0.9$) may work equally well; we haven't tested it. The asymmetric init is conceptually orthogonal to the routing mechanism itself and stacks cleanly.

</details>

### Two-lane parallel residual routing

The last three decoder layers (8, 9, 10) maintain two parallel residual streams instead of one. At each layer, attention reads from lane 0, MLP reads from lane 1, and each writes back to both lanes via a learned $2 \times 2$ routing matrix. The final output reads lane 1. Total cost: 66 new scalar parameters ($11 \times 2 \times 2$ post-lambdas + $11 \times 2$ resid-lambdas). This is the single largest mechanism in this submission.

<details>
<summary><i>How we got here: controls-only TTT, a theory that predicted the wrong thing, and the gradient-flow reading that survived</i></summary>

#### Controls-only TTT already gets 70%

Before any architecture change, we ran a diagnostic. Unfreeze only the 9,700 control parameters (`skip_weights`, `resid_mix`, `attn_scale`, `mlp_scale` — scalar tensors that scale and mix things but compute no new features) and run the same TTT loop.

| TTT scope | Params trained | Δ from raw SW |
|---|---|---|
| Full model | 35.9 M | −0.00444 BPB |
| Controls only | 9.7 K | **−0.00310 BPB** |

Seventy percent of TTT's gain comes from 0.03% of the parameters. Most of what TTT does at test time is re-weight the flow of existing features, not learn new ones.

#### A theory that fit until it didn't

From that reading we wrote down a coupling-flexibility theory. If TTT is mostly routing adjustments on a single residual stream, then a more-trained model has layers more tightly co-adapted, so a TTT perturbation at one layer's routing breaks downstream layers that were expecting the old distribution. On that reading, raw quality and TTT extraction are opposed on a single-stream architecture.

This fit every negative result on our books. Earlier loop activation (more shared training) improved raw BPB but cut TTT extraction 3×. Extra trained scalar gates on every block improved raw BPB and cut V6+TTT. Lower weight decay helped fp32 fit and blew up the quantization gap. All consistent with the theory.

The theory predicted the fix. Two parallel lanes with a learned routing matrix would let TTT move mass between them without breaking either, because routing between lanes is zero-sum and total compute is preserved. We implemented it expecting more TTT surface.

#### What we actually got

|  | Baseline | Two-lane | Δ |
|---|---|---|---|
| Pre-quant BPB | 1.08784 | 1.08476 | **−0.00308** |
| Raw SW BPB | 1.08296 | 1.07900 | **−0.00396** |
| V6+TTT BPB | 1.07738 | 1.07606 | −0.00132 |
| TTT extraction alone | −0.00558 | −0.00294 | +0.00264 (less) |

Raw SW moved by 0.004, a very large number on a stack where deltas live around 0.0001. TTT extraction almost halved. The win was real; the prediction was wrong about where it came from.

#### Why the routing cannot be the thing

A single parallel block, dropping norms for clarity, is

$$h_0' = \gamma_0 \, h_0 + \alpha_0 \, y_a + \beta_0 \, y_m$$

$$h_1' = \gamma_1 \, h_1 + \alpha_1 \, y_a + \beta_1 \, y_m$$

where $h_0, h_1$ are the two lanes, $y_a, y_m$ are the attention and MLP sublayer outputs, and the final output reads $h_1$. Take the loss gradient with respect to the two attention-routing scalars:

$$\frac{\partial \mathcal{L}}{\partial \alpha_0} = \frac{\partial \mathcal{L}}{\partial h_0'} \cdot y_a \qquad \frac{\partial \mathcal{L}}{\partial \alpha_1} = \frac{\partial \mathcal{L}}{\partial h_1'} \cdot y_a$$

At initialization both lanes equal the layer input, the routing is symmetric ($\alpha_0 = \alpha_1 = 1$), and downstream treats the two lanes the same way. The two partials are equal. SGD updates $\alpha_0$ and $\alpha_1$ by the same amount, same sign, same magnitude. The difference $\alpha_0 - \alpha_1$ has no driving term.

This is a $\mathbb{Z}_2$ gauge symmetry in the parametrization. The shift $(\alpha_0, \alpha_1) \to (\alpha_0 + \varepsilon, \alpha_1 - \varepsilon)$ is a null direction of the loss at initialization, and the gradient descent trajectory stays on the symmetric submanifold. Roughly half of the 66 routing scalars are gauge. TTT cannot adapt along a coordinate the gradient does not see, so the routing-as-TTT-surface reading was empty.

This is the standard failure mode of parallel ensembles without explicit symmetry breaking. We had assumed the architecture provided it; the math says it does not.

#### The reading that survived the math

If the routing scalars are half gauge, the win has to come from something that would also show up with the routing frozen at initialization. So look at the gradient path, not the parameter count.

In a single-stream transformer, the gradient from the loss to $y_a$ at layer $\ell$ walks backward through every later layer:

$$\frac{\partial \mathcal{L}}{\partial y_a^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial o} \prod_{\ell' > \ell} J_{\ell'}$$

Each Jacobian attenuates and rotates. Layer 8's attention gets a signal already shaped by layers 9 and 10, the final norm, the head, and the softcap. Layer 8 ends up learning features conditional on what 9 and 10 happen to be doing.

In the two-lane architecture every sublayer writes directly into $h_1$ through its routing coefficient, so the gradient has an order-1 term independent of depth:

$$\frac{\partial \mathcal{L}}{\partial y_a^{(\ell)}} \;=\; \underbrace{\frac{\partial \mathcal{L}}{\partial o} \cdot \alpha_1^{(\ell)}}_{\text{direct}} \;+\; \underbrace{\text{cascading terms}}_{\text{order} \geq 2}$$

The direct term is a single scalar multiply away from the loss, regardless of whether the block is at layer 8, 9, or 10. It gets there on the first backprop step and keeps getting there. This is DenseNet connectivity applied to the last few decoder layers of a language model.

#### Why the effect is so large at 10 minutes

The DenseNet shortcut's benefit at convergence is modest. At 588 seconds on 8×H100 we run about 4,873 SGD steps, which is severely undertrained by any classical standard, and the decoder — deepest in the gradient chain, furthest from the loss — is the region whose weights are least converged when wallclock hits. A 24-hour run would narrow the gap between the two architectures. Our run does not.

That is why almost the entire improvement lives in pre-quant BPB ($-0.00308$ of the $-0.00396$ raw SW delta). It is a training-time effect. The post-GPTQ weights are better because the pre-GPTQ weights are better, because each decoder block saw cleaner gradients per step.

#### One line

Every sublayer between a weight and the loss is a tax on that weight's training signal. At 10 minutes we cannot afford taxes we don't have to pay. Parallel routing is the cheapest exemption we have found: 66 scalars, no new matrices, no kernel issues, and an order-1 gradient path from every late-decoder sublayer to the loss.

The routing-as-adaptation reading is the one we started with. The routing-as-gradient-shortcut reading is the one that survives the math and the numbers. We were looking for adaptability and we found training efficiency hiding inside the same architecture.

</details>

### Wider depth recurrence + per-pass loop embeddings

`LOOP_START=3`, `LOOP_END=5`, `NUM_LOOPS=2` — three passes through three distinct loop blocks instead of four passes through two. 9 loop block executions, 17 virtual layers total. Three zero-init learned vectors (`nn.Embedding(3, 512)`), one added to the residual at the start of each pass.

<details>
<summary><i>Wider loop + per-pass embeddings — mechanistic analysis</i></summary>

Depth recurrence reuses block weights across passes, creating virtual depth without new parameters. The cost: quantization error amplifies through reuse by $A(k) = (1 - \rho^k) / (1 - \rho)$, which at our contraction ratio $\rho \approx 0.63$ is $1.96\times$ for three passes or $1.67\times$ for two.

**Wider loop.** Blocks $(3, 4, 5)$ looped three times instead of $(4, 5)$ looped four times. Same 17 virtual layers, three distinct parameter sets instead of two. Gives $-0.0007$ BPB at identical pre-quant — the improvement is entirely post-quantization.

**Per-pass embeddings.** Three zero-init learned vectors ($e_i \in \mathbb{R}^{512}$, 1,536 parameters total) added to the residual before each pass. Combined with the wider topology: $-0.00124$ BPB on a 5-seed mean at $p < 0.003$. On the narrow topology: only $-0.0005$. The mechanism is topology-dependent.

**Where the gain lives.** The embeddings barely improve fp32 modeling. Nearly all of the gain comes from a collapsed quantization gap ($0.0131 \to 0.0114$). The weights become more quantization-friendly, not more expressive.

We traced this through per-matrix statistics, then per-head decomposition, then direct intervention. The weight-distribution signature localizes to two attention heads (K head 2, V head 1) in the loop blocks. Injecting a bias directly at those heads recovers about half of the gain (the modeling part) but none of the compression part. The per-head signature is downstream of the mechanism, not its cause.

The embedding mechanism has two separable effects: a modeling effect (K specialization in the newly-added block 3, reproducible by a 192-parameter direct bias) and a compression effect (quant-gap collapse, not reproducible by any targeted head-level intervention we tested). The full residual-stream embedding constrains K from over-specializing and trades that headroom for compression-friendliness. Direct bias takes the unconstrained modeling win but misses the compression side. The mechanism requires the embedding to propagate through shared RMSNorm into both attention and MLP simultaneously; neither pathway alone reproduces it.

</details>

### Tap-In V6 with `TAPIN_V6_CROSS_W=0.12`

Cross-window n-gram + cross-window lost-length rule, applied at eval time by a C++ matcher (~135 s on 8×H100). `TAPIN_V6_CROSS_W` is the weight on the cross-window hint signal, set to `0.12` (double the upstream default of `0.06`) to push the Tap-In nudge harder.

<details>
<summary><i>Tap-In — what it is and why we call it that</i></summary>

*Why "Tap-In"?* In golf, the tap-in is the tiny final stroke that rolls the ball the last inch into the hole after the big drive has done all the work. The model does the big swing; Tap-In is the small eval-time nudge that finishes the putt.

*Intuitively:* Tap-In is a document-local scribe. As the model predicts each token, the scribe scans backward through the same document for the exact phrase the model just generated and whispers what came after it last time. If the model is already considering that token, the scribe nudges its probability up a tiny bit. If the phrase fell out of the model's 2048-token attention window (think: a proper name introduced 3000 tokens ago), the scribe is the only one who can recover it. Wrong whispers cost almost nothing because the nudge is small; right whispers — especially for forgotten long-range repetitions — cut several nats off the loss at that single position. It fires hundreds of thousands of times across the eval; each win is small but they stack into $-0.001$ BPB on top of the model.

</details>

### Legal Score-First TTT

`TTT_LR=0.005`, `TTT_FREEZE_BLOCKS=0`, stacked on top of V6 in the SCORE phase. All 35.9M parameters trainable. Score accumulated under `torch.no_grad()` before any optimizer step runs, so every scored token was predicted by a model that had not yet seen it.

### Muon momentum 0.98

Lower Nesterov momentum reduces the effective gradient memory from 100 steps to 50 steps, better matched to the short training run. A 4-point sweep (0.95, 0.97, 0.98, 0.99) identifies 0.98 as the sweet spot at $-0.00108$ BPB (3-seed mean) — about $-0.0006$ from better pre-quant convergence and $-0.0005$ from a reduced quantization gap.

### Muon weight decay 0.12 + `MATRIX_CLIP_SIGMAS=13.10`

Higher WD shrinks weight magnitudes so the quantization gap closes to $0.0095$. Pre-quant BPB is essentially unchanged from lower-WD alternatives; the gain comes from a tightened quant gap. The slightly higher `MATRIX_CLIP_SIGMAS=13.1` absorbs the byte overhead of the routing scalars while keeping all seeds under 16 MB.

### `HESSIAN_CLIP_LAMBDA=0`

An upstream code default of $0.175$ was a known-failed feature left in by an earlier PR. Pinning it to 0 gives $-0.0006$ BPB and about 40 KB smaller model.

### `GPTQ_RESERVE_SECONDS=4` + `GPTQ_CALIBRATION_BATCHES=16`

The training loop stops `GPTQ_RESERVE_SECONDS` before the 600s wall-clock cap so GPTQ Hessian collection has room to run. Upstream defaults are $12$ seconds of reserve and $64$ calibration batches, which collect Hessians in ~$13$ seconds. Research ("Hessians already converged well before 64 batches") suggests the calibration budget is over-provisioned; cutting it to 16 quarters the collection time to ~$3.5$ seconds with no meaningful quality loss. With the faster GPTQ, we can safely drop the reserve to 4 seconds, reclaiming ~$17$ seconds of wall clock (from the combined 12 → 4 reserve cut and the 13 → 3.5 collection cut) for warmdown training. All three seeds stay under 16 MB.

## What gets evaluated

The competition harness runs `torchrun --nproc_per_node=8 train_gpt.py`. This single file is the entire scored submission — it decompresses, trains, quantizes, and evaluates end-to-end.

The `human_readable/` directory contains the identical unminified source code for reviewer convenience; it is not used at runtime. The LZMA stub in `train_gpt.py` carries its own compressed copy of the same source files.

## Methodology — single pass, no double evaluation

**The headline number is from a single causal left-to-right pass through the val set** with Tap-In V6 + Legal TTT applied during scoring. There is no double pass, no second-pass rescoring, no information leak between runs.

| File | What it is | Passes through val |
|---|---|---|
| `train_seed{42,1234,2025}.log` | Full training run including Phase 1 (train + GPTQ + raw SW eval) — reports the **Raw SW BPB** column. | 1 |
| `eval_v6_ttt_s{42,1234,2025}.log` | Phase 2 from the same run: re-loads the saved `.int6.ptz` and runs **V6 + Legal TTT** — these are the headline numbers per seed. | 1 each |

Each eval is a fresh load of the same saved int6 model — no state carried between runs, no information leak from any earlier run into a later one. The leaderboard-scored number is the **+V6+TTT** column of the per-seed table, produced by a single pass.

## Legality

Every gain comes from the strict prefix. Score-first TTT accumulates `loss_sum` under `torch.no_grad()` before `optimizer.step()` runs, and the chunk math is airtight: chunk `c`'s training targets max out at global position `(c+1)*32768`, while chunk `c+1`'s scored targets start at `>= (c+1)*32768 + 1`. Strict inequality; no token is ever predicted by a model that has already been trained on it.

The Tap-In V6 C++ matcher is byte-identical to the previously-audited reference. Within-window matches require `p+1 < t` so `cont = ids[p+1]` is strict prefix. Cross-window's `lost_len_at_t` upper bound resolves to `(ws + t) - window_size + 1 < ws + t + 1`. The linked-list `head / tail / fwd[tok]` update happens after the score block, not before. There is no `is_bnd_[tok]` or `has_ls_[tok]` target-dependent gating anywhere in the matcher — the Category 15 attack surface is structurally absent, not disabled.

The probability mixing sums to 1 by construction. Eval is one left-to-right sliding pass with non-overlapping 64-token scored ranges, so no position is ever rescored. GPTQ Hessians are collected from `train_loader.next_batch()` with zero val-data exposure during training. The model is deserialized from `.int6.ptz` before TTT touches anything, so this is eval-time adaptation, not pre-quant TTT.

The two-lane parallel routing is training-time architecture only. Its 66 scalars are trained on the same `train_loader` as everything else, with no eval-time tuning.

<details>
<summary><b>What did not work</b></summary>

Every row is a controlled single-seed experiment against the same baseline (s42, pre–parallel-routing stack). The column is Δ to V6+TTT BPB — positive means worse. Kept here for the record; none of these are in the submission.

| # | Change | Δ V6+TTT | Notes |
|---|---|---|---|
| 1 | LoRA TTT v2, rank 32, lr 3e-4 | +0.147 | Full eval. LoRA updates propagate into downstream layers that were not trained to absorb them. |
| 2 | LoRA TTT v2, rank 32, lr 1e-4 | +0.168 | Lower LR, same failure mode. The learning rate was not the problem. |
| 3 | Z-lane $\gamma$-only TTT | +0.00236 (weaker than controls-only) | Adding a learned gamma-gated path and TTT-ing only the gates. Rank too low; controls-only TTT at 9.7K params is strictly better. |
| 4 | `MUON_WD=0.05` | +0.00135, over budget | Lower decay improves fp32 fit but blows up the quantization gap and the artifact size. |
| 5 | `ENABLE_LOOPING_AT=0.35` | +0.00088 | Earlier loop activation. Raw SW improves by 0.00222, V6+TTT regresses: the model becomes more co-adapted to the looped phase and TTT extraction drops. Useful data point — first experiment where we saw raw and V6+TTT move in opposite directions. |
| 6 | `LOOP_START=3 LOOP_END=6` | +0.01734 | Wider 4-block loop. Costs one non-loop layer which the model was using for something important. |
| 7 | `PARALLEL_RESIDUAL_START=6` | +0.00416 | Applies two-lane routing starting at layer 6 instead of 8. More parallel layers is not better here — the gradient-shortcut benefit is specifically about the last few decoder layers being undertrained. |
| 8 | `attn_temp + block_gate + skip_routing` | +0.00195 | Trained scalar gates on every block. Improves raw BPB, destroys V6+TTT. Each new gate tightens co-adaptation. |
| 9 | `block_bias + skip_routing` | +0.00117 | Same story, fewer parameters, same outcome. |

The LoRA TTT experiments (1, 2) are the largest regressions because they corrupt the model's forward pass at eval time. The trained-gate experiments (8, 9) are the most diagnostic: they directly motivated the coupling-flexibility theory that eventually led us to implement parallel routing for the wrong reason.

</details>

## Files

- `train_gpt.py` — **the scored artifact**. Self-contained LZMA stub that decompresses, builds the CUTLASS kernel, trains the model, then runs V6 + TTT eval. Contains minified versions of all source files below.
- `human_readable/` — the identical unminified source code, provided for reviewer convenience and ease of review. Not used at runtime; the stub carries its own compressed copy.
  - `train_gpt.py` — model (including two-lane routing), training loop, GPTQ, serialization, eval functions
  - `tapin_cpp.py` — C++ Tap-In matcher (single-file `load_inline`)
  - `_runner.py` — end-to-end orchestrator: train → monkey-patch MLP → install V6 → TTT eval
  - `cutlass_evt_fusion/` — fused MLP backward kernel

<details>
<summary><b>Reproduce — end-to-end on a fresh 8×H100 box</b></summary>

### 0. Hardware

- 8×H100 80GB SXM (Hopper, sm_90a). The CUTLASS EVT kernel and FA3 require Hopper.
- ECC OFF gives consistent results.

### 1. Python + PyTorch + FA3

```bash
# Python 3.10 or 3.12
python3 -m venv venv && source venv/bin/activate

# PyTorch 2.9.1+cu128
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Flash Attention 3 prebuilt wheel (do NOT compile from source)
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

# Other deps
pip install sentencepiece brotli numpy ninja
```

### 2. CUTLASS headers (one-time, system-wide)

```bash
sudo git clone --depth=1 https://github.com/NVIDIA/cutlass /opt/cutlass
```

### 3. Download the SP8192 dataset

The dataset and tokenizer are pre-built on HuggingFace under the parameter-golf data repo. Place them so the structure is:

```
~/data/
  datasets/fineweb10B_sp8192/
    fineweb_train_*.bin     (128 shards)
    fineweb_val_*.bin       (1 shard)
  tokenizers/
    fineweb_8192_bpe.model
```

Then `export DATA_DIR=~/data/`.

### 4. Run (train + V6 + TTT eval, end-to-end)

```bash
SEED=2025 DATA_DIR=$DATA_DIR \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`train_gpt.py` is self-contained — it decompresses the code, builds the CUTLASS kernel, trains the model (~10 min), then automatically runs V6 + TTT eval (~7 min). No separate eval step needed. All tuned env vars (`LOOP_START=3`, `MUON_MOMENTUM=0.98`, `MUON_WD=0.12`, `MATRIX_CLIP_SIGMAS=13.1`, `PARALLEL_RESIDUAL_START=8`, `HESSIAN_CLIP_LAMBDA=0`, `GPTQ_RESERVE_SECONDS=4`, `GPTQ_CALIBRATION_BATCHES=16`, `TAPIN_V6_CROSS_W=0.12`) are set by the stub itself via `os.environ.update(...)` before `exec`'ing the runner.

### 5. Expected output

For `SEED=2025`:

```
=== V6 + TTT ===
  TTT: lr=0.005 epochs=3 chunk=32768 freeze=0
  val_loss: 2.772468  val_bpb: 1.073313  time: 405s
```

The headline number is **`val_bpb: 1.073313`**. To reproduce the 3-seed mean of **1.073938** run with `SEED=42` and `SEED=1234` and average.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `val_bpb` $\approx 1.16$ instead of $1.08$ | `torch.compile` was stripped — verify `eval_val_sliding_ttt` has `logits_fn = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)` |
| `val_bpb` $\approx 1.085$ instead of $1.078$ (no V6 effect) | `TAPIN_CPP=1 TAPIN_V4_ENABLED=1 TAPIN_V6_CROSS=1` env vars are set automatically by the stub; check `human_readable/_runner.py` if running manually |
| Training BPB is $0.001$ worse than expected | Check `HESSIAN_CLIP_LAMBDA=0` is set |
| `RuntimeError: Ninja is required` | `pip install ninja` |
| `RuntimeError: operator cutlass_evt::gemm_mul does not exist` | CUTLASS headers not found at `/opt/cutlass` (step 2) |
| `Inference tensors cannot be saved for backward` (during TTT) | The TTT SCORE phase must use `torch.no_grad()`, NOT `torch.inference_mode()` (this is correct in the shipped code) |

</details>
