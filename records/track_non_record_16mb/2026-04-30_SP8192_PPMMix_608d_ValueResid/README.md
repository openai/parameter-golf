# SP8192 + Value Residual + Byte-Level PPM Mixture

## Overview

This submission is the result of an incremental research process rather than a single clean-sheet design.  
The codebase was built step by step across many rounds of experiments. Instead of hard-coding one architecture, we intentionally exposed most research ideas as environment-controlled switches so that we could run controlled ablations quickly and compare alternatives under the same training and evaluation framework.

The final code therefore serves two purposes:

1. **A trainable compression model**
2. **A flexible experiment platform** for architecture, optimization, tokenizer, evaluation, and mixture research

Our final strongest submission combines:

- **SentencePiece 8192 tokenizer**
- **9-layer Transformer backbone**
- **BiFPN2 / XSA / N-gram baseline stack**
- **Value Residual in the last 2 layers**
- **Larger-capacity backbone (608d / MLP×3)**
- **Byte-level PPM mixture at evaluation time**

The best reproduced result in this branch is:

- **Neural roundtrip exact val_bpb:** `1.15864608`
- **PPM mixture val_bpb:** `0.832925`

This is a **non-record / unlimited-compute style submission**.  
It is intended to document the method and results clearly. It is **not claimed here as a 10-minute 8×H100 record-track run**.

---

## Design Philosophy

A major goal of this project was to avoid baking one fragile idea directly into the model.  
Instead, we built a single training script with many research switches so we could answer questions like:

- Does a tokenizer change matter more than a block change?
- Does capacity help more than architectural novelty?
- Is value-path routing more useful than additional parallel branches?
- Are small state-space side lanes actually more efficient than attention here?
- Can a lossless/statistical mixture dominate pure-neural improvements?

Because of that, the code includes toggles for:

- tokenizer-dependent training and evaluation
- multiple skip/fusion schemes
- XSA, V-skip, cross-layer V and KV sharing
- PLE
- MTP
- N-gram augmentation and fade scheduling
- depth recurrence
- value residual
- parallel residual v1 and parallel residual v2
- gated linear / conv-gate / tiny SSM side lanes
- LoRA-TTT
- byte-level PPM mixture evaluation

Most of these ideas were kept behind flags so they could be turned on and off in ablation sweeps without rewriting the training loop.

---

## Code Structure and Research Platform

The final script is not a minimal competition-only script. It is a research scaffold that gradually accumulated features as experiments progressed.

### Core training infrastructure
The script includes:

- distributed training support
- Muon optimizer for matrix parameters
- Adam for scalar/control parameters
- EMA
- late QAT
- tokenizer-aware val_bpb evaluation
- sliding-window evaluation
- telemetry and profiling hooks

### Architecture flags
The following major research directions are supported through hyperparameter switches:

- `BIFPN2_MODE`
- `XSA_ENABLED`
- `NGRAM_*`
- `VALUE_RESIDUAL_*`
- `CROSS_LAYER_V_*`
- `CROSS_LAYER_KV_*`
- `PLE_*`
- `PARALLEL_RESIDUAL_*`
- `PARALLEL_V2_*`
- `DEPTH_RECUR_*`
- `TTT_*`
- `LORA_TTT_*`
- `PPM_*`

This allowed us to run many controlled sweeps without changing the surrounding code.

---

## Experimental Path

Our final result did **not** come from one idea.  
It came from a sequence of findings.

### 1. Tokenizer scaling mattered a lot
Early SP1024 experiments consistently plateaued around the high `1.27x` range.  
Moving to larger tokenizers gave immediate gains:

- **SP1024:** roughly `~1.27`
- **SP4096:** roughly `~1.24`
- **SP8192:** roughly `~1.22` before stronger backbone tuning

This showed that tokenization efficiency was one of the highest-leverage early improvements.

### 2. Capacity still mattered strongly
We then tested model-capacity changes around the strongest SP8192 line.

Representative pure-neural results:

- `512d / mlp2` class: around `~1.21`
- `512d / mlp3`: around `~1.19`
- `576d / mlp3`: around `~1.17`
- `608d / mlp3`: **`1.1587258`**
- `576d / mlp4`: **`1.15911626`**

This strongly suggested that in our regime, increasing effective model capacity was still more valuable than adding many exotic modules.

### 3. Value Residual became the strongest architectural improvement
Across many rounds of ablations, **Value Residual** was the most consistent structural gain.

A representative comparison:

- baseline strongline (`qk400`): about `1.1794`
- `value_resid_last2`: about `1.1712`

This was a clear and stable gain.  
Further ablations showed:

- `last2` worked better than `last4` or `last6`
- moderate value blending was better than aggressive late-layer replacement
- the gain remained strong on larger-capacity SP8192 backbones

This became the main architectural direction that survived repeated testing.

### 4. Many other structural ideas were explored, but did not become the mainline
We tested a wide range of alternatives:

- PLE
- parallel residual v1
- parallel residual v2
- gated merge variants
- cross-layer V residual variants
- cross-layer KV sharing
- tiny SSM / Mamba-like side lanes
- conv-gate side lanes
- LoRA-TTT
- depth recurrence

Some produced small gains in isolated runs, but none were as consistently useful as:

- tokenizer scaling
- capacity scaling
- value residual

In particular, small SSM-style side lanes did **not** outperform a stronger conventional backbone in our experiments.

### 5. Byte-level PPM mixture changed the regime
The largest jump came when we moved beyond pure-neural evaluation and added a **byte-level PPM mixture**.

We first validated that the mixture was real rather than noise.  
For example, with a strong 608d / mlp3 / value-residual backbone:

- neural roundtrip exact: about `1.1589`
- PPM mixture (`order=5`, `thr=0.9`, `lo=0.10`, `hi=0.80`): about `0.9385`

We then investigated lambda sensitivity and found a stronger regime:

- `hi=0.775`: `0.883714`
- `hi=0.75`: `0.832925`

The `0.832925` result was repeated and therefore treated as real, not as a one-off anomaly.

This was the point where the project moved from “improving the neural backbone” to “combining a strong neural model with a byte-level statistical corrector.”

---

## Summary of Key Experimental Findings

### Strong positive findings
- Larger tokenizer vocabularies helped substantially
- Capacity scaling remained very effective
- Value Residual was the strongest consistent architecture change
- Byte-level PPM mixture produced the largest overall gain

### Weak or inconsistent findings
- PLE sometimes helped slightly, but did not remain on the mainline
- Parallel residual variants were at best marginal on top of value residual
- Cross-layer V and KV sharing were not strong mainline improvements

### Negative or deprioritized findings
- gated parallel merge
- conv-gate side lanes
- small SSM/Mamba-inspired side lanes
- current LoRA-TTT variants for mainline use
- depth recurrence in its current form

---

## Final Mainline Configuration

The final strongest reproduced backbone used:

- **Tokenizer:** SentencePiece 8192
- **Layers:** 9
- **Model dimension:** 608
- **Heads:** 8
- **KV heads:** 4
- **MLP multiplier:** 3
- **QK gain init:** 4.0
- **BiFPN2:** enabled
- **XSA:** enabled in the last 4 layers
- **N-gram features:** enabled
- **Value Residual:** enabled in the last 2 layers
- **EMA:** enabled
- **Late QAT:** enabled

The final strongest mixture used:

- **PPM enabled**
- **Order:** 5
- **Confidence threshold:** 0.9
- **Lambda low:** 0.10
- **Lambda high:** 0.75
- **Neural byte projection:** `spread_root`

---

## Why the Code Has So Many Switches

The code may appear larger and more feature-heavy than a minimal submission script.  
This is intentional.

We were not optimizing only for final compactness during research.  
We were optimizing for:

- fast iteration
- controlled ablation
- fair comparisons between ideas
- reuse of one stable training loop
- reproducible experiment sweeps

This let us test ideas without changing unrelated components.  
For example, we could compare:

- tokenizer changes vs architecture changes
- value residual vs parallel residual
- MLP capacity vs SSM side lanes
- neural-only vs byte-level mixture

using the same general framework.

In practice, this made it much easier to discover which ideas were truly load-bearing.

---

## Lessons Learned

### 1. Simple capacity improvements beat many clever block modifications
A stronger standard backbone often outperformed more exotic second-lane or recurrent additions.

### 2. Value-path routing mattered more than many alternative residual tricks
Value Residual consistently helped more than most parallel or side-lane variants.

### 3. Tokenization and byte-level evaluation are first-class concerns in this benchmark
This benchmark is not only about building a stronger LM.  
Tokenizer efficiency and byte-level correction matter enormously.

### 4. System-level methods can dominate pure-neural improvements
The transition from pure-neural `~1.1586` to mixed `0.832925` was much larger than any single block-level improvement.  
This suggests that at the current frontier, system design is at least as important as backbone design.

---

## Reproduction Notes

This submission was built and tested through multiple sweeps using environment-variable controlled configs.

Representative final backbone:
- `MODEL_DIM=608`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3`
- `VALUE_RESIDUAL_ENABLED=1`
- `VALUE_RESIDUAL_LAST_N_LAYERS=2`
- `QK_GAIN_INIT=4.0`

Representative final mixture:
- `PPM_ENABLED=1`
- `PPM_ORDER=5`
- `PPM_CONF_THRESHOLD=0.9`
- `LAMBDA_LO=0.10`
- `LAMBDA_HI=0.75`

The strongest reproduced run in this folder is the repeated `hi=0.75` configuration.

---

## Submission Status

This folder documents a **non-record / unlimited-compute** style submission.

It is intended to capture:

- the final strongest reproduced method
- the progression of the experimental mainline
- the code path that enabled the result

It should be read as a record of the method and its experimental evolution, rather than as a claim of record-track compliance.

---

## Included Files

This folder contains:

- `train_gpt.py` — experiment and training script with all major research switches
- `submission.json` — metadata and best-result summary
- `config.json` — final selected configuration
- `seed_runs.csv` — representative run summary
- `train.log` — log from the final best reproduced run
- `requirements.txt` — Python dependencies

---

## Final Result

### Best reproduced pure-neural score
- **`1.15864608`** roundtrip exact val_bpb

### Best reproduced mixed score
- **`0.832925`** ppm_mix_bpb

This final result emerged from a long sequence of ablations, with the most important steps being:

1. tokenizer scaling
2. capacity scaling
3. value residual
4. byte-level PPM mixture
