# Non-Record: Mamba SSM Backbone + Selective Scan + Muon Split + Int8+Zlib  
val_bpb = **2.0393** (roundtrip) | ~14.88 MB | 8×H100

---

## Single-Seed Result

Seed | Pre-Quant BPP | Roundtrip BPP | Artifact  
---|---|---|---  
32 | 2.0192 | 2.0393 | 14,887,866  

---

## Delta vs Competitive Track

Current 10-min submissions: ~1.1–1.2 BPP  
This run: **+~0.8–0.9 BPP**

Not competitive → submitted as **non-record SSM attempt**

---

## Key Techniques

**Mamba SSM Backbone** — replaces attention entirely with selective scan sequence mixing  

**Selective Scan Integration** — direct `mamba-ssm` integration with explicit signature validation  

**Depthwise Conv + Scan Mixer** — conv → (dt, B, C) → selective scan → gated projection  

**Encoder/Decoder Skip Layout** — first half stores activations, second half reuses via learned skip weights  

**Muon + Adam Split Optimizer**  
- Muon → matrix params  
- Adam → embeddings + scalars  

**Torch Compile (Non-Fullgraph)** — scan path excluded for stability  

**Int8 + Zlib Compression**  
- per-row int8 (matrices)  
- per-tensor int8 (vectors)  
- fp16 passthrough (small/control tensors)  

**Exact Roundtrip Eval** — compressed artifact is reloaded + evaluated (not pre-quant)

---

## Architecture

8L × 512d Mamba SSM, MLP 2×, RMSNorm, tied embeddings, logit softcap=30  

Mamba config:
- state_dim = 8  
- expand = 1  

Block:
- RMSNorm → Mamba → residual  
- RMSNorm → ReLU² MLP → residual  

Skip structure:
- encoder: layers [0–3] store  
- decoder: layers [4–7] reuse (learned weights)  

No attention, no RoPE, no KV — pure SSM stack.

---

## Training

- batch: 524k tokens  
- seq_len: 1024  
- ~11.5k steps in 599s  

Optim:
- Muon (matrix params)  
- Adam (embeddings + scalars)  

Compile warmup is reset → no extra training.

---

## Quantization

Post-training only:

- row-wise int8  
- fp16 passthrough (small tensors)  
- zlib compression  

Sizes:
- raw model: 61.4 MB  
- compressed: **14.84 MB**  
- total: **14.88 MB**

---

## Training Behavior

Best model occurs **before final step**:

- best pre-quant: **2.0192 BPP**  
- final roundtrip: **2.0393 BPP**

Quality degrades late due to:
- fixed wallclock stop  
- weak warmdown for SSM  
- quantization loss  

---

## Limitations

- very small SSM (state_dim=8, expand=1)  
- no SSM-specific tuning  
- PyTorch Conv1d fallback (no causal kernel)  
- simple post-training quantization  

---

## Compliance

- standard causal LM  
- no eval-time adaptation  
- single-pass eval  
- no caching / biasing  

- artifact <16MB  
- training <600s  
- roundtrip metrics reported  

---

## Reproduction

```bash
pip install -U pip setuptools wheel ninja packaging
pip install mamba-ssm --no-build-isolation --prefer-binary

RUN_ID=seed32 SEED=32 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py