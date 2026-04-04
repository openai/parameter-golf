# LOGOS-44: Parameter Golf Submission (Z=0 Architecture)

**Artifact Size:** 1.2 MB (`logos44_quantum.pth`) — well below the 16 MB limit  
**Parameters:** ~300,000 trainable  
**Compute:** Designed for 10-minute convergence on 8×H100  
**License:** MIT

## The Paradigm: Coherence Engineering

Current LLMs waste megabytes on flat layers and grammatical noise. LOGOS-44 abandons the standard Transformer architecture. Instead, it utilizes a **Toroidal Bottleneck** with **CDMA Field Decoding** and **Coherence Gating**.

The model does not "learn" in the traditional sense. It undergoes a thermodynamic phase transition driven by iterative signal routing through a single geometric throat.

### Key Innovations:

1. **Toroidal Bottleneck** — Information mapped to sin·cos geometry (rank-32) to prevent byte-level interference. 44 passes through the throat create depth without parameter duplication.
2. **CDMA Field Decoder** — Knowledge stored as a superposition of 128 entangled signals, accessed via learned spreading codes. Optional quantum initialization from IBM Quantum hardware.
3. **Coherence Gate** — Learned sigmoid gate balances field signal vs. residual at each depth step. Converges to ~0.50 (perfect balance).
4. **Quantum Seed** — Model locked to a point in time where 48 qubits collapsed on `ibm_fez`. Seed: `268097198940526`.

### Architecture

| Component        | Value                              |
|------------------|------------------------------------|
| Embedding dim    | 256                                |
| Toroidal rank    | 32                                 |
| Routing depth    | 44 passes                         |
| Field signals    | 128 (CDMA-encoded)                 |
| Vocab size       | 512 (byte + 40 archetypes)         |
| Max sequence     | 512                                |
| Parameters       | ~300k                              |

### Repository Structure

```
├── core/               # Original lattice architecture
├── data/               # Data pipeline & tokenizer specs
├── logos44/             # Quantum Edition (micro-LLM)
│   ├── train.py         # Full training pipeline
│   ├── logos44_micro.py  # Micro architecture
│   ├── quantum_codes.py  # Quantum CDMA generation
│   ├── README.md
│   └── logs/
│       └── training_300k.txt
├── nucleation/          # Supersaturated training data
├── records/             # Competition submission records
├── run_golf.py          # Quick nucleation launcher
├── train_gpt.py         # Baseline GPT trainer (from competition)
├── submission.json      # Competition metadata
└── logos44_quantum.pth  # Trained model (1.2 MB)
```

### Quick Start

```bash
# Train from scratch (classical mode)
cd logos44
python train.py

# Or run the nucleation launcher
python run_golf.py
```

### Status

⚠️ **val_bpb pending** — FineWeb validation evaluation requires 8×H100 compute.  
This is an experimental architecture submission exploring iterative compute depth as a substitute for parameter count.

*Built for the [OpenAI Parameter Golf Challenge](https://openai.com/index/model-craft-parameter-golf/).*
