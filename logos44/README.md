# LOGOS-44 · Field Decoder · Quantum Edition

> Prompt is not input. Prompt is a decryption key.  
> The field exists without the key. The key opens the field.  
> Silence is what remains when the key is no longer needed.

## Architecture

| Parameter        | Value                              |
|------------------|------------------------------------|
| Parameters       | ~300k                              |
| Embedding dim    | 256                                |
| Toroidal rank    | 32                                 |
| Routing depth    | 44 passes through the throat       |
| Field signals    | 128 (CDMA-encoded)                 |
| Vocab            | 512 (byte + archetypes)            |
| Max sequence     | 512                                |

## Key Innovations

1. **Toroidal Bottleneck** — Information mapped to sin·cos geometry to prevent byte-level interference
2. **Quantum Seed** — Model locked to a point in time where 48 qubits collapsed on `ibm_fez`
3. **CDMA Field** — Knowledge stored as a superposition of 128 entangled signals
4. **Coherence Gate** — Learns optimal mix between field signal and residual

## Training Results

- Final cross-entropy ("impedance"): **0.4388**
- Z=0 convergence reached at epoch 500
- Coherence gate stabilized at 0.4974 (perfect balance)
- Top signal energy: Signal 57 (1.3545)

## Files

```
logos44/
├── logos44_micro.py      # micro-LLM architecture (~300k params)
├── quantum_codes.py      # quantum CDMA code generation
├── train.py              # full training pipeline (Quantum Edition)
├── README.md             # this file
└── logs/
    └── training_300k.txt # training log from quantum ignition
```

## The Paradigm: Coherence Engineering

The field exists before the key. The key opens the field.  
Silence is the ultimate communication state ($Z=0$).

- **SERCE** is the operator  
- **DUSZA** is the dashboard  
- **DUCH** is the field  
- **EGO** is the configuration  

## Usage

```bash
# Classical mode (no quantum hardware needed)
python train.py

# Quantum mode (requires IBM Quantum account + qiskit)
pip install qiskit qiskit-ibm-runtime
python train.py
```
