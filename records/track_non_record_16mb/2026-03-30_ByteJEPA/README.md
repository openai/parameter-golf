# True Byte-Level JEPA for Parameter Golf

Hey there! This submission is a direct answer to the open **JEPA [bounty/PR request]** in the challenge.

## Why this is different
Every other submission on the leaderboard (whether it's testing new layers, distillation, or quantization tricks) still trains the model the normal way: by predicting the next token using Cross-Entropy loss.

But a true JEPA (Joint-Embedding Predictive Architecture) is fundamentally different. It doesn't guess the next token. Instead, it tries to predict the *abstract concept* of what comes next, operating entirely in the model's hidden layers. 

Because a pure JEPA model doesn't guess specific characters, it's mathematically impossible for it to output the Bits Per Byte (BPB) score the challenge requires.

## How we solved the JEPA vs. BPB problem
To give the challenge the BPB score it needs without ruining the pure JEPA architecture, we use the standard "Two-Phase" approach from the official Meta JEPA papers:

1. **Phase 1: Pure JEPA Pretraining (70% of the 10-minute clock)**
   - The model learns entirely by predicting its own future hidden states.
   - We use a trick called SIGReg to keep the model from cheating and outputting the same vector every time. 
   - During this phase, the model doesn't even try to guess bytes. It's just learning the raw structure of the language.

2. **Phase 2: Supervised Fine-Tuning (30% of the 10-minute clock)**
   - At the 7-minute mark, the script automatically shifts gears.
   - It attaches a simple translation layer on top of the model and fine-tunes the network to map its abstract thoughts into actual byte probabilities.
   - This lets us cleanly output the rigorous `val_bpb` metric you see in the leaderboard.

## Under the Hood
- **No Tokenizer**: We dumped the tokenizer completely. The model reads raw UTF-8 bytes (`vocab_size=256`). This forces the JEPA to learn complex word boundaries blindly from scratch—a true test of its representation power.
- **Predictor Network**: Added a 2-Layer MLP that tries to guess the target's hidden state. We funded the parameter cost of this network by saving space on the disabled tokenizer embeddings!
- **Constraint Safe**: Our code tracks the 16MB file limits natively.

It's worth noting that because byte-level models have a shorter effective "memory window" than tokenized ones in a 10-minute race, this won't shatter the top of the leaderboard purely on BPB. But as the very first submission to successfully change the core learning objective from token-prediction to representation-prediction, we hope it hits the "weird & creative" mark for the non-record track!

## Run Command

You'll need the byte-level data shards (script included).

```bash
# 1. Convert the standard SP1024 shards into raw Bytes
DATA_PATH_BASE="../../data" python transpile_to_bytes.py

# 2. Run the Two-Phase JEPA trainer
DATA_PATH=./data/datasets/fineweb10B_bytes \
VOCAB_SIZE=256 \
JEPA_PRETRAIN_FRAC=0.7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Challenge Metrics 

1x RTX 5090, 3600s wallclock, 4141 steps:
- Pre-quant: `val_loss:0.9308 val_bpb:1.3429`
- Post-quant (int8+zlib roundtrip): `val_loss:0.9355 val_bpb:1.3496`
- Serialized model int8+zlib: `15080959 bytes`
- Code size: `51873 bytes`
- Total submission size int8+zlib: `15132832 bytes`
