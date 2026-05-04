import torch
import torch.nn.functional as F
import sentencepiece as spm
import zlib
import io
import os

from train_gpt_optimized import dequantize_state_dict_int8, GPT, Hyperparameters

# Configure hyperparameters to match the fast training run
os.environ["NUM_LAYERS"] = "4"
os.environ["MODEL_DIM"] = "256"
os.environ["NUM_HEADS"] = "4"
os.environ["NUM_KV_HEADS"] = "4"

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("./data/tokenizers/fineweb_1024_bpe.model")

# Load and decompress model weights
print("Loading and dequantizing model...")
with open("final_model.int8.ptz", "rb") as f:
    q_obj = torch.load(io.BytesIO(zlib.decompress(f.read())), map_location="cpu")
    
state_dict = dequantize_state_dict_int8(q_obj)

# Initialize model
args = Hyperparameters()
args.num_steps = 4
args.model_dim = 256
args.num_heads = 4
args.num_kv_heads = 4
model = GPT(args).bfloat16()
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")

def patched_forward(self, input_ids):
    x = F.rms_norm(self.tok_emb(input_ids), (self.args.model_dim,))
    x0 = x
    for i in range(self.args.num_steps):
        block_idx = i % self.args.num_unique_blocks
        x = self.unique_blocks[block_idx](x, x0)
    
    x = self.final_norm(x) # [bsz, seq_len, dim]
    logits_proj = F.linear(x, self.tok_emb.weight) if self.args.tie_embeddings else self.lm_head(x)
    logits = self.args.logit_softcap * torch.tanh(logits_proj / self.args.logit_softcap)
    return logits

GPT.forward = patched_forward

def sample_logits(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Top-k
    if top_k > 0:
        values, indices = torch.topk(probs, top_k)
        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(0, indices, values)
        probs = probs_filtered / probs_filtered.sum()

    # Top-p (nucleus)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    cutoff = cumulative_probs > top_p
    if torch.any(cutoff):
        cutoff_idx = torch.where(cutoff)[0][0]
        sorted_probs[cutoff_idx:] = 0
        if sorted_probs.sum() > 0:
            sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs)
        probs.scatter_(0, sorted_indices, sorted_probs)

    return torch.multinomial(probs, 1).item()

def generate(prompt, max_tokens=80, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2):
    tokens = sp.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.clone()

    for _ in range(max_tokens):
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(generated)[0, -1].float()
        
        # repetition penalty
        for token_id in set(generated[0].tolist()):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

        next_token = sample_logits(logits, temperature, top_k, top_p)
        generated = torch.cat([generated, torch.tensor([[next_token]])], dim=1)

    return sp.decode(generated[0].tolist())

# Test prompts
prompts = [
    "The future of AI is",
    "Once upon a time",
    "India is known for",
    "The meaning of life is"
]

for p in prompts:
    print("\nPROMPT:", p)
    print(generate(p))

