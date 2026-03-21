import sys, ast
text = open('train_gpt.py').read()

# 1. Add SmearGate class before Block class
old = 'class Block(nn.Module):'
new = '''class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))
        prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        return (1 - g) * x + g * prev

class Block(nn.Module):'''
if old not in text:
    print("ERROR: Could not find Block class"); sys.exit(1)
text = text.replace(old, new, 1)

# 2. Add SmearGate to GPT init after tok_emb
old = '        self.tok_emb = nn.Embedding(vocab_size, model_dim)'
new = '        self.tok_emb = nn.Embedding(vocab_size, model_dim)\n        self.smeargate = SmearGate(model_dim) if bool(int(os.environ.get("USE_SMEARGATE", "0"))) else None'
if old not in text:
    print("ERROR: Could not find tok_emb"); sys.exit(1)
text = text.replace(old, new, 1)

# 3. Add SmearGate to forward pass
old = '        x = self.tok_emb(input_ids)\n'
new = '        x = self.tok_emb(input_ids)\n        if self.smeargate is not None:\n            x = self.smeargate(x)\n'
if old not in text:
    print("ERROR: Could not find forward tok_emb"); sys.exit(1)
text = text.replace(old, new, 1)

# 4. Add MuonWD: find the Muon step where weights are updated
old = '''            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()'''
new = '''            wd = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
            curr = 0
            for p in params:
                if wd > 0:
                    p.mul_(1 - wd * lr)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()'''
if old not in text:
    print("WARNING: Could not find Muon step loop for MuonWD. Trying alternative...")
    # Try alternative pattern
    old2 = 'p.add_(g, alpha=-lr)\n                curr += p.numel()'
    new2 = 'if float(os.environ.get("MUON_WEIGHT_DECAY", 0.0)) > 0:\n                    p.mul_(1 - float(os.environ.get("MUON_WEIGHT_DECAY", 0.0)) * lr)\n                p.add_(g, alpha=-lr)\n                curr += p.numel()'
    if old2 not in text:
        print("WARNING: Could not add MuonWD. SmearGate only.")
    else:
        text = text.replace(old2, new2, 1)
        print("MuonWD added (alternative pattern)")
else:
    text = text.replace(old, new, 1)
    print("MuonWD added")

open('train_gpt.py', 'w').write(text)

try:
    ast.parse(text)
    print("PATCHED OK: SmearGate + MuonWD. Enable: USE_SMEARGATE=1 MUON_WEIGHT_DECAY=0.01")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
