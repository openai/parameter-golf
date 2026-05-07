import re, sys
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

# 3. Add SmearGate to forward pass after tok_emb
old = '        x = self.tok_emb(input_ids)\n'
new = '        x = self.tok_emb(input_ids)\n        if self.smeargate is not None:\n            x = self.smeargate(x)\n'
if old not in text:
    print("ERROR: Could not find forward tok_emb"); sys.exit(1)
text = text.replace(old, new, 1)

open('train_gpt.py', 'w').write(text)

# Verify syntax
import ast
try:
    ast.parse(text)
    print("PATCHED OK: SmearGate added. Enable with USE_SMEARGATE=1")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
