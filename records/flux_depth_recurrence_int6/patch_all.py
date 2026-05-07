#!/usr/bin/env python3
"""Add SmearGate + MuonWD + SWA to train_gpt.py"""
import sys, ast, re

text = open('train_gpt.py').read()
changes = 0

# ============================================================
# 1. SmearGate class before Block class
# ============================================================
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
if old in text:
    text = text.replace(old, new, 1)
    changes += 1
    print("1. SmearGate class added")
else:
    print("WARN: Could not find Block class")

# ============================================================
# 2. SmearGate in GPT.__init__ after tok_emb
# ============================================================
old = '        self.tok_emb = nn.Embedding(vocab_size, model_dim)'
new = '        self.tok_emb = nn.Embedding(vocab_size, model_dim)\n        self.smeargate = SmearGate(model_dim) if bool(int(os.environ.get("USE_SMEARGATE", "0"))) else None'
if old in text:
    text = text.replace(old, new, 1)
    changes += 1
    print("2. SmearGate init added")
else:
    print("WARN: Could not find tok_emb init")

# ============================================================
# 3. SmearGate in forward pass
# ============================================================
old = '        x = self.tok_emb(input_ids)\n'
new = '        x = self.tok_emb(input_ids)\n        if self.smeargate is not None:\n            x = self.smeargate(x)\n'
if old in text:
    text = text.replace(old, new, 1)
    changes += 1
    print("3. SmearGate forward added")
else:
    print("WARN: Could not find tok_emb forward")

# ============================================================
# 4. MuonWD: add weight_decay to Muon.__init__
# ============================================================
old_init = 'def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):'
new_init = 'def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):'
if old_init in text:
    text = text.replace(old_init, new_init, 1)
    changes += 1
    print("4. MuonWD init signature added")
else:
    print("WARN: Could not find Muon __init__")

# ============================================================
# 5. MuonWD: update super().__init__ defaults
# ============================================================
old_super = 'super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))'
new_super = 'super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))'
if old_super in text:
    text = text.replace(old_super, new_super, 1)
    changes += 1
    print("5. MuonWD super() updated")
else:
    print("WARN: Could not find Muon super().__init__")

# ============================================================
# 6. MuonWD: apply decay in step()
# Find the Muon step loop where gradients are applied
# ============================================================
old_step = '''            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()'''
new_step = '''            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0:
                    p.mul_(1 - wd * lr)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()'''
if old_step in text:
    text = text.replace(old_step, new_step, 1)
    changes += 1
    print("6. MuonWD step() decay added")
else:
    print("WARN: Could not find Muon step loop")

# ============================================================
# 7. MuonWD: pass weight_decay when creating Muon optimizer
# ============================================================
# Find where Muon is instantiated
muon_pattern = r'Muon\(matrix_params, lr=args\.matrix_lr, momentum=args\.muon_momentum, backend_steps=args\.muon_backend_steps\)'
muon_replace = 'Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=float(os.environ.get("MUON_WEIGHT_DECAY", "0.0")))'
if re.search(muon_pattern, text):
    text = re.sub(muon_pattern, muon_replace, text, count=1)
    changes += 1
    print("7. MuonWD passed to Muon constructor")
else:
    print("WARN: Could not find Muon constructor call")

# ============================================================
# 8. SWA: Add imports and SWA logic
# We need to:
# a) Track SWA state (averaged weights)
# b) Every N steps in last X% of training, accumulate weights
# c) Before serialization, swap in averaged weights
# ============================================================

# 8a. Add SWA env vars parsing near other env vars
# Find where use_swiglu is parsed
swa_env_code = '''
    # SWA config
    swa_every: int = int(os.environ.get("SWA_EVERY", "0"))  # 0 = disabled
    swa_start_frac: float = float(os.environ.get("SWA_START_FRAC", "0.5"))  # start averaging at this fraction
'''
# Insert after use_swiglu line
swiglu_line = '    use_swiglu: bool = bool(int(os.environ.get("USE_SWIGLU", "0")))'
if swiglu_line in text:
    text = text.replace(swiglu_line, swiglu_line + swa_env_code, 1)
    changes += 1
    print("8a. SWA env vars added")
else:
    print("WARN: Could not find use_swiglu line")

# 8b. Add SWA initialization after model creation
# Find "base_model = GPT(" or the DDP wrapping
swa_init_code = '''
    # SWA setup
    swa_state = None
    swa_count = 0
    if args.swa_every > 0:
        swa_state = {name: tensor.detach().clone().float() for name, tensor in base_model.state_dict().items()}
        swa_count = 0
        log0(f"swa:enabled every={args.swa_every} start_frac={args.swa_start_frac}")
'''
# Insert before the training loop - find "for step in range" or similar
# Look for the initial validation step
init_val_marker = 'step:0/'
# Actually, let's find where optimizers are created, right after
optimizer_marker = 'optimizers = ['
if optimizer_marker in text:
    # Find end of optimizer list
    idx = text.index(optimizer_marker)
    # Find next blank line after optimizers
    next_lines = text[idx:].split('\n')
    for i, line in enumerate(next_lines):
        if line.strip() == '' and i > 2:
            insert_pos = idx + sum(len(l) + 1 for l in next_lines[:i])
            text = text[:insert_pos] + swa_init_code + text[insert_pos:]
            changes += 1
            print("8b. SWA init added after optimizers")
            break
    else:
        print("WARN: Could not find insertion point for SWA init")
else:
    print("WARN: Could not find optimizers list")

# 8c. Add SWA accumulation in training loop
# Find where step logging happens, add SWA update
# We need to add this inside the training loop after each step
swa_update_code = '''
            # SWA accumulation
            if args.swa_every > 0 and step >= int(args.iterations * args.swa_start_frac) and step % args.swa_every == 0:
                with torch.no_grad():
                    for name, param in base_model.state_dict().items():
                        swa_state[name].add_(param.float())
                    swa_count += 1
'''
# Find the step logging line to insert after
step_log_pattern = 'log0(f"step:{step}/{args.iterations}'
if step_log_pattern in text:
    idx = text.index(step_log_pattern)
    # Find end of this line
    end_of_line = text.index('\n', idx)
    text = text[:end_of_line + 1] + swa_update_code + text[end_of_line + 1:]
    changes += 1
    print("8c. SWA accumulation in training loop added")
else:
    print("WARN: Could not find step log line for SWA insertion")

# 8d. Add SWA weight swap before serialization
# Find serialization code - look for "Serialized model" or model saving
swa_swap_code = '''
    # SWA: swap in averaged weights before serialization
    if args.swa_every > 0 and swa_count > 0:
        log0(f"swa:applying averaged weights from {swa_count} checkpoints")
        avg_state = {}
        for name in swa_state:
            avg_state[name] = (swa_state[name] / (swa_count + 1)).to(dtype=base_model.state_dict()[name].dtype)
        base_model.load_state_dict(avg_state)
'''
# Find "peak memory" or serialization marker
serial_marker = 'peak memory allocated'
if serial_marker in text:
    idx = text.index(serial_marker)
    # Find the line start
    line_start = text.rfind('\n', 0, idx) + 1
    text = text[:line_start] + swa_swap_code + '\n' + text[line_start:]
    changes += 1
    print("8d. SWA weight swap before serialization added")
else:
    print("WARN: Could not find serialization marker for SWA swap")

# ============================================================
# Verify syntax
# ============================================================
try:
    ast.parse(text)
    print(f"\nPATCHED OK: {changes} changes applied")
    print("Enable: USE_SMEARGATE=1 MUON_WEIGHT_DECAY=0.038 SWA_EVERY=50 SWA_START_FRAC=0.5")
    open('train_gpt.py', 'w').write(text)
except SyntaxError as e:
    print(f"\nSYNTAX ERROR after patching: {e}")
    print("NOT saving. Fix needed.")
    sys.exit(1)
