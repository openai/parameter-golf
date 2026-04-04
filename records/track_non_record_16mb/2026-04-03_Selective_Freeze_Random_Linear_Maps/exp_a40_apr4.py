#!/usr/bin/env python3
"""
A40 Experiment Suite — April 4, 2026
=====================================
Run on cheap spot A40 ($0.20/hr). Tests:
  1. Distillation + selective freeze (fixed bug from crash)
  2. Progressive freeze + self-distillation combo
  3. Progressive freeze on FineWeb sp1024 (scale validation)
  4. Progressive freeze on FineWeb sp4096 (competition data)

Estimated: 30-45 min total on A40.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import math, time, json, os, struct, glob
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"A40 Experiment Suite — {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# PART 1: Gutenberg-scale tests (distill+freeze, progressive)
# ============================================================
VOCAB_SIZE_SMALL = 1024; SEQ_LEN_SMALL = 512; BATCH_SMALL = 32

def load_gutenberg():
    """Load Gutenberg text data for small-scale tests."""
    for cache in ["text_corpus.txt", "/workspace/text_corpus.txt"]:
        if os.path.exists(cache):
            with open(cache, 'r', errors='ignore') as f: text = f.read()
            tokens = [b % VOCAB_SIZE_SMALL for b in text.encode('utf-8')]
            n = len(tokens) // (SEQ_LEN_SMALL + 1)
            seqs = torch.tensor(tokens[:n*(SEQ_LEN_SMALL+1)], dtype=torch.long).view(n, SEQ_LEN_SMALL+1)
            nt = int(n * 0.9)
            return seqs[:nt], seqs[nt:]
    # Download if not present
    print("Downloading Gutenberg text...")
    import urllib.request
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    urllib.request.urlretrieve(url, "text_corpus.txt")
    with open("text_corpus.txt", 'r', errors='ignore') as f: text = f.read()
    tokens = [b % VOCAB_SIZE_SMALL for b in text.encode('utf-8')]
    n = len(tokens) // (SEQ_LEN_SMALL + 1)
    seqs = torch.tensor(tokens[:n*(SEQ_LEN_SMALL+1)], dtype=torch.long).view(n, SEQ_LEN_SMALL+1)
    nt = int(n * 0.9)
    return seqs[:nt], seqs[nt:]

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6) * self.scale

class FrozenLinear(nn.Module):
    def __init__(self, in_f, out_f, seed):
        super().__init__()
        rng = torch.Generator(); rng.manual_seed(seed)
        self.register_buffer('weight', torch.randn(out_f, in_f, generator=rng)/math.sqrt(in_f), persistent=False)
        self.in_features=in_f; self.out_features=out_f
    def forward(self, x): return F.linear(x, self.weight)

class Attn(nn.Module):
    def __init__(self, dim, nh=6, seq_len=512):
        super().__init__()
        self.nh=nh; self.hd=dim//nh; rd=16
        self.qkv=nn.Linear(dim,3*dim,bias=False); self.out=nn.Linear(dim,dim,bias=False)
        nn.init.normal_(self.qkv.weight,std=0.02); nn.init.normal_(self.out.weight,std=0.02)
        freqs=1.0/(10000.0**(torch.arange(0,rd,2).float()/rd))
        f=torch.outer(torch.arange(seq_len).float(),freqs)
        self.register_buffer('cos',f.cos()[None,None],persistent=False)
        self.register_buffer('sin',f.sin()[None,None],persistent=False); self.rd=rd
    def forward(self, x):
        B,T,C=x.shape; qkv=self.qkv(x).reshape(B,T,3,self.nh,self.hd)
        q,k,v=qkv.unbind(2); q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        rd=self.rd; c=self.cos[:,:,:T]; s=self.sin[:,:,:T]
        def rope(t):
            r,p=t[...,:rd],t[...,rd:]; r1,r2=r[...,:rd//2],r[...,rd//2:]
            return torch.cat([torch.cat([r1*c-r2*s,r2*c+r1*s],-1),p],-1)
        q,k=rope(q),rope(k)
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,C))

class Block(nn.Module):
    def __init__(self, dim, nh, mlp, seq_len=512):
        super().__init__()
        self.ln1=RMSNorm(dim); self.attn=Attn(dim,nh,seq_len)
        self.ln2=RMSNorm(dim); self.mlp=mlp
    def forward(self, x):
        x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x

class StandardMLP(nn.Module):
    def __init__(self, dim, exp=2.0):
        super().__init__()
        h=int(dim*exp)
        self.gate=nn.Linear(dim,h,bias=False); self.up=nn.Linear(dim,h,bias=False); self.down=nn.Linear(h,dim,bias=False)
        for m in [self.gate,self.up,self.down]: nn.init.normal_(m.weight,std=0.02)
    def forward(self, x): return self.down(F.gelu(self.gate(x))*self.up(x))

class FreezeMLP(nn.Module):
    def __init__(self, dim, exp=2.0, seed=0):
        super().__init__()
        h=int(dim*exp)
        self.gate=FrozenLinear(dim,h,seed*10+3); self.up=FrozenLinear(dim,h,seed*10+4)
        self.down=nn.Linear(h,dim,bias=False); nn.init.normal_(self.down.weight,std=0.02)
    def forward(self, x): return self.down(F.gelu(self.gate(x))*self.up(x))

class LM(nn.Module):
    def __init__(self, dim, blocks, vocab_size=1024):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,dim); self.blocks=nn.ModuleList(blocks); self.ln_f=RMSNorm(dim)
        nn.init.normal_(self.tok_emb.weight,std=0.02); self.vocab_size=vocab_size
    def forward(self, idx):
        x=self.tok_emb(idx)
        for b in self.blocks: x=b(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

def eval_ce(model, data, vocab_size=1024, n=200):
    model.eval()
    with torch.no_grad():
        eb=data[:n].to(DEVICE)
        return F.cross_entropy(model(eb[:,:-1]).reshape(-1,vocab_size),eb[:,1:].reshape(-1)).item()

def train_model(model, train_seq, steps, lr=3e-4, wd=0.1, vocab_size=1024, batch_size=32, trainable=None):
    """Generic training loop."""
    if trainable is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(steps):
        bi = torch.randint(0, train_seq.size(0), (batch_size,))
        batch = train_seq[bi].to(DEVICE)
        loss = F.cross_entropy(model(batch[:,:-1]).reshape(-1,vocab_size), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step()
    return model

def run_gutenberg_tests():
    """Run distillation + freeze tests on Gutenberg data."""
    print(f"\n{'='*60}")
    print(f"PART 1: Gutenberg-scale tests")
    print(f"{'='*60}")

    train_seq, eval_seq = load_gutenberg()
    print(f"Train: {train_seq.shape}, Eval: {eval_seq.shape}")
    results = {}

    # --- Test 1: Baseline 6L 192d ---
    print(f"\n--- Baseline: 6L 192d (3000 steps) ---")
    torch.manual_seed(42)
    baseline = LM(192, [Block(192, 6, StandardMLP(192)) for _ in range(6)]).to(DEVICE)
    t0 = time.time()
    train_model(baseline, train_seq, 3000)
    ce = eval_ce(baseline, eval_seq)
    print(f"  CE: {ce:.4f} ({time.time()-t0:.0f}s)")
    results["baseline"] = ce

    # --- Test 2: Direct freeze 8L 256d (3000 steps, no teacher) ---
    print(f"\n--- Direct freeze: 8L 256d (3000 steps) ---")
    torch.manual_seed(42)
    direct = LM(256, [Block(256, 4, FreezeMLP(256, seed=42+i)) for i in range(8)]).to(DEVICE)
    t0 = time.time()
    train_model(direct, train_seq, 3000, trainable=[p for p in direct.parameters() if p.requires_grad])
    ce = eval_ce(direct, eval_seq)
    learned = sum(p.numel() for p in direct.parameters() if p.requires_grad)
    print(f"  CE: {ce:.4f} Learned: {learned:,} ({time.time()-t0:.0f}s)")
    results["direct_freeze"] = ce
    del direct

    # --- Test 3: Distill teacher→freeze student (1500+1500) ---
    print(f"\n--- Distill: Teacher 6L 192d (1500) → Student 8L 256d freeze (1500) ---")
    torch.manual_seed(42)
    teacher = LM(192, [Block(192, 6, StandardMLP(192)) for _ in range(6)]).to(DEVICE)
    t0 = time.time()
    train_model(teacher, train_seq, 1500)
    ce_teacher = eval_ce(teacher, eval_seq)
    print(f"  Teacher CE: {ce_teacher:.4f}")
    teacher.eval()

    torch.manual_seed(123)
    student = LM(256, [Block(256, 4, FreezeMLP(256, seed=42+i)) for i in range(8)]).to(DEVICE)
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    student.train(); T_temp=2.0; alpha=0.5
    for step in range(1500):
        bi = torch.randint(0, train_seq.size(0), (BATCH_SMALL,))
        batch = train_seq[bi].to(DEVICE)
        x, y = batch[:,:-1], batch[:,1:]
        sl = student(x)
        with torch.no_grad(): tl = teacher(x)  # Both take same token indices
        hard = F.cross_entropy(sl.reshape(-1,VOCAB_SIZE_SMALL), y.reshape(-1))
        soft = F.kl_div(F.log_softmax(sl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE_SMALL),
                       F.softmax(tl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE_SMALL),
                       reduction='batchmean') * T_temp**2
        loss = alpha*hard + (1-alpha)*soft
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sch.step()
    ce = eval_ce(student, eval_seq)
    print(f"  Distilled student CE: {ce:.4f} ({time.time()-t0:.0f}s)")
    results["distill_freeze"] = ce
    del teacher, student

    # --- Test 4: Progressive freeze (train 1000 fully → freeze gate+up → train 2000 more) ---
    print(f"\n--- Progressive freeze: 8L 256d (1000 full + 2000 frozen) ---")
    torch.manual_seed(42)
    pf = LM(256, [Block(256, 4, StandardMLP(256)) for _ in range(8)]).to(DEVICE)
    t0 = time.time()
    train_model(pf, train_seq, 1000)
    ce_p1 = eval_ce(pf, eval_seq)
    print(f"  After 1000 full steps: CE={ce_p1:.4f}")
    # Freeze gate+up
    for block in pf.blocks:
        block.mlp.gate.weight.requires_grad = False
        block.mlp.up.weight.requires_grad = False
    trainable = [p for p in pf.parameters() if p.requires_grad]
    train_model(pf, train_seq, 2000, lr=3e-4, trainable=trainable)
    ce = eval_ce(pf, eval_seq)
    frozen = sum(p.numel() for p in pf.parameters() if not p.requires_grad)
    print(f"  Progressive freeze CE: {ce:.4f} Frozen: {frozen:,} ({time.time()-t0:.0f}s)")
    results["progressive_freeze"] = ce
    del pf

    # --- Test 5: Progressive freeze + self-distillation combo ---
    print(f"\n--- Progressive + self-distill: 8L 256d (1000 train + 1000 self-distill + 1000 frozen) ---")
    import copy
    torch.manual_seed(42)
    model = LM(256, [Block(256, 4, StandardMLP(256)) for _ in range(8)]).to(DEVICE)
    t0 = time.time()
    train_model(model, train_seq, 1000)
    ce_p1 = eval_ce(model, eval_seq)
    print(f"  After 1000 full steps: CE={ce_p1:.4f}")

    # Self-distill phase
    teacher_sd = copy.deepcopy(model); teacher_sd.eval()
    trainable = list(model.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    model.train(); T_temp=2.0; alpha=0.5
    for step in range(1000):
        bi = torch.randint(0, train_seq.size(0), (BATCH_SMALL,))
        batch = train_seq[bi].to(DEVICE)
        x, y = batch[:,:-1], batch[:,1:]
        sl = model(x)
        with torch.no_grad(): tl = teacher_sd(x)
        hard = F.cross_entropy(sl.reshape(-1,VOCAB_SIZE_SMALL), y.reshape(-1))
        soft = F.kl_div(F.log_softmax(sl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE_SMALL),
                       F.softmax(tl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE_SMALL),
                       reduction='batchmean') * T_temp**2
        loss = alpha*hard + (1-alpha)*soft
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sch.step()
    ce_p2 = eval_ce(model, eval_seq)
    print(f"  After self-distill: CE={ce_p2:.4f}")
    del teacher_sd

    # Freeze gate+up, continue
    for block in model.blocks:
        block.mlp.gate.weight.requires_grad = False
        block.mlp.up.weight.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.1)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    model.train()
    for step in range(1000):
        bi = torch.randint(0, train_seq.size(0), (BATCH_SMALL,))
        batch = train_seq[bi].to(DEVICE)
        loss = F.cross_entropy(model(batch[:,:-1]).reshape(-1,VOCAB_SIZE_SMALL), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sch2.step()
    ce = eval_ce(model, eval_seq)
    print(f"  Progressive+distill CE: {ce:.4f} ({time.time()-t0:.0f}s)")
    results["progressive_distill"] = ce
    del model

    # Summary
    print(f"\n{'='*60}")
    print(f"PART 1 SUMMARY (Gutenberg)")
    print(f"{'='*60}")
    for k, v in sorted(results.items()):
        delta = (v - results['baseline'])/results['baseline']*100
        print(f"  {k:30s}: CE={v:.4f} ({delta:+.1f}%)")

    return results

# ============================================================
# PART 2: FineWeb sp1024 scale validation
# ============================================================

def load_fineweb_sp1024(data_dir="data/datasets/fineweb10B_sp1024", max_shards=20):
    """Load FineWeb sp1024 data from shards."""
    shard_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not shard_files:
        print(f"  No sp1024 shards found in {data_dir}")
        return None, None

    all_tokens = []
    for sf in shard_files[:max_shards]:
        with open(sf, 'rb') as f:
            header = struct.unpack('<3i', f.read(12))
            tokens = np.fromfile(f, dtype=np.uint16)
            all_tokens.append(torch.from_numpy(tokens.astype(np.int64)))
    tokens = torch.cat(all_tokens)
    print(f"  Loaded {len(tokens):,} tokens from {len(shard_files[:max_shards])} shards")

    seq_len = 513  # 512 + 1
    n = len(tokens) // seq_len
    seqs = tokens[:n*seq_len].view(n, seq_len)
    nt = int(n * 0.9)
    return seqs[:nt], seqs[nt:]

def load_fineweb_sp4096(data_dir="data/datasets/fineweb10B_sp4096", max_shards=20):
    """Load FineWeb sp4096 data from shards."""
    shard_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not shard_files:
        print(f"  No sp4096 shards found in {data_dir}")
        return None, None

    all_tokens = []
    for sf in shard_files[:max_shards]:
        with open(sf, 'rb') as f:
            header = struct.unpack('<3i', f.read(12))
            tokens = np.fromfile(f, dtype=np.uint16)
            all_tokens.append(torch.from_numpy(tokens.astype(np.int64)))
    tokens = torch.cat(all_tokens)
    print(f"  Loaded {len(tokens):,} tokens from {len(shard_files[:max_shards])} shards")

    seq_len = 513
    n = len(tokens) // seq_len
    seqs = tokens[:n*seq_len].view(n, seq_len)
    nt = int(n * 0.9)
    return seqs[:nt], seqs[nt:]

def run_fineweb_progressive_freeze(vocab_size=1024, data_loader=None, data_dir=None, label="sp1024"):
    """Test progressive freeze on FineWeb data."""
    print(f"\n{'='*60}")
    print(f"PART 2: Progressive freeze on FineWeb {label}")
    print(f"{'='*60}")

    if data_loader:
        train_seq, eval_seq = data_loader(data_dir)
    else:
        return None

    if train_seq is None:
        print(f"  Skipping — no data")
        return None

    print(f"  Train: {train_seq.shape}, Eval: {eval_seq.shape}")
    results = {}
    batch_size = 64 if DEVICE == "cuda" else 32
    steps = 3000

    # --- Baseline: 6L 192d fully trained ---
    print(f"\n--- Baseline: 6L 192d ({steps} steps) ---")
    torch.manual_seed(42)
    model = LM(192, [Block(192, 6, StandardMLP(192)) for _ in range(6)], vocab_size=vocab_size).to(DEVICE)
    t0 = time.time()
    train_model(model, train_seq, steps, vocab_size=vocab_size, batch_size=batch_size)
    ce = eval_ce(model, eval_seq, vocab_size=vocab_size)
    learned = sum(p.numel() for p in model.parameters())
    print(f"  CE: {ce:.4f} Params: {learned:,} ({time.time()-t0:.0f}s)")
    results[f"{label}_baseline_6L"] = ce
    del model

    # --- Progressive freeze: 8L 256d ---
    print(f"\n--- Progressive freeze: 8L 256d ({steps} steps: 1000 full + 2000 frozen) ---")
    torch.manual_seed(42)
    model = LM(256, [Block(256, 4, StandardMLP(256)) for _ in range(8)], vocab_size=vocab_size).to(DEVICE)
    t0 = time.time()
    train_model(model, train_seq, 1000, vocab_size=vocab_size, batch_size=batch_size)
    ce_p1 = eval_ce(model, eval_seq, vocab_size=vocab_size)
    print(f"  After 1000 full: CE={ce_p1:.4f}")
    for block in model.blocks:
        block.mlp.gate.weight.requires_grad = False
        block.mlp.up.weight.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    train_model(model, train_seq, 2000, lr=3e-4, vocab_size=vocab_size, batch_size=batch_size, trainable=trainable)
    ce = eval_ce(model, eval_seq, vocab_size=vocab_size)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    learned = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Progressive freeze CE: {ce:.4f} Learned: {learned:,} Frozen: {frozen:,} ({time.time()-t0:.0f}s)")
    results[f"{label}_progressive_8L"] = ce
    del model

    # --- Selective freeze (random init): 8L 256d ---
    print(f"\n--- Selective freeze (random): 8L 256d ({steps} steps) ---")
    torch.manual_seed(42)
    model = LM(256, [Block(256, 4, FreezeMLP(256, seed=42+i)) for i in range(8)], vocab_size=vocab_size).to(DEVICE)
    t0 = time.time()
    trainable = [p for p in model.parameters() if p.requires_grad]
    train_model(model, train_seq, steps, vocab_size=vocab_size, batch_size=batch_size, trainable=trainable)
    ce = eval_ce(model, eval_seq, vocab_size=vocab_size)
    learned = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Selective freeze CE: {ce:.4f} Learned: {learned:,} ({time.time()-t0:.0f}s)")
    results[f"{label}_selective_8L"] = ce
    del model

    # --- Progressive freeze: 12L 384d (our best architecture) ---
    print(f"\n--- Progressive freeze: 12L 384d ({steps} steps: 1000 full + 2000 frozen) ---")
    torch.manual_seed(42)
    model = LM(384, [Block(384, 6, StandardMLP(384)) for _ in range(12)], vocab_size=vocab_size).to(DEVICE)
    t0 = time.time()
    train_model(model, train_seq, 1000, vocab_size=vocab_size, batch_size=batch_size)
    ce_p1 = eval_ce(model, eval_seq, vocab_size=vocab_size)
    print(f"  After 1000 full: CE={ce_p1:.4f}")
    for block in model.blocks:
        block.mlp.gate.weight.requires_grad = False
        block.mlp.up.weight.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    train_model(model, train_seq, 2000, lr=3e-4, vocab_size=vocab_size, batch_size=batch_size, trainable=trainable)
    ce = eval_ce(model, eval_seq, vocab_size=vocab_size)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    learned = sum(p.numel() for p in model.parameters() if p.requires_grad)
    artifact_est = learned * 0.75 / 1024 / 1024  # int6 estimate
    print(f"  Progressive freeze 12L CE: {ce:.4f} Learned: {learned:,} Frozen: {frozen:,} Artifact~{artifact_est:.1f}MB ({time.time()-t0:.0f}s)")
    results[f"{label}_progressive_12L"] = ce
    del model

    # Summary
    print(f"\n{'='*60}")
    print(f"PART 2 SUMMARY (FineWeb {label})")
    print(f"{'='*60}")
    baseline_key = f"{label}_baseline_6L"
    for k, v in sorted(results.items()):
        delta = (v - results[baseline_key])/results[baseline_key]*100
        print(f"  {k:35s}: CE={v:.4f} ({delta:+.1f}%)")

    return results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_results = {}
    t_start = time.time()

    # Part 1: Gutenberg
    r1 = run_gutenberg_tests()
    all_results.update(r1)

    # Part 2a: FineWeb sp1024
    sp1024_dirs = [
        "data/datasets/fineweb10B_sp1024",
        "/workspace/data/datasets/fineweb10B_sp1024",
    ]
    sp1024_dir = None
    for d in sp1024_dirs:
        if os.path.isdir(d):
            sp1024_dir = d; break
    if sp1024_dir:
        r2 = run_fineweb_progressive_freeze(
            vocab_size=1024, data_loader=load_fineweb_sp1024,
            data_dir=sp1024_dir, label="sp1024"
        )
        if r2: all_results.update(r2)
    else:
        print("\n  No sp1024 data found, skipping")

    # Part 2b: FineWeb sp4096
    sp4096_dirs = [
        "data/datasets/fineweb10B_sp4096",
        "/workspace/data/datasets/fineweb10B_sp4096",
    ]
    sp4096_dir = None
    for d in sp4096_dirs:
        if os.path.isdir(d):
            sp4096_dir = d; break
    if sp4096_dir:
        r3 = run_fineweb_progressive_freeze(
            vocab_size=4096, data_loader=load_fineweb_sp4096,
            data_dir=sp4096_dir, label="sp4096"
        )
        if r3: all_results.update(r3)
    else:
        print("\n  No sp4096 data found, skipping")

    # Save everything
    print(f"\n{'='*60}")
    print(f"ALL RESULTS")
    print(f"{'='*60}")
    for k, v in sorted(all_results.items()):
        print(f"  {k:35s}: CE={v:.4f}")
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

    with open("results_a40_apr4.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to results_a40_apr4.json")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
