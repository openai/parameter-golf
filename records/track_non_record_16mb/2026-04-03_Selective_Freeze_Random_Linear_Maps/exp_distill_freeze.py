#!/usr/bin/env python3
"""
Self-Distillation + Selective Freeze
======================================
Train fully-learned teacher → distill to larger selective-freeze student.
Student has MORE effective params but SMALLER artifact.

IDEA:
  Phase 1: Train 6L 192d teacher fully (1500 steps)
  Phase 2: Distill to 8L 256d freeze student (1500 steps)
    - Student frozen gate+up MLPs provide regularization
    - Teacher soft targets guide learned attention + down projection
    - Student has 2× effective params in same artifact budget
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import math, time, json, os

VOCAB_SIZE = 1024; SEQ_LEN = 512
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
print(f"Device: {DEVICE}")
print(f"Distillation + Selective Freeze — {time.strftime('%H:%M:%S')}")

def load_data():
    for cache in ["text_corpus.txt", "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"]:
        if os.path.exists(cache):
            with open(cache, 'r', errors='ignore') as f: text = f.read()
            tokens = [b % VOCAB_SIZE for b in text.encode('utf-8')]
            n = len(tokens) // (SEQ_LEN + 1)
            seqs = torch.tensor(tokens[:n*(SEQ_LEN+1)], dtype=torch.long).view(n, SEQ_LEN+1)
            nt = int(n * 0.9)
            return seqs[:nt], seqs[nt:]
    raise FileNotFoundError("No data")

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
    def __init__(self, dim, nh=6):
        super().__init__()
        self.nh=nh; self.hd=dim//nh; rd=16
        self.qkv=nn.Linear(dim,3*dim,bias=False); self.out=nn.Linear(dim,dim,bias=False)
        nn.init.normal_(self.qkv.weight,std=0.02); nn.init.normal_(self.out.weight,std=0.02)
        freqs=1.0/(10000.0**(torch.arange(0,rd,2).float()/rd))
        f=torch.outer(torch.arange(SEQ_LEN).float(),freqs)
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
    def __init__(self, dim, nh, mlp):
        super().__init__()
        self.ln1=RMSNorm(dim); self.attn=Attn(dim,nh)
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
    def __init__(self, dim, blocks):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB_SIZE,dim); self.blocks=nn.ModuleList(blocks); self.ln_f=RMSNorm(dim)
        nn.init.normal_(self.tok_emb.weight,std=0.02)
    def forward(self, idx):
        x=self.tok_emb(idx)
        for b in self.blocks: x=b(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

def eval_ce(model, data, n=200):
    model.eval()
    with torch.no_grad():
        eb=data[:n].to(DEVICE)
        return F.cross_entropy(model(eb[:,:-1]).reshape(-1,VOCAB_SIZE),eb[:,1:].reshape(-1)).item()

if __name__ == "__main__":
    train_seq, eval_seq = load_data()
    print(f"Train: {train_seq.shape}, Eval: {eval_seq.shape}")
    results = {}

    # === Baseline: 6L 192d direct training 3000 steps ===
    print(f"\n{'='*50}\nBaseline: 6L 192d direct (3000 steps)\n{'='*50}")
    torch.manual_seed(42)
    baseline = LM(192, [Block(192, 6, StandardMLP(192)) for _ in range(6)]).to(DEVICE)
    opt = torch.optim.AdamW(baseline.parameters(), lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
    baseline.train(); t0=time.time()
    for step in range(3000):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(baseline(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.0); opt.step(); sch.step()
    ce_baseline = eval_ce(baseline, eval_seq)
    print(f"  Baseline CE: {ce_baseline:.4f} ({time.time()-t0:.0f}s)")
    results["baseline"] = ce_baseline

    # === Direct freeze student (no distillation) ===
    print(f"\n{'='*50}\nDirect: 8L 256d freeze (3000 steps, no teacher)\n{'='*50}")
    torch.manual_seed(42)
    direct = LM(256, [Block(256, 4, FreezeMLP(256, seed=42+i)) for i in range(8)]).to(DEVICE)
    opt = torch.optim.AdamW([p for p in direct.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
    direct.train(); t0=time.time()
    for step in range(3000):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(direct(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in direct.parameters() if p.requires_grad],1.0); opt.step(); sch.step()
    ce_direct = eval_ce(direct, eval_seq)
    learned_d = sum(p.numel() for p in direct.parameters() if p.requires_grad)
    print(f"  Direct CE: {ce_direct:.4f} Learned: {learned_d:,} ({time.time()-t0:.0f}s)")
    results["direct_freeze"] = ce_direct
    del direct

    # === Distilled freeze student ===
    print(f"\n{'='*50}\nDistill: Teacher 6L 192d (1500) → Student 8L 256d freeze (1500)\n{'='*50}")
    # Phase 1: Train teacher
    torch.manual_seed(42)
    teacher = LM(192, [Block(192, 6, StandardMLP(192)) for _ in range(6)]).to(DEVICE)
    opt = torch.optim.AdamW(teacher.parameters(), lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    teacher.train(); t0=time.time()
    for step in range(1500):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(teacher(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step(); sch.step()
    ce_teacher = eval_ce(teacher, eval_seq)
    print(f"  Teacher CE: {ce_teacher:.4f} ({time.time()-t0:.0f}s)")
    teacher.eval()

    # Phase 2: Distill to freeze student
    torch.manual_seed(123)
    student = LM(256, [Block(256, 4, FreezeMLP(256, seed=42+i)) for i in range(8)]).to(DEVICE)
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    student.train(); T_temp = 2.0; alpha = 0.5
    for step in range(1500):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        x,y = batch[:,:-1], batch[:,1:]
        sl = student(x)
        # Both teacher and student take token indices → same input, both output (B,T,VOCAB_SIZE)
        with torch.no_grad(): tl = teacher(x)
        hard = F.cross_entropy(sl.reshape(-1,VOCAB_SIZE),y.reshape(-1))
        soft = F.kl_div(F.log_softmax(sl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE),
                       F.softmax(tl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE),
                       reduction='batchmean')*T_temp**2
        loss = alpha*hard + (1-alpha)*soft
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); sch.step()
    ce_distill = eval_ce(student, eval_seq)
    learned_s = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Distilled student CE: {ce_distill:.4f} Learned: {learned_s:,} ({time.time()-t0:.0f}s)")
    results["distill_freeze"] = ce_distill

    # === Summary ===
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    print(f"  Baseline 6L 192d (3000 steps):          CE={results['baseline']:.4f}")
    print(f"  Direct 8L 256d freeze (3000 steps):      CE={results['direct_freeze']:.4f} ({(results['direct_freeze']-results['baseline'])/results['baseline']*100:+.1f}%)")
    print(f"  Distilled 8L 256d freeze (1500+1500):    CE={results['distill_freeze']:.4f} ({(results['distill_freeze']-results['baseline'])/results['baseline']*100:+.1f}%)")
    print(f"\n  Does distillation help freeze student?")
    if results['distill_freeze'] < results['direct_freeze']:
        print(f"  YES: distilled is {(results['direct_freeze']-results['distill_freeze'])/results['direct_freeze']*100:.1f}% better")
    else:
        print(f"  NO: direct training is better")

    # === Progressive freeze + self-distillation ===
    # Best combo: train fully 1000 steps, self-distill 1000 steps, freeze+continue 1000 steps
    print(f"\n{'='*50}\nProgressive freeze + self-distill (1000+1000+1000)\n{'='*50}")
    torch.manual_seed(42)
    model_pd = LM(256, [Block(256, 4, StandardMLP(256)) for _ in range(8)]).to(DEVICE)
    opt = torch.optim.AdamW(model_pd.parameters(), lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
    model_pd.train(); t0=time.time()
    # Phase 1: Train fully for 1000 steps
    for step in range(1000):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(model_pd(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model_pd.parameters(),1.0); opt.step(); sch.step()
    ce_p1 = eval_ce(model_pd, eval_seq)
    print(f"  After phase 1 (1000 full): CE={ce_p1:.4f}")

    # Phase 2: Self-distill — use current model as teacher, reset trainable params
    # Save teacher state
    import copy
    teacher_pd = copy.deepcopy(model_pd)
    teacher_pd.eval()
    # Continue training student with distillation from self
    model_pd.train(); T_temp=2.0; alpha=0.5
    for step in range(1000):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        x,y = batch[:,:-1], batch[:,1:]
        sl = model_pd(x)
        with torch.no_grad(): tl = teacher_pd(x)
        hard = F.cross_entropy(sl.reshape(-1,VOCAB_SIZE),y.reshape(-1))
        soft = F.kl_div(F.log_softmax(sl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE),
                       F.softmax(tl/T_temp,dim=-1).reshape(-1,VOCAB_SIZE),
                       reduction='batchmean')*T_temp**2
        loss = alpha*hard + (1-alpha)*soft
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model_pd.parameters(),1.0); opt.step(); sch.step()
    ce_p2 = eval_ce(model_pd, eval_seq)
    print(f"  After phase 2 (1000 self-distill): CE={ce_p2:.4f}")
    del teacher_pd

    # Phase 3: Freeze gate+up, continue training (progressive freeze)
    frozen_count = 0
    for block in model_pd.blocks:
        mlp = block.mlp
        mlp.gate.weight.requires_grad = False
        mlp.up.weight.requires_grad = False
        frozen_count += mlp.gate.weight.numel() + mlp.up.weight.numel()
    trainable_pd = [p for p in model_pd.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_pd, lr=1e-4, weight_decay=0.1)  # lower LR for fine-tuning
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    model_pd.train()
    for step in range(1000):
        bi=torch.randint(0,train_seq.size(0),(BATCH_SIZE,)); batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(model_pd(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable_pd,1.0); opt.step(); sch.step()
    ce_pd = eval_ce(model_pd, eval_seq)
    learned_pd = sum(p.numel() for p in model_pd.parameters() if p.requires_grad)
    print(f"  Progressive+distill CE: {ce_pd:.4f} Learned: {learned_pd:,} Frozen: {frozen_count:,} ({time.time()-t0:.0f}s)")
    results["progressive_distill"] = ce_pd
    del model_pd

    # === Summary ===
    print(f"\n{'='*50}\nFINAL SUMMARY\n{'='*50}")
    for k,v in sorted(results.items()):
        delta = (v - results['baseline'])/results['baseline']*100
        print(f"  {k:30s}: CE={v:.4f} ({delta:+.1f}%)")

    with open("results_distill_freeze.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFinished: {time.strftime('%H:%M:%S')}")
