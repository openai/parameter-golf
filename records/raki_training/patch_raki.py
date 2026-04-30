#!/usr/bin/env python3
"""
Patches baseline train_gpt.py IN-PLACE to add Raki techniques.
Uses exact string matching — no guessing.

Run on RunPod:
  python3 patch_raki.py
  RUN_ID=raki NUM_LAYERS=10 WARMDOWN_ITERS=3500 MUON_WD=0.04 torchrun --standalone --nproc_per_node=1 train_gpt.py
"""
import sys

f = open("train_gpt.py", "r")
code = f.read()
f.close()

changes = 0

# ================================================================
# PATCH 1: Add Markov + EMA classes after DDP import
# ================================================================
ANCHOR1 = "from torch.nn.parallel import DistributedDataParallel as DDP"
INSERT1 = '''from torch.nn.parallel import DistributedDataParallel as DDP

# ---- RAKI: Markov curriculum + EMA ----
RAKI_POWER = float(os.environ.get("RAKI_POWER", "0.15"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.995"))
EMA_START_FRAC = float(os.environ.get("EMA_START_FRAC", "0.85"))
BH_EVAL_WEIGHT = float(os.environ.get("BH_EVAL_WEIGHT", "0.3"))

class _MarkovTable:
    def __init__(self, pattern, V, device):
        files = sorted(glob.glob(pattern))
        hdr_bytes = 256 * np.dtype("<i4").itemsize
        hdr = np.fromfile(files[0], dtype="<i4", count=256)
        tok = np.fromfile(files[0], dtype="<u2", count=int(hdr[2]),
                          offset=hdr_bytes).astype(np.int32)
        counts = np.zeros((V, V), dtype=np.float64)
        np.add.at(counts, (tok[:-1], tok[1:]), 1.0)
        sm = counts + 0.01
        probs = sm / sm.sum(axis=1, keepdims=True)
        lp = np.log(probs)
        self.log_probs = lp.astype(np.float16)
        self.bias = torch.tensor(lp, dtype=torch.float32, device=device)
        ent = -(probs * lp).sum(axis=1).astype(np.float32)
        mn, mx = ent.min(), ent.max()
        self.ent_n = (ent - mn) / (mx - mn) if mx > mn else np.full_like(ent, 0.5)
    def batch_weight(self, x, y):
        """Returns scalar weight >= 1.0 for this batch."""
        xn = x.cpu().numpy().astype(np.int32)
        yn = y.cpu().numpy().astype(np.int32)
        surp = -self.log_probs[xn.ravel(), yn.ravel()].astype(np.float32).reshape(xn.shape)
        ent = self.ent_n[xn.ravel()].reshape(xn.shape)
        score = float((surp * ent).mean())
        return 1.0 + RAKI_POWER * min(score / 5.0, 1.0)

class _EMA:
    def __init__(self): self.shadow = None; self.on = False
    def start(self, m):
        self.shadow = {n: p.data.clone() for n, p in m.named_parameters()}
        self.on = True
    def update(self, m):
        if not self.on: return
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in self.shadow: self.shadow[n].lerp_(p.data, 1.0 - EMA_DECAY)
    def apply(self, m):
        if not self.shadow: return
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in self.shadow: p.data.copy_(self.shadow[n])
# ---- END RAKI ----'''

if ANCHOR1 in code:
    code = code.replace(ANCHOR1, INSERT1, 1)
    changes += 1
    print(f"  PATCH 1 OK: Markov + EMA classes added")
else:
    print(f"  PATCH 1 FAIL: anchor not found"); sys.exit(1)

# ================================================================
# PATCH 2: Init Markov + EMA before training loop
# Insert just before "training_time_ms = 0.0"
# ================================================================
ANCHOR2 = "    training_time_ms = 0.0"
INSERT2 = '''    # ---- RAKI: Init ----
    _mk = _MarkovTable(args.train_files, args.vocab_size, device)
    _ema = _EMA()
    _ema_on = False
    log0(f"raki:markov_ok power={RAKI_POWER} ema_decay={EMA_DECAY} bh_eval={BH_EVAL_WEIGHT}")

    training_time_ms = 0.0'''

if ANCHOR2 in code:
    code = code.replace(ANCHOR2, INSERT2, 1)
    changes += 1
    print(f"  PATCH 2 OK: Markov + EMA init added")
else:
    print(f"  PATCH 2 FAIL: anchor not found"); sys.exit(1)

# ================================================================
# PATCH 3: Add curriculum weighting to training loss
# Modify: (loss * grad_scale).backward()
# To: (loss * grad_scale * curriculum_weight).backward()
# ================================================================
ANCHOR3 = "            (loss * grad_scale).backward()"
INSERT3 = '''            # ---- RAKI: curriculum weight (surprising batches get stronger gradients) ----
            _cw = _mk.batch_weight(x, y) if RAKI_POWER > 0 else 1.0
            (loss * grad_scale * _cw).backward()'''

if ANCHOR3 in code:
    code = code.replace(ANCHOR3, INSERT3, 1)
    changes += 1
    print(f"  PATCH 3 OK: curriculum weighting added")
else:
    print(f"  PATCH 3 FAIL: anchor not found"); sys.exit(1)

# ================================================================
# PATCH 4: EMA update after optimizer step + zero_grad
# Insert after "zero_grad_all()" in main training loop (second occurrence)
# The pattern is: "zero_grad_all()\n\n        step += 1"
# ================================================================
ANCHOR4 = "        zero_grad_all()\n\n        step += 1"
INSERT4 = '''        zero_grad_all()

        # ---- RAKI: EMA update ----
        _prog = (training_time_ms + 1000.0 * (time.perf_counter() - t0)) / max(max_wallclock_ms or 1e18, 1.0)
        if _prog >= EMA_START_FRAC and not _ema_on:
            _ema.start(base_model); _ema_on = True
            log0(f"raki:ema_started step={step+1}")
        _ema.update(base_model)

        step += 1'''

if ANCHOR4 in code:
    code = code.replace(ANCHOR4, INSERT4, 1)
    changes += 1
    print(f"  PATCH 4 OK: EMA update added")
else:
    print(f"  PATCH 4 FAIL: anchor not found"); sys.exit(1)

# ================================================================
# PATCH 5: Apply EMA before final serialization
# Insert before "if master_process:\n        torch.save(base_model.state_dict()"
# ================================================================
ANCHOR5 = '    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")'
INSERT5 = '''    # ---- RAKI: Apply EMA before saving ----
    if _ema.on:
        _ema.apply(base_model)
        log0("raki:ema_applied")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")'''

if ANCHOR5 in code:
    code = code.replace(ANCHOR5, INSERT5, 1)
    changes += 1
    print(f"  PATCH 5 OK: EMA apply before save")
else:
    print(f"  PATCH 5 FAIL: anchor not found"); sys.exit(1)

# ================================================================
# WRITE
# ================================================================
with open("train_gpt.py", "w") as f:
    f.write(code)

print(f"\n{'='*60}")
print(f" RAKI PATCH COMPLETE — {changes}/5 patches applied")
print(f"{'='*60}")
print(f" Run:")
print(f"   RUN_ID=raki NUM_LAYERS=10 WARMDOWN_ITERS=3500 MUON_WD=0.04 \\")
print(f"   torchrun --standalone --nproc_per_node=1 train_gpt.py")
print(f"{'='*60}")
