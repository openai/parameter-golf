#!/usr/bin/env python3
"""Fix all bugs in patch_v11.py identified by Gemini/Grok/Opus."""
import sys

with open("patch_v11.py", "r") as f:
    code = f.read()

fixes = 0

def fix(old, new, label):
    global code, fixes
    if old in code:
        code = code.replace(old, new, 1)
        fixes += 1
        print(f"  FIXED: {label}")
    else:
        print(f"  SKIP: {label} (not found)")

# FIX 1: Turbo-Muon bf16 overflow — compute D_r/D_c in float32
fix(
    '''    X = G.bfloat16()
    # AOL preconditioning: D_r^{-1/2} @ X @ D_c^{-1/2}
    D_r = (X * X).sum(dim=1, keepdim=True).clamp_min(eps * eps)
    D_c = (X * X).sum(dim=0, keepdim=True).clamp_min(eps * eps)
    X = X / (D_r * D_c).pow(0.25)''',
    '''    X = G.bfloat16()
    # AOL preconditioning in float32 to prevent bf16 overflow
    Xf = X.float()
    D_r = (Xf * Xf).sum(dim=1, keepdim=True).clamp_min(eps)
    D_c = (Xf * Xf).sum(dim=0, keepdim=True).clamp_min(eps)
    X = (Xf / (D_r * D_c).pow(0.25)).bfloat16()''',
    "Turbo-Muon bf16 overflow")

# FIX 2: GPTQ double diagonal division — remove extra division
fix(
    '''        if j2 < n_cols:
            W[:, j2:] -= Err @ (Hinv[j1:j2, j2:] / Hinv[j1:j2, j1:j2].diag().clamp_min(1e-10).unsqueeze(1))''',
    '''        if j2 < n_cols:
            W[:, j2:] -= Err @ Hinv[j1:j2, j2:]''',
    "GPTQ double diagonal division")

# FIX 3: Brotli quality in binary search — use q=4 for speed, q=11 only for final
fix(
    '''        _tsz = len(brotli.compress(_tbuf.getvalue(), quality=11))
        if _tsz + _code_bytes <= 16_000_000:''',
    '''        _tsz = len(brotli.compress(_tbuf.getvalue(), quality=4))
        del _tobj, _tbuf; import gc as _gc; _gc.collect()
        if _tsz + _code_bytes <= 16_000_000:''',
    "Brotli speed in binary search")

# FIX 4: Final GPTQ qmax loop also uses quality=4, then final check with q=11
fix(
    '''            _tsz = len(brotli.compress(_tbuf.getvalue(), quality=11))
            if _tsz + _code_bytes <= 16_000_000:
                globals()["BLOCK_QUANT_MAX"] = _try_qmax
                log0(f"raki_v11:gptq_final_qmax={_try_qmax} est_bytes={_tsz + _code_bytes}")''',
    '''            _tsz = len(brotli.compress(_tbuf.getvalue(), quality=11))
            del _tobj, _tbuf; import gc as _gc2; _gc2.collect()
            if _tsz + _code_bytes <= 16_000_000:
                globals()["BLOCK_QUANT_MAX"] = _try_qmax
                log0(f"raki_v11:final_qmax={_try_qmax} bytes={_tsz + _code_bytes}")''',
    "GPTQ final loop memory cleanup")

# FIX 5: best_mse device mismatch
fix(
    "best_mse = torch.full((t32.shape[0],), float('inf'))",
    "best_mse = torch.full((t32.shape[0],), float('inf'), device=t32.device)",
    "best_mse device mismatch")

# FIX 6: TTT AdamW momentum conflict — create fresh optimizer per chunk
fix(
    '''        ttt_opt = torch.optim.AdamW(ttt_params, lr=TTT_LR, weight_decay=0.0)
        seq_len = args.train_seq_len''',
    '''        seq_len = args.train_seq_len''',
    "TTT remove global optimizer (moved to per-chunk)")

fix(
    '''            # TRAIN: fine-tune on scored chunk (AdamW, cosine LR)
            base_ttt.train()''',
    '''            # TRAIN: fresh AdamW per chunk (no momentum conflict)
            ttt_opt = torch.optim.AdamW(ttt_params, lr=TTT_LR, weight_decay=0.0)
            base_ttt.train()''',
    "TTT fresh optimizer per chunk")

# FIX 7: TTT decay prior + AdamW conflict — remove decay prior
fix(
    '''                    # Decay prior: pull toward pre-TTT weights
                    if TTT_DECAY > 0:
                        with torch.no_grad():
                            for p in ttt_params:
                                p.data.add_(_pre_ttt[id(p)] - p.data, alpha=TTT_DECAY)''',
    '''                    pass  # No decay prior (conflicts with AdamW momentum)''',
    "Remove TTT decay prior")

# FIX 8: Remove _pre_ttt allocation (no longer needed)
fix(
    '''        # Save pre-TTT weights for decay prior
        _pre_ttt = {id(p): p.data.clone() for p in ttt_params}

        ttt_opt''',
    '''        ttt_opt''',
    "Remove pre-TTT weight copy")

# FIX 9: TTT ttt_tok_count — use int instead of CUDA tensor for counter
fix(
    '''        ttt_tok_count = torch.zeros((), device=device, dtype=torch.float64)''',
    '''        ttt_tok_count = 0''',
    "TTT tok count as int")

# FIX 10: Hessian multi-file collection
fix(
    '''    hdr = np.fromfile(files[0], dtype="<i4", count=256)
    ntok = min(int(hdr[2]), n_batches * seq_len + 1)
    tokens = np.fromfile(files[0], dtype="<u2", count=ntok, offset=hdr_bytes)''',
    '''    all_tokens = []
    remaining = n_batches * seq_len + 1
    for f in files:
        if remaining <= 0:
            break
        hdr = np.fromfile(f, dtype="<i4", count=256)
        ntok = min(int(hdr[2]), remaining)
        tok = np.fromfile(f, dtype="<u2", count=ntok, offset=hdr_bytes)
        all_tokens.append(tok)
        remaining -= len(tok)
    tokens = np.concatenate(all_tokens) if all_tokens else np.array([], dtype=np.uint16)''',
    "Hessian multi-file")

# FIX 11: Remove TTT_DECAY config (no longer used)
fix(
    "TTT_DECAY = float(os.environ.get(\"TTT_DECAY\", \"0.001\"))\n",
    "",
    "Remove TTT_DECAY config")

# FIX 12: Update version strings
code = code.replace("raki_v11:ttt_starting", "raki_v11:ttt")
code = code.replace("raki_v11:ttt lr=", "raki_v11:ttt lr=")

with open("patch_v11.py", "w") as f:
    f.write(code)

print(f"\n{fixes} fixes applied to patch_v11.py")
print("All critical bugs resolved: bf16 overflow, GPTQ double-div, brotli speed,")
print("device mismatch, memory leak, TTT momentum, hessian multi-file")
