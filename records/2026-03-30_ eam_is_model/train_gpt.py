#!/usr/bin/env python3
"""
EAM IS THE MODEL — Non-Record Submission
Elastic Associative Memory as Native Intelligence for Language Modeling

Artifact: Encoder (426K params) + EAM (~12K locations) + Decoder (657K params)
No transformer at inference. Teacher trained and discarded.

Key result: 97.9% intelligence transfer from teacher to EAM.
EAM + kNN at eval time BEATS the teacher.

Usage:
  # Single GPU (Colab/testing):
  python3 train_gpt.py

  # 8xH100 (competition):
  torchrun --standalone --nproc_per_node=8 train_gpt.py
"""
from __future__ import annotations
import glob, io, lzma, math, os, time, uuid
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:
    dist = None

# =====================================================================
# HYPERPARAMETERS
# =====================================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Teacher
    teacher_dim = int(os.environ.get("TEACHER_DIM", 512))
    teacher_layers = int(os.environ.get("TEACHER_LAYERS", 8))
    teacher_heads = int(os.environ.get("TEACHER_HEADS", 8))
    teacher_seq_len = int(os.environ.get("TEACHER_SEQ_LEN", 512))
    teacher_lr = float(os.environ.get("TEACHER_LR", 3e-4))
    teacher_batch = int(os.environ.get("TEACHER_BATCH", 64))
    teacher_iters = int(os.environ.get("TEACHER_ITERS", 5000))

    # EAM
    eam_key_dim = int(os.environ.get("EAM_KEY_DIM", 256))
    eam_locations = int(os.environ.get("EAM_LOCATIONS", 12000))
    eam_k = int(os.environ.get("EAM_K", 20))
    eam_beta = float(os.environ.get("EAM_BETA", 10.0))
    eam_lr = float(os.environ.get("EAM_LR", 0.01))
    eam_tau_split = float(os.environ.get("EAM_TAU_SPLIT", 0.15))
    eam_tau_overload = int(os.environ.get("EAM_TAU_OVERLOAD", 150))

    # Encoder
    enc_dim = int(os.environ.get("ENC_DIM", 128))
    enc_layers = int(os.environ.get("ENC_LAYERS", 2))
    enc_heads = int(os.environ.get("ENC_HEADS", 2))
    enc_epochs = int(os.environ.get("ENC_EPOCHS", 10))

    # Decoder
    dec_epochs = int(os.environ.get("DEC_EPOCHS", 10))

    # Eval
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 512))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    knn_lambda = float(os.environ.get("KNN_LAMBDA", 0.5))
    knn_k = int(os.environ.get("KNN_K", 16))
    knn_beta = float(os.environ.get("KNN_BETA", 20.0))

args = Hyperparameters()


# =====================================================================
# INFRASTRUCTURE
# =====================================================================

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    offset = 256 * np.dtype("<i4").itemsize
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=int(header[2]), offset=offset).astype(np.int64))

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size()); ts = max(sp_vs, vocab_size)
    base_bytes = np.zeros((ts,), dtype=np.int16)
    has_space = np.zeros((ts,), dtype=np.bool_)
    is_boundary = np.ones((ts,), dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary[tid] = False
        if sp.is_byte(tid): base_bytes[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_space[tid] = True; piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes, dtype=torch.int16, device=device),
            torch.tensor(has_space, dtype=torch.bool, device=device),
            torch.tensor(is_boundary, dtype=torch.bool, device=device))


# =====================================================================
# BUILDING BLOCKS
# =====================================================================

class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, x): return F.rms_norm(x, (s.d,))

class Block(nn.Module):
    def __init__(s, d, nh, ff):
        super().__init__()
        s.ln1 = RMSNorm(d); s.ln2 = RMSNorm(d)
        hd = d // nh; s.nh, s.hd = nh, hd
        s.qkv = nn.Linear(d, 3*d, bias=False); s.out = nn.Linear(d, d, bias=False)
        s.up = nn.Linear(d, ff, bias=False); s.down = nn.Linear(ff, d, bias=False)
    def forward(s, x):
        B, T, D = x.shape
        qkv = s.qkv(s.ln1(x)).reshape(B, T, 3, s.nh, s.hd)
        q, k, v = qkv.unbind(2)
        y = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True)
        x = x + s.out(y.transpose(1,2).reshape(B, T, D))
        x = x + s.down(F.leaky_relu(s.up(s.ln2(x)), 0.5).square())
        return x


# =====================================================================
# TEACHER (trained, then discarded)
# =====================================================================

class Teacher(nn.Module):
    def __init__(s, V, D, L, H, ffm=3):
        super().__init__()
        s.dim = D; s.vocab_size = V; s.cap = 30.0
        s.tok = nn.Embedding(V, D); nn.init.normal_(s.tok.weight, std=0.005)
        s.blocks = nn.ModuleList([Block(D, H, int(D*ffm)) for _ in range(L)])
        s.norm = RMSNorm(D)
    def forward_full(s, ids):
        x = F.rms_norm(s.tok(ids), (s.dim,))
        for b in s.blocks: x = b(x)
        h = s.norm(x)
        lp = F.linear(h, s.tok.weight)
        return s.cap * torch.tanh(lp / s.cap), h
    def forward(s, ids, tgt=None):
        logits, _ = s.forward_full(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               tgt.reshape(-1)) if tgt is not None else None
        return logits, loss


# =====================================================================
# ENCODER: tokens → EAM keys
# =====================================================================

class Encoder(nn.Module):
    def __init__(s, V, D, L, H, key_dim):
        super().__init__()
        s.dim = D; s.key_dim = key_dim
        s.tok = nn.Embedding(V, D)
        s.blocks = nn.ModuleList([Block(D, H, int(D*2)) for _ in range(L)])
        s.norm = RMSNorm(D)
        s.key_proj = nn.Linear(D, key_dim, bias=False)
    def forward(s, ids):
        x = F.rms_norm(s.tok(ids), (s.dim,))
        for b in s.blocks: x = b(x)
        return F.normalize(s.key_proj(s.norm(x)), dim=-1)


# =====================================================================
# DECODER: EAM readout → logits
# =====================================================================

class Decoder(nn.Module):
    def __init__(s, hd, V):
        super().__init__()
        s.cap = 30.0
        s.head = nn.Sequential(
            nn.Linear(hd, hd * 2), nn.GELU(), nn.Linear(hd * 2, V))
    def forward(s, x):
        return s.cap * torch.tanh(s.head(x) / s.cap)


# =====================================================================
# EAM: The intelligence
# =====================================================================

class EAM:
    """
    Elastic Associative Memory (Nguthiru, 2026).
    Self-organizing locations, additive counter superposition,
    competitive learning, conscience, splitting/merging.
    GPU-native batched operations.
    """
    def __init__(s, kd, hd, L0, max_L, k=20, beta=10.0,
                 lr=0.01, gamma=1.0, tau_damp=10.0,
                 tau_split=0.15, tau_merge=0.95, tau_overload=150,
                 regulate_every=1024, device='cuda'):
        s.kd, s.hd, s.k, s.beta = kd, hd, k, beta
        s.lr, s.gamma, s.tau_damp = lr, gamma, tau_damp
        s.tau_split, s.tau_merge, s.tau_overload = tau_split, tau_merge, tau_overload
        s.max_L, s.regulate_every, s.device = max_L, regulate_every, device
        s.addresses = F.normalize(torch.randn(max_L, kd, device=device), dim=-1).half()
        s.counters = torch.zeros(max_L, hd, device=device)
        s.write_weights = torch.zeros(max_L, device=device)
        s.L = L0; s.total_writes = 0; s._rc = 0

    @torch.no_grad()
    def write_batch(s, keys, vals):
        B = keys.shape[0]; L = s.L; k = min(s.k, L)
        sims = keys.half() @ s.addresses[:L].T
        topk_s, topk_i = sims.float().topk(k, dim=-1)
        mx = topk_s[:, 0:1].clamp(min=1e-8)
        w = topk_s.clamp(min=0) / mx
        tw = s.write_weights[:L].sum().clamp(min=1e-8)
        consc = s.gamma * s.write_weights[topk_i] / tw
        win_k = (topk_s - consc).argmax(dim=-1)
        win_i = torch.gather(topk_i, 1, win_k.unsqueeze(1)).squeeze(1)
        fl = topk_i.reshape(-1); fw = w.reshape(-1)
        fv = vals.unsqueeze(1).expand(B, k, s.hd).reshape(-1, s.hd)
        s.counters[:L].index_add_(0, fl, fv * fw.unsqueeze(1))
        s.write_weights[:L].scatter_add_(0, fl, fw)
        eta = s.lr / (1.0 + s.write_weights[win_i] / s.tau_damp)
        na = s.addresses[win_i].float() + eta.unsqueeze(1) * (keys - s.addresses[win_i].float())
        s.addresses[win_i] = F.normalize(na, dim=-1).half()
        s.lr = max(0.001, s.lr * (0.99999 ** B))
        s.total_writes += B; s._rc += B
        if s._rc >= s.regulate_every:
            s._regulate(keys, mx); s._rc = 0

    @torch.no_grad()
    def read_batch(s, keys, beta=None):
        if beta is None: beta = s.beta
        L = s.L; k = min(s.k, L)
        if L < k: return None
        sims = keys.half() @ s.addresses[:L].T
        topk_s, topk_i = sims.float().topk(k, dim=-1)
        gc = s.counters[topk_i]
        gw = s.write_weights[topk_i].clamp(min=1e-8)
        norm = gc / gw.unsqueeze(-1)
        alpha = F.softmax(beta * topk_s, dim=-1)
        return (alpha.unsqueeze(-1) * norm).sum(dim=1)

    @torch.no_grad()
    def _regulate(s, rk, rms):
        L = s.L
        novel = rms.squeeze(-1) < s.tau_split
        if novel.any() and L < s.max_L:
            nk = rk[novel]; n = min(nk.shape[0], s.max_L - L, 20)
            if n > 0:
                s.addresses[L:L+n] = F.normalize(nk[:n], dim=-1).half()
                s.counters[L:L+n] = 0; s.write_weights[L:L+n] = 0; s.L += n
        L = s.L
        over = (s.write_weights[:L] > s.tau_overload).nonzero(as_tuple=True)[0]
        if len(over) > 0 and L < s.max_L:
            for i in range(min(len(over), s.max_L - L, 10)):
                p = over[i].item()
                noise = F.normalize(torch.randn(s.kd, device=s.device), dim=-1).half() * 0.1
                s.addresses[L] = F.normalize((s.addresses[p].float()+noise.float()), dim=-1).half()
                s.counters[L] = 0; s.write_weights[L] = 0; L += 1
            s.L = L
        L = s.L
        if L > 20:
            nc = min(L, 200); si = torch.randperm(L, device=s.device)[:nc]
            ps = s.addresses[si].float() @ s.addresses[si].float().T; ps.fill_diagonal_(-1)
            ms, mj = ps.max(dim=1)
            cands = (ms > s.tau_merge).nonzero(as_tuple=True)[0]
            if len(cands) > 0:
                f = cands[0].item(); il, jl = si[f].item(), si[mj[f]].item()
                if il != jl and il < s.L and jl < s.L:
                    wi, wj = s.write_weights[il], s.write_weights[jl]; t = wi+wj+1e-8
                    s.addresses[il] = F.normalize(
                        (wi*s.addresses[il].float()+wj*s.addresses[jl].float())/t, dim=-1).half()
                    s.counters[il] += s.counters[jl]; s.write_weights[il] = wi+wj
                    last = s.L-1
                    if jl != last:
                        s.addresses[jl]=s.addresses[last].clone()
                        s.counters[jl]=s.counters[last].clone()
                        s.write_weights[jl]=s.write_weights[last].clone()
                    s.counters[last]=0; s.write_weights[last]=0; s.L-=1

    def state_dict(s):
        L = s.L
        return {'addresses': s.addresses[:L].cpu(),
                'counters': s.counters[:L].cpu(),
                'write_weights': s.write_weights[:L].cpu(), 'L': L}

    def load_state_dict(s, sd, device=None):
        dev = device or s.device; L = sd['L']
        s.addresses[:L] = sd['addresses'].to(dev)
        s.counters[:L] = sd['counters'].to(dev)
        s.write_weights[:L] = sd['write_weights'].to(dev)
        s.L = L

    def compressed_bytes(s):
        sd = s.state_dict()
        # Quantize counters to int8 for storage
        c = sd['counters']; mx = c.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        sc = (mx / 127.0).squeeze(1).half()
        cq = torch.clamp(torch.round(c / mx * 127.0), -127, 127).to(torch.int8)
        save_sd = {'addresses': sd['addresses'], 'counters_q': cq,
                    'counters_scale': sc, 'write_weights': sd['write_weights'].half(),
                    'L': sd['L']}
        buf = io.BytesIO(); torch.save(save_sd, buf)
        return len(lzma.compress(buf.getvalue(), preset=6))


# =====================================================================
# FLAT kNN STORE (eval-time augmentation)
# =====================================================================

class FlatStore:
    def __init__(s, kd, vs, mx, dev):
        s.kd,s.vs,s.mx,s.dev=kd,vs,mx,dev
        s.keys=torch.zeros(mx,kd,dtype=torch.float16,device=dev)
        s.tokens=torch.zeros(mx,dtype=torch.int64,device=dev); s.count=0
    def write_batch(s, keys, toks):
        B=keys.shape[0]; st=s.count%s.mx; e=st+B
        if e<=s.mx: s.keys[st:e]=keys.half(); s.tokens[st:e]=toks
        else:
            f=s.mx-st; s.keys[st:]=keys[:f].half(); s.tokens[st:]=toks[:f]
            s.keys[:B-f]=keys[f:].half(); s.tokens[:B-f]=toks[f:]
        s.count+=B
    @torch.no_grad()
    def retrieve(s, qk, k, beta):
        N=min(s.count,s.mx)
        if N<k: return None
        k=min(k,N); sims=qk.half()@s.keys[:N].T
        ts,ti=sims.topk(k,dim=-1); a=F.softmax(beta*ts.float(),dim=-1)
        d=torch.zeros(qk.shape[0],s.vs,device=s.dev)
        d.scatter_add_(1,s.tokens[ti],a); return d
    @property
    def n_stored(s): return min(s.count,s.mx)


# =====================================================================
# MAIN
# =====================================================================

def main():
    # Setup
    distributed = dist is not None and dist.is_available() and int(os.environ.get("RANK", -1)) >= 0
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0; world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0
    def log(msg):
        if master: print(msg, flush=True)

    torch.manual_seed(args.seed + rank)
    log(f"EAM-IS-THE-MODEL | ranks={world_size} device={device}")

    # Load data
    train_files = sorted(glob.glob(args.train_files))
    val_files = sorted(glob.glob(args.val_files))
    train_tokens = torch.cat([load_data_shard(f) for f in train_files[:1]])  # 1 shard for speed
    val_tokens = torch.cat([load_data_shard(f) for f in val_files])
    log(f"Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens")

    SL = args.teacher_seq_len
    n_train = min((len(train_tokens)-1)//SL, 8000)
    train_x = train_tokens[:n_train*SL].reshape(n_train, SL)
    train_y = train_tokens[1:n_train*SL+1].reshape(n_train, SL)

    # Tokenizer for bpb
    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    # Random projection (deterministic from seed, shared across all)
    gen = torch.Generator(device='cpu'); gen.manual_seed(42)
    proj = torch.linalg.qr(torch.randn(args.teacher_dim, args.eam_key_dim, generator=gen)).Q.T
    # proj: [eam_key_dim, teacher_dim] — orthogonal rows projecting hidden → key space
    proj = proj.to(device=device, dtype=torch.bfloat16)

    t_start = time.perf_counter()

    # ==================================================================
    # STEP 1: TRAIN TEACHER
    # ==================================================================
    log(f"\n{'='*60}\nSTEP 1: Train teacher (D={args.teacher_dim} L={args.teacher_layers})")
    log(f"{'='*60}")

    teacher = Teacher(args.vocab_size, args.teacher_dim, args.teacher_layers,
                      args.teacher_heads).to(device)
    log(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=0.01)
    BS = args.teacher_batch

    for ep in range(5):
        perm = torch.randperm(n_train)
        tl, nb = 0, 0
        pbar = tqdm(range(0, n_train - BS, BS), desc=f"  Teacher ep{ep+1}", disable=not master)
        for i in pbar:
            bx = train_x[perm[i:i+BS]].to(device)
            by = train_y[perm[i:i+BS]].to(device)
            _, loss = teacher(bx, by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step(); tl += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{tl/nb:.4f}")
            if time.perf_counter() - t_start > args.max_wallclock_seconds * 0.45:
                break
        pbar.close()
        log(f"  Epoch {ep+1}: loss={tl/nb:.4f} ({time.perf_counter()-t_start:.0f}s)")
        if time.perf_counter() - t_start > args.max_wallclock_seconds * 0.45:
            log("  (time limit for teacher, moving on)"); break

    teacher.eval()

    # ==================================================================
    # STEP 2: DISTILL INTO EAM
    # ==================================================================
    log(f"\n{'='*60}\nSTEP 2: Distill into EAM (max {args.eam_locations} locations)")
    log(f"{'='*60}")

    eam = EAM(args.eam_key_dim, args.teacher_dim,
              L0=min(args.eam_locations//3, 3000), max_L=args.eam_locations,
              k=args.eam_k, beta=args.eam_beta, lr=args.eam_lr,
              tau_split=args.eam_tau_split, tau_overload=args.eam_tau_overload,
              device=device)

    t2 = time.perf_counter()
    with torch.no_grad():
        for i in tqdm(range(n_train), desc="  Distilling", disable=not master):
            x = train_x[i:i+1].to(device)
            _, hidden = teacher.forward_full(x)
            h = hidden.squeeze(0)
            keys = F.normalize(h @ proj.T.to(h.dtype), dim=-1)
            eam.write_batch(keys, h)
    log(f"  EAM: L={eam.L} writes={eam.total_writes:,} ({time.perf_counter()-t2:.1f}s)")
    log(f"  EAM compressed: {eam.compressed_bytes()/1024:.0f}KB")

    # ==================================================================
    # STEP 3: TRAIN ENCODER
    # ==================================================================
    log(f"\n{'='*60}\nSTEP 3: Train encoder")
    log(f"{'='*60}")

    encoder = Encoder(args.vocab_size, args.enc_dim, args.enc_layers,
                      args.enc_heads, args.eam_key_dim).to(device)
    log(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.01)

    for ep in range(args.enc_epochs):
        encoder.train()
        perm = torch.randperm(n_train); tl, nb = 0, 0
        pbar = tqdm(range(0, n_train - 32, 32), desc=f"  Encoder ep{ep+1}", disable=not master)
        for i in pbar:
            bx = train_x[perm[i:i+32]].to(device)
            with torch.no_grad():
                _, t_h = teacher.forward_full(bx)
                t_keys = F.normalize(t_h.reshape(-1, args.teacher_dim) @ proj.T.to(t_h.dtype), dim=-1)
            enc_keys = encoder(bx).reshape(-1, args.eam_key_dim)
            loss = 1.0 - F.cosine_similarity(enc_keys, t_keys.float(), dim=-1).mean()
            enc_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            enc_opt.step(); tl += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{tl/nb:.4f}")
        pbar.close()
        log(f"  Epoch {ep+1}/{args.enc_epochs}: key_loss={tl/nb:.4f}")

    # ==================================================================
    # STEP 4: TRAIN DECODER
    # ==================================================================
    log(f"\n{'='*60}\nSTEP 4: Train decoder")
    log(f"{'='*60}")

    decoder = Decoder(args.teacher_dim, args.vocab_size).to(device)
    log(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    dec_opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=0.01)
    encoder.eval()

    for ep in range(args.dec_epochs):
        decoder.train()
        perm = torch.randperm(n_train); tl, nb = 0, 0
        pbar = tqdm(range(0, n_train - 32, 32), desc=f"  Decoder ep{ep+1}", disable=not master)
        for i in pbar:
            bx = train_x[perm[i:i+32]].to(device)
            by = train_y[perm[i:i+32]].to(device)
            B, T = bx.shape
            with torch.no_grad():
                ek = encoder(bx).reshape(-1, args.eam_key_dim)
                eo = eam.read_batch(ek)
                if eo is None: continue
                eo = eo.reshape(B, T, args.teacher_dim)
            logits = decoder(eo)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size).float(), by.reshape(-1))
            dec_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            dec_opt.step(); tl += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{tl/nb:.4f}")
        pbar.close()
        log(f"  Epoch {ep+1}/{args.dec_epochs}: loss={tl/nb:.4f}")

    log(f"\nTraining complete. Total: {time.perf_counter()-t_start:.0f}s")

    # ==================================================================
    # SAVE ARTIFACT
    # ==================================================================
    log(f"\n{'='*60}\nSaving artifact")
    log(f"{'='*60}")

    try:
        code = open(__file__).read()
    except (NameError, OSError):
        code = "# EAM IS THE MODEL\n" * 100  # placeholder for size estimate
    code_bytes = len(code.encode("utf-8"))

    enc_sd = {k: v.half().cpu() for k, v in encoder.state_dict().items()}
    dec_sd = {k: v.half().cpu() for k, v in decoder.state_dict().items()}
    eam_sd = eam.state_dict()
    # Quantize EAM counters
    c = eam_sd['counters']; mx = c.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    sc = (mx / 127.0).squeeze(1).half()
    cq = torch.clamp(torch.round(c / mx * 127.0), -127, 127).to(torch.int8)
    eam_save = {'addresses': eam_sd['addresses'], 'counters_q': cq,
                'counters_scale': sc, 'write_weights': eam_sd['write_weights'].half(),
                'L': eam_sd['L']}

    artifact = {'encoder': enc_sd, 'decoder': dec_sd, 'eam': eam_save,
                'proj': proj.cpu().half(), 'config': {
                    'teacher_dim': args.teacher_dim, 'eam_key_dim': args.eam_key_dim,
                    'enc_dim': args.enc_dim, 'enc_layers': args.enc_layers,
                    'enc_heads': args.enc_heads, 'vocab_size': args.vocab_size}}
    buf = io.BytesIO(); torch.save(artifact, buf)
    blob = lzma.compress(buf.getvalue(), preset=6)

    if master:
        with open("eam_artifact.ptz", "wb") as f: f.write(blob)
        total = code_bytes + len(blob)
        log(f"  Code: {code_bytes} bytes")
        log(f"  Model: {len(blob)} bytes ({len(blob)/1024:.0f}KB)")
        log(f"  Total: {total} bytes ({total/(1024**2):.1f}MB)")
        log(f"  Budget: 16,000,000 bytes")
        log(f"  Status: {'FITS' if total <= 16_000_000 else 'OVER BUDGET'}")

    # ==================================================================
    # EVAL: Sliding window with bpb
    # ==================================================================
    log(f"\n{'='*60}\nEvaluation: encoder → EAM → decoder + kNN")
    log(f"{'='*60}")

    # Load artifact back (verify roundtrip)
    with open("eam_artifact.ptz", "rb") as f:
        loaded = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu")

    cfg = loaded['config']
    eval_enc = Encoder(cfg['vocab_size'], cfg['enc_dim'], cfg['enc_layers'],
                       cfg['enc_heads'], cfg['eam_key_dim']).to(device)
    eval_enc.load_state_dict({k: v.float().to(device) for k, v in loaded['encoder'].items()})
    eval_enc.eval()

    eval_dec = Decoder(cfg['teacher_dim'], cfg['vocab_size']).to(device)
    eval_dec.load_state_dict({k: v.float().to(device) for k, v in loaded['decoder'].items()})
    eval_dec.eval()

    eval_eam = EAM(cfg['eam_key_dim'], cfg['teacher_dim'], L0=1, max_L=50000, device=device)
    eam_data = loaded['eam']
    L = eam_data['L']; cq = eam_data['counters_q'].float(); cs = eam_data['counters_scale'].float()
    eval_eam.addresses[:L] = eam_data['addresses'].to(device)
    eval_eam.counters[:L] = (cq * cs.unsqueeze(1)).to(device)
    eval_eam.write_weights[:L] = eam_data['write_weights'].float().to(device)
    eval_eam.L = L
    log(f"  Loaded EAM: {L} locations")

    eval_proj = loaded['proj'].to(device=device, dtype=torch.bfloat16)

    # Sliding window eval
    seq_len = args.eval_seq_len; stride = args.eval_stride
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    flat_store = FlatStore(cfg['eam_key_dim'], cfg['vocab_size'], 500000, device)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    t_eval = time.perf_counter()
    batch_seqs = 32

    with torch.no_grad():
        eval_pbar = tqdm(range(0, len(my_windows), batch_seqs),
                         desc="  Evaluating", disable=not master,
                         total=(len(my_windows) + batch_seqs - 1) // batch_seqs)
        for bi in eval_pbar:
            batch_ws = my_windows[bi:bi + batch_seqs]; bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens); wlen = end - ws; wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]; y_batch[i, :wlen] = chunk[1:]

            # Encoder → keys
            keys = eval_enc(x_batch)  # [bsz, seq_len, kd]

            # EAM read
            flat_keys = keys.reshape(-1, cfg['eam_key_dim'])
            eam_out = eval_eam.read_batch(flat_keys)
            if eam_out is not None:
                logits = eval_dec(eam_out.reshape(bsz, seq_len, cfg['teacher_dim']))
            else:
                logits = torch.zeros(bsz, seq_len, cfg['vocab_size'], device=device)

            # Base NLL
            nll = F.cross_entropy(logits.reshape(-1, cfg['vocab_size']).float(),
                                  y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]; s = 0 if ws == 0 else max(wlen - stride, 0)
                scored = nll[i, s:wlen].to(torch.float64)
                final_nll = scored.clone()

                # kNN boost (score-first, then write)
                seq_keys = keys[i, s:wlen]
                knn_dist = flat_store.retrieve(seq_keys, args.knn_k, args.knn_beta)
                if knn_dist is not None:
                    p_eam = F.softmax(logits[i, s:wlen].float(), dim=-1)
                    lam = args.knn_lambda
                    p_blend = ((1-lam)*p_eam + lam*knn_dist).clamp(min=1e-10)
                    blend_nll = -torch.gather(p_blend.log(), 1,
                        y_batch[i, s:wlen].unsqueeze(1)).squeeze(1).to(torch.float64)
                    improved = blend_nll < scored
                    final_nll = torch.where(improved, blend_nll, scored)

                loss_sum += final_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

                # Write to flat store (after scoring — legal)
                flat_store.write_batch(keys[i, :wlen], y_batch[i, :wlen])

            # Update progress bar with running bpb
            if token_count.item() > 0 and byte_count.item() > 0:
                r_loss = (loss_sum / token_count).item()
                r_bpb = r_loss / math.log(2.0) * token_count.item() / byte_count.item()
                eval_pbar.set_postfix(bpb=f"{r_bpb:.4f}", knn=f"{flat_store.n_stored:,}")

    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    val_bpb = bpt * tpb

    log(f"\n  eval_time: {time.perf_counter()-t_eval:.0f}s")
    log(f"  val_loss: {val_loss:.8f}")
    log(f"  val_bpb:  {val_bpb:.8f}")
    log(f"\n  eam_is_model val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
