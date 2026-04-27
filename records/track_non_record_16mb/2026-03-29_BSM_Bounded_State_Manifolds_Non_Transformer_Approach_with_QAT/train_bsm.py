%%writefile train_bsm.py

"""
Pure BSM (Bounded State Manifold) - Ternary Parameter Golf Challenge Entry
Architecture: 75M Ternary BSM (<16MB compressed)
Features: O(N) Box Intersection Mixer, Exact SentencePiece BPB Eval, Muon Optimizer, Base-3 Packing
"""

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./parameter-golf/data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"bsm_run_{int(time.time())}")
    seed = int(os.environ.get("SEED", 42))
    
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))
    
    iterations = int(os.environ.get("ITERATIONS", 2000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_fraction = float(os.environ.get("WARMDOWN_FRACTION", 0.2))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 5850000000.0)) # 9.75 mins
    
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))

# ---------------------------------------------------------------------------
# Muon Optimizer & Distributed Loader (From Official Script)
# ---------------------------------------------------------------------------
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps))

    @torch.no_grad()
    def step(self):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps = group["lr"], group["momentum"], group["backend_steps"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError("Invalid shard")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.file_idx])
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ---------------------------------------------------------------------------
# Ternary Compression (Base-3 Packing into 16MB)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:,0] + g[:,1]*3 + g[:,2]*9 + g[:,3]*27 + g[:,4]*81).tobytes(), n

def q_sd(state_dict: dict, group_size: int = 64) -> dict:
    quantized = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float().contiguous()
        if t.ndim == 2 and t.numel() > 65_536 and "star_map" not in name:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half()
            q = (t_grouped / scale.float()).round().clamp(-1, 1).to(torch.int8)
            packed_bytes, n_trits = pack_ternary(q)
            quantized[name] = {
                "type": "ternary", "packed": packed_bytes, "scale": scale.squeeze(-1),
                "shape": list(t.shape), "padded_cols": t_padded.shape[1],
                "group_size": group_size, "n_trits": n_trits
            }
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
    return quantized

# ---------------------------------------------------------------------------
# Pure BSM Architecture
# ---------------------------------------------------------------------------
class TernaryBoxLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int=64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)
        self.group_size = group_size
        self.enable_qat = False # Starts FALSE

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.bfloat16()
        
        # Fast path for the first 85% of training
        if not self.enable_qat:
            return F.linear(x, w)
            
        # Slow path only activated at the end
        w_g = w.reshape(-1, self.group_size)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary)

class StaticStarMapEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        init_centers = torch.empty(vocab_size, dim)
        nn.init.orthogonal_(init_centers)
        self.centers = nn.Parameter(init_centers, requires_grad=True)
        self.widths_raw = nn.Parameter(torch.zeros(vocab_size, dim))

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.centers[token_ids], F.softplus(self.widths_raw[token_ids]) + 1e-4

class BoxIntersectionMixer(nn.Module):
    def __init__(self, dim: int, layer_idx: int):
        super().__init__()
        
        # 1. EXPONENTIAL DILATION: Memory grows powers of 2 (1, 2, 4, 8...)
        # Cap dilation at 8 to prevent overstretching on 12-layer models
        self.dilation = 2 ** (layer_idx % 8) 
        self.kernel_size = 4
        
        # Causal padding scales with dilation
        self.padding = (self.kernel_size - 1) * self.dilation
        
        self.box_conv = nn.Conv1d(
            in_channels=dim, out_channels=dim,
            kernel_size=self.kernel_size, padding=self.padding, 
            dilation=self.dilation, groups=dim, bias=False
        )
        self.gate_proj = TernaryBoxLinear(dim, dim)
        self.shift_proj = TernaryBoxLinear(dim, dim)
        
        with torch.no_grad():
            self.shift_proj.weight.zero_()

    def forward(self, x_c: Tensor, x_w: Tensor) -> tuple[Tensor, Tensor]:
        c_transposed = x_c.transpose(1, 2)
        
        # Apply dilated conv and dynamically slice the exact padding amount off the right
        c_mixed = self.box_conv(c_transposed)[:, :, :-self.padding].transpose(1, 2)
        
        gates = torch.sigmoid(self.gate_proj(x_c))
        delta_c = self.shift_proj(c_mixed) * gates
        state_w = x_w * (1.0 - gates * 0.1)
        
        return delta_c, state_w


class BSMBlock(nn.Module):
    def __init__(self, dim: int, layer_idx: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.mixer = BoxIntersectionMixer(dim, layer_idx)
        self.mixer_scale = nn.Parameter(torch.ones(dim))
        
        self.norm2 = nn.RMSNorm(dim)
        
        # 2. GATED MLP (SwiGLU Style) 
        # We output 2x the hidden dim, chunk it, and multiply them together
        hidden_dim = int(dim * 2.66) # Standard efficient scaling ratio
        self.up = TernaryBoxLinear(dim, hidden_dim * 2) 
        self.down = TernaryBoxLinear(hidden_dim, dim)
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        
        with torch.no_grad():
            self.down.weight.zero_()

    def forward(self, c: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        mix_c_delta, mix_w = self.mixer(self.norm1(c), w)
        c = c + self.mixer_scale.to(c.dtype)[None, None, :] * mix_c_delta
        w = torch.min(w, mix_w)
        
        # Gated Forward Pass
        up_out = self.up(self.norm2(c))
        # Split the output in half: one is the gate, one is the projection
        gate, proj = up_out.chunk(2, dim=-1) 
        
        # Swish-Gated Linear Unit (SwiGLU) magic
        route = F.silu(gate) * proj 
        mlp_delta = self.down(route)
        
        c = c + self.mlp_scale.to(c.dtype)[None, None, :] * mlp_delta
        return c, w

class PureBSM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int):
        super().__init__()
        self.star_map = StaticStarMapEmbedding(vocab_size, dim)
        self.layers = nn.ModuleList([BSMBlock(dim, i) for i in range(num_layers)])
        self.final_norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        c, w = self.star_map(x)
        for layer in self.layers:
            c, w = layer(c, w)
        return self.final_norm(c), w
        
    def compute_logits(self, pred_c: Tensor) -> Tensor:
        # Standard un-normalized dot product. 
        # Gradients flow beautifully through this.
        return F.linear(pred_c, self.star_map.centers)

    def calculate_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        pred_c, pred_w = self(input_ids)
        
        # Pure cross entropy. Nothing else. Let the optimizer do its job.
        logits = self.compute_logits(pred_c)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

# ---------------------------------------------------------------------------
# EXACT Validation Evaluator (SentencePiece BPB)
# ---------------------------------------------------------------------------
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    base_bytes_np = np.zeros((vocab_size,), dtype=np.int16)
    has_ls_np = np.zeros((vocab_size,), dtype=np.bool_)
    is_bound_np = np.ones((vocab_size,), dtype=np.bool_)
    for i in range(sp_vocab_size):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
        is_bound_np[i] = False
        if sp.is_byte(i):
            base_bytes_np[i] = 1
            continue
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            has_ls_np[i] = True
            piece = piece[1:]
        base_bytes_np[i] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, device=device), 
            torch.tensor(has_ls_np, device=device), 
            torch.tensor(is_bound_np, device=device))

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, b_lut, ls_lut, ib_lut):
    local_batch_seqs = max(1, (args.val_batch_size // (world_size * grad_accum_steps)) // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    
    val_loss_sum, val_token_count, val_byte_count = 0.0, 0.0, 0.0
    model.eval()
    
    with torch.inference_mode():
        for b_start in range(seq_start, seq_end, local_batch_seqs):
            b_end = min(b_start + local_batch_seqs, seq_end)
            local = val_tokens[b_start * args.train_seq_len : b_end * args.train_seq_len + 1].to(device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_loss = model.module.calculate_loss(x, y) if isinstance(model, DDP) else model.calculate_loss(x, y)
                
            n = float(y.numel())
            val_loss_sum += batch_loss.item() * n
            val_token_count += n
            
            p_ids, t_ids = x.reshape(-1), y.reshape(-1)
            tb = b_lut[t_ids] + (ls_lut[t_ids] & ~ib_lut[p_ids]).to(torch.int16)
            val_byte_count += tb.sum().item()

    if dist.is_available() and dist.is_initialized():
        metrics = torch.tensor([val_loss_sum, val_token_count, val_byte_count], device=device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_loss_sum, val_token_count, val_byte_count = metrics.tolist()

    val_loss = val_loss_sum / val_token_count
    bpb = (val_loss / math.log(2.0)) * (val_token_count / val_byte_count)
    model.train()
    return val_loss, bpb

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def main():
    args = Hyperparameters()
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ
    rank, local_rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed: dist.init_process_group("nccl", device_id=device)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = torch.cat([load_data_shard(Path(p)) for p in sorted(glob.glob(args.val_files))]).contiguous()
    b_lut, ls_lut, ib_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    print("Loading Model")
    base_model = PureBSM(args.vocab_size, args.model_dim, args.num_layers).to(device).bfloat16()
    compiled_model = torch.compile(base_model)
    print("Model Compiled")
    model = DDP(compiled_model, device_ids=[local_rank]) if distributed else compiled_model
    print("Model Loaded")

    matrix_params = []
    scalar_params = []
    
    # We use named_parameters() so we can check the names of the layers
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
            
        # If it's 2D AND it's not part of our embeddings, give it to Muon
        if p.ndim == 2 and "star_map" not in name:
            matrix_params.append(p)
        # Otherwise (1D parameters, or any embedding weights), give it to AdamW
        else:
            scalar_params.append(p)
    
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    opt_scalar = torch.optim.AdamW([{"params": scalar_params}], lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    optimizers = [opt_muon, opt_scalar]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    t0 = time.perf_counter()
    training_time_ms = 0.0
    qat_activated = False

    print("Starting Training")
    for step in range(args.iterations):
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        
        if elapsed_ms >= args.max_wallclock_seconds * 1000:
            if rank == 0: print("Time cap reached. Compressing and exiting...")
            break

        # -------------------------------------------------------------------
        # LATE QAT ACTIVATION CHECK (Trigger at 85% of total allowed time)
        # -------------------------------------------------------------------
        # QAT FIx
        if not qat_activated and step >= int(args.iterations * 0.85):
            if rank == 0: print(f"Step {step} | Time: {elapsed_ms/1000:.1f}s | ACTIVATING LATE TERNARY QAT")
            for m in model.modules():
                # We check the class name as a string to avoid import issues if wrapped in DDP
                if m.__class__.__name__ == "TernaryBoxLinear":
                    m.enable_qat = True
            qat_activated = True
        # -------------------------------------------------------------------
        

        if step % args.val_loss_every == 0 or step == args.iterations - 1:
            print("Validating")
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, b_lut, ls_lut, ib_lut)
            if rank == 0: print(f"Step {step} | Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.4f}")

        for opt in optimizers: opt.zero_grad(set_to_none=True)
        
        train_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.module.calculate_loss(x, y) if isinstance(model, DDP) else model.calculate_loss(x, y)
            (loss * grad_scale).backward()
            train_loss += loss.item() * grad_scale
            
        for opt in optimizers: opt.step()
        
        if rank == 0 and step % args.train_log_every == 0:
            print(f"Step {step} | Train GeoLoss: {train_loss:.4f} | Time: {elapsed_ms/1000:.1f}s")
            
    # Serialize Base-3
    if rank == 0:
        q_obj = q_sd(base_model.state_dict())
        buf = io.BytesIO()
        torch.save(q_obj, buf)
        with open("final_model.ternary.ptz", "wb") as f:
            f.write(lzma.compress(buf.getvalue(), preset=9))
        print(f"Successfully saved. Compressed size: {os.path.getsize('final_model.ternary.ptz') / 1e6:.2f} MB")

if __name__ == "__main__":
    main()