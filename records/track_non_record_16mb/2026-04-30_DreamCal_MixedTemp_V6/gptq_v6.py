"""
Full GPTQ quantization for V6 architecture with real Hessian calibration.
Wires the V6 model class from sliding_window_eval_v6.py with the GPTQ
algorithm from gptq.py.

Clip-search int4 gave 1.527 BPB (catastrophic — 0.368 gap from 1.159 float).
Full GPTQ with Hessian should be much better.

Usage:
  CUDA_VISIBLE_DEVICES=0 python gptq_v6.py --bits 4 --calib-seqs 128
  CUDA_VISIBLE_DEVICES=0 python gptq_v6.py --mixed --calib-seqs 128
"""
import os, sys, math, time, argparse, io, lzma, glob
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from torch import nn, Tensor
from pathlib import Path

sys.path.insert(0, '.')
from gptq import compute_hessian, gptq_quantize_weight, compress_artifact

# ── V6 Architecture (from sliding_window_eval_v6.py) ──

dim = 512
ROPE_DIMS = 16
BIGRAM_VOCAB = 3072
BIGRAM_DIM = 112

class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.eps = 1e-6
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def build_rope(seq_len, head_dim, rope_dims, base=10000.0, device='cpu'):
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims))
    angles = pos[:, None] * freqs[None, :]
    return torch.cos(angles).to(device), torch.sin(angles).to(device)

def apply_partial_rope(x, cos, sin, rope_dims):
    B, nh, T, hd = x.shape
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    x1, x2 = x_rope[..., 0::2], x_rope[..., 1::2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    o1 = x1 * cos_t - x2 * sin_t
    o2 = x2 * cos_t + x1 * sin_t
    x_rotated = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([x_rotated, x_pass], dim=-1)

class SmearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHash(nn.Module):
    def __init__(self, vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        nn.init.zeros_(self.embed.weight)
        nn.init.zeros_(self.proj.weight)
    def forward(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        h = self.embed(out.long())
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

# Need rope as global for Block.forward
rope_cos, rope_sin = None, None

class Block(nn.Module):
    def __init__(self, d, mm, layer_idx, n_layers, nh=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = CastedLinear(d, d, bias=False)
        self.k = CastedLinear(d, d//2, bias=False)
        self.v = CastedLinear(d, d//2, bias=False)
        self.o = CastedLinear(d, d, bias=False)
        self.fc = CastedLinear(d, d*mm, bias=False)
        self.proj = CastedLinear(d*mm, d, bias=False)
        self.nh, self.hd = nh, d // nh
        self.attn_scale = nn.Parameter(torch.ones(d))
        self.mlp_scale = nn.Parameter(torch.ones(d))
        self.q_gain = nn.Parameter(torch.full((nh,), 5.0))
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n1(x) * self.ln_scale
        q = self.q(h).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).reshape(B, T, self.nh//2, self.hd).transpose(1, 2)
        v = self.v(h).reshape(B, T, self.nh//2, self.hd).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q * self.q_gain[None, :, None, None]
        q = apply_partial_rope(q, rope_cos, rope_sin, ROPE_DIMS)
        k = apply_partial_rope(k, rope_cos, rope_sin, ROPE_DIMS)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = a.transpose(1, 2)
        v_t = v.transpose(1, 2)
        Hkv = v_t.size(2)
        group = self.nh // Hkv
        y_g = y.reshape(B, T, Hkv, group, self.hd)
        vn = F.normalize(v_t, dim=-1).unsqueeze(3)
        proj_xsa = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        y = (y_g - proj_xsa).reshape(B, T, self.nh, self.hd)
        attn_out = self.o(y.contiguous().reshape(B, T, C))
        x = x + self.attn_scale * attn_out
        h2 = self.n2(x) * self.ln_scale
        x = x + self.mlp_scale * self.proj(F.leaky_relu(self.fc(h2), negative_slope=0.5).square())
        return x

class GPT(nn.Module):
    def __init__(self, nl, mm):
        super().__init__()
        vs = max(4096, 4096)
        self.emb = nn.Embedding(vs, dim)
        self.bigram = BigramHash(BIGRAM_VOCAB, BIGRAM_DIM, dim)
        self.smear = SmearGate(dim)
        self.blocks = nn.ModuleList([Block(dim, mm, i, nl) for i in range(nl)])
        self.ln = RMSNorm(dim)
        self.n_enc = nl // 2
        self.n_dec = nl - self.n_enc
        self.skip_weights = nn.Parameter(torch.ones(min(self.n_enc, self.n_dec), dim))

    def forward(self, idx):
        x = F.rms_norm(self.emb(idx), (dim,))
        x = x + self.bigram(idx)
        x = self.smear(x)
        skips = []
        for i in range(self.n_enc):
            x = self.blocks[i](x); skips.append(x)
        for i in range(self.n_dec):
            if skips: x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.n_enc + i](x)
        logits = F.linear(self.ln(x), self.emb.weight)
        return 30.0 * torch.tanh(logits / 30.0)


# ── Calibration + GPTQ ──

def collect_activations(model, data_tokens, n_seqs=128, seq_len=512, device='cpu'):
    """Run forward passes and collect per-layer input activations for Hessian."""
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.size(-1))
            if name not in activations:
                activations[name] = []
            activations[name].append(x.cpu())
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, CastedLinear)) and module.weight.numel() > 65536:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Hook final RMSNorm output for embedding Hessian (emb.weight is output head)
    def make_output_hook(name):
        def hook_fn(module, input, output):
            x = output.detach().float()  # Use OUTPUT of RMSNorm, not input
            if x.ndim == 3:
                x = x.reshape(-1, x.size(-1))
            if name not in activations:
                activations[name] = []
            activations[name].append(x.cpu())
        return hook_fn

    if hasattr(model, 'ln'):
        hooks.append(model.ln.register_forward_hook(make_output_hook('emb')))

    model.eval()
    n_tokens = data_tokens.numel()
    with torch.no_grad():
        for i in range(min(n_seqs, n_tokens // seq_len)):
            start = i * seq_len
            x = data_tokens[start:start + seq_len].long().unsqueeze(0).to(device)
            model(x)
            if (i + 1) % 32 == 0:
                print(f'  Calibration: {i+1}/{n_seqs} seqs', flush=True)

    for h in hooks:
        h.remove()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)
        print(f'  {name}: {activations[name].shape}', flush=True)

    return activations


def gptq_quantize_model(state_dict, activations, bits=4, mixed=False, emb6=False, attn6=False, block_size=128, damp=0.005):
    """Full GPTQ quantization with real Hessian."""
    result = {
        '__format__': f'gptq_{"mixed" if mixed else f"int{bits}"}_v2_hessian',
        'quantized': {},
        'scales': {},
        'bits_per_layer': {},
        'passthrough': {},
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if t.numel() <= 65536 or not t.is_floating_point():
            result['passthrough'][name] = t.to(torch.float16) if t.is_floating_point() else t
            continue
        if t.ndim != 2:
            result['passthrough'][name] = t.to(torch.float16)
            continue

        is_mlp = any(k in name for k in ('fc.weight', 'proj.weight'))
        is_attn = any(k in name for k in ('q.weight', 'k.weight', 'v.weight', 'o.weight'))
        is_emb = name in ('emb.weight',)
        if attn6 and is_attn:
            layer_bits = 6
        elif emb6 and is_emb:
            layer_bits = 6
        elif mixed:
            layer_bits = 4 if is_mlp else 6
        else:
            layer_bits = bits

        # Find matching activation key
        # Weight name: "blocks.0.fc.weight" -> module name: "blocks.0.fc"
        act_key = name.rsplit('.weight', 1)[0] if name.endswith('.weight') else name

        if act_key in activations:
            H = compute_hessian(activations[act_key])
            print(f'  GPTQ {name}: {t.shape} -> int{layer_bits} (Hessian {H.shape})', flush=True)
            q, s = gptq_quantize_weight(t, H, bits=layer_bits, block_size=block_size, damp=damp)
        else:
            # Fallback to clip search
            print(f'  Clip  {name}: {t.shape} -> int{layer_bits} (no Hessian)', flush=True)
            max_val = 7 if layer_bits == 4 else 31
            row_max = t.float().abs().amax(dim=1).clamp(min=1e-12)
            s = (row_max / max_val).to(torch.float16)
            q = torch.round(t.float() / s.float()[:, None]).clamp(-max_val, max_val).to(torch.int8)

        result['quantized'][name] = q
        result['scales'][name] = s
        result['bits_per_layer'][name] = layer_bits

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_model_v6_ema.pt')
    parser.add_argument('--bits', type=int, default=4, choices=[4, 6])
    parser.add_argument('--mixed', action='store_true', help='int4 MLP + int6 attention (all)')
    parser.add_argument('--attn6', action='store_true', help='int6 for q/k/v/o only, int4 for fc/proj/emb (+2.2MB, fits 16MB)')
    parser.add_argument('--calib-seqs', type=int, default=128)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--damp', type=float, default=0.005)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--self-gen', action='store_true', help='Use model-generated text for calibration (SOTA technique)')
    parser.add_argument('--emb6', action='store_true', help='Keep embedding at int6 (95%% less MSE for +493KB LZMA)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seq_len = args.seq_len

    # Build RoPE (module-level globals used by Block.forward)
    import gptq_v6
    gptq_v6.rope_cos, gptq_v6.rope_sin = build_rope(seq_len, dim // 8, ROPE_DIMS, device=device)
    rope_cos, rope_sin = gptq_v6.rope_cos, gptq_v6.rope_sin

    # Load model
    print(f'Loading {args.model}...', flush=True)
    model = GPT(11, 3).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} params on {device}', flush=True)

    # Load or generate calibration data
    if args.self_gen:
        # Self-generated calibration: generate text from model itself (SOTA technique)
        print(f'\nGenerating self-calibration data ({args.calib_seqs} seqs, seq_len={seq_len})...', flush=True)
        t0 = time.time()
        gen_seqs = []
        train_files = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_train_*.bin'))
        seed_tokens = torch.from_numpy(
            np.fromfile(Path(train_files[0]), dtype='<u2', offset=256*4).astype(np.uint16)
        )
        with torch.no_grad():
            for i in range(args.calib_seqs):
                # Start with a short seed from training data, then generate the rest
                seed_len = 32
                seed = seed_tokens[i * seed_len:(i + 1) * seed_len].long().unsqueeze(0).to(device)
                tokens = seed
                for _ in range(seq_len - seed_len):
                    logits = model(tokens[:, -seq_len:])
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens = torch.cat([tokens, next_token], dim=1)
                gen_seqs.append(tokens.squeeze(0).cpu())
                if (i + 1) % 32 == 0:
                    print(f'  Generated {i+1}/{args.calib_seqs} seqs', flush=True)
        calib_tokens = torch.cat(gen_seqs)
        print(f'Self-gen calibration: {calib_tokens.numel():,} tokens in {time.time()-t0:.1f}s', flush=True)
    else:
        # Standard calibration from training data
        train_files = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_train_*.bin'))
        if not train_files:
            print('ERROR: No SP4096 training files found. Falling back to val.', flush=True)
            train_files = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_val_*.bin'))
        calib_tokens = torch.from_numpy(
            np.fromfile(Path(train_files[0]), dtype='<u2', offset=256*4).astype(np.uint16)
        )
        print(f'Calibration tokens: {calib_tokens.numel():,} from {train_files[0]}', flush=True)

    # Collect activations
    print(f'\nCollecting activations ({args.calib_seqs} seqs, seq_len={seq_len})...', flush=True)
    t0 = time.time()
    activations = collect_activations(model, calib_tokens, n_seqs=args.calib_seqs,
                                       seq_len=seq_len, device=device)
    print(f'Activations collected in {time.time()-t0:.1f}s\n', flush=True)

    # Get state dict for quantization
    state_dict = {k: v for k, v in model.state_dict().items()}

    # Quantize
    bits = args.bits
    if args.attn6:
        print(f'GPTQ int{bits} + int6 attention q/k/v/o (with Hessian)', flush=True)
    elif args.mixed:
        print(f'Mixed GPTQ: int4 MLP + int6 attention+proj (with Hessian)', flush=True)
    elif args.emb6:
        print(f'GPTQ int{bits} + int6 embedding (with Hessian)', flush=True)
    else:
        print(f'Uniform GPTQ int{bits} (with Hessian)', flush=True)

    t0 = time.time()
    quant = gptq_quantize_model(state_dict, activations, bits=bits, mixed=args.mixed,
                                 emb6=args.emb6, attn6=args.attn6,
                                 block_size=args.block_size, damp=args.damp)
    print(f'\nQuantized in {time.time()-t0:.1f}s', flush=True)

    # Compress
    compressed = compress_artifact(quant)
    code_est = 50000
    ngram_est = 800000
    total = len(compressed) + code_est + ngram_est
    print(f'\nLZMA: {len(compressed)/1e6:.3f} MB', flush=True)
    print(f'Total (model + code + ngram): {total/1e6:.3f} MB', flush=True)
    print(f'{"OK" if total < 16e6 else "OVER"} (headroom: {(16e6-total)/1024:.0f} KB)', flush=True)

    # Save artifact
    suffix = 'mixed' if args.mixed else ('attn6' if args.attn6 else (f'{bits}bit_emb6' if args.emb6 else f'{bits}bit'))
    artifact_path = args.model.replace('.pt', f'.gptq_{suffix}_hessian.lzma')
    with open(artifact_path, 'wb') as f:
        f.write(compressed)
    print(f'Artifact: {artifact_path}', flush=True)

    # Save dequantized roundtrip for eval
    from gptq import dequantize_gptq_model
    rt_state = dequantize_gptq_model(quant)
    rt_path = args.model.replace('.pt', f'_gptq_{suffix}_hessian_roundtrip.pt')
    torch.save(rt_state, rt_path)
    print(f'Roundtrip model: {rt_path}', flush=True)
    print(f'\nNext: python sliding_window_eval_v6.py {rt_path} --gpu 0', flush=True)
