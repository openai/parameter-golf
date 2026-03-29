"""
Run inference locally on Mac (CPU/MPS) using trained weights from experiments/run_N/.
Loads the GPT model from train_gpt.py, restores weights, and generates text autoregressively.

Usage:
    uv run --with torch --with sentencepiece --with numpy python inference.py \
        --weights experiments/run_1/full_weights.pt \
        --prompt "The meaning of life is"

    # Use quantized weights (int8+zlib):
    uv run --with torch --with sentencepiece --with numpy python inference.py \
        --weights experiments/run_1/quantized_weights.int8.ptz \
        --prompt "Once upon a time"
"""
from __future__ import annotations

import argparse
import io
import sys
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool,
                 tied_embed_init_std: float, logit_softcap: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return raw logits (with softcap) instead of loss."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def load_model(weights_path: str, device: torch.device) -> GPT:
    path = Path(weights_path)
    if path.suffix == ".ptz":
        with open(path, "rb") as f:
            quant_blob = f.read()
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu", weights_only=False)
        state_dict = dequantize_state_dict_int8(quant_state)
        print(f"Loaded quantized weights from {path} ({len(quant_blob)} bytes compressed)")
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Loaded full weights from {path} ({path.stat().st_size} bytes)")

    has_lm_head = any(k.startswith("lm_head.") for k in state_dict)
    vocab_size = state_dict["tok_emb.weight"].shape[0]
    model_dim = state_dict["tok_emb.weight"].shape[1]
    num_blocks = len([k for k in state_dict if k.startswith("blocks.") and k.endswith(".attn_scale")])
    num_heads = state_dict["blocks.0.attn.q_gain"].shape[0]
    head_dim = model_dim // num_heads
    num_kv_heads = state_dict["blocks.0.attn.c_k.weight"].shape[0] // head_dim
    mlp_hidden = state_dict["blocks.0.mlp.fc.weight"].shape[0]
    mlp_mult = mlp_hidden // model_dim
    qk_gain_init = float(state_dict["blocks.0.attn.q_gain"].mean().item())

    print(f"Model: vocab={vocab_size} dim={model_dim} layers={num_blocks} heads={num_heads} "
          f"kv_heads={num_kv_heads} mlp_mult={mlp_mult} tie_emb={not has_lm_head}")

    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_blocks,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=not has_lm_head,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=qk_gain_init,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def generate(
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_id() if tokenizer.bos_id() >= 0 else 0]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {len(tokens)}")
    print(f"Generating {max_new_tokens} tokens (temp={temperature}, top_k={top_k}, top_p={top_p})")
    print("-" * 60)

    prev_text = tokenizer.decode(tokens)
    sys.stdout.write(prev_text)
    sys.stdout.flush()

    for _ in range(max_new_tokens):
        ctx = input_ids[:, -1024:]
        logits = model.forward_logits(ctx)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        full_text = tokenizer.decode(input_ids[0].tolist())
        new_text = full_text[len(prev_text):]
        sys.stdout.write(new_text)
        sys.stdout.flush()
        prev_text = full_text

    print("\n" + "-" * 60)
    return prev_text


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained Parameter Golf model")
    parser.add_argument("--weights", type=str, default="experiments/run_1/full_weights.pt",
                        help="Path to weights file (.pt or .ptz)")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to SentencePiece .model file")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus (top-p) filtering")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'mps', or 'auto'")
    parser.add_argument("--prompts-file", type=str, default="",
                        help="File with one prompt per line (overrides --prompt)")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab_size={tokenizer.vocab_size()})")

    model = load_model(args.weights, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    prompts = [args.prompt]
    if args.prompts_file:
        prompts = [line.strip() for line in Path(args.prompts_file).read_text().splitlines() if line.strip()]

    results = []
    for prompt in prompts:
        text = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, device=device)
        results.append({"prompt": prompt, "generated": text})

    if len(results) > 1:
        print(f"\n{'='*60}\nGenerated {len(results)} completions.")

    return results


if __name__ == "__main__":
    main()
