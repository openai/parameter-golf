#!/usr/bin/env python3
"""Patch PR #1953 train_gpt.py with the BigramHashEmbedding branch.

This is intentionally surgical and idempotent. It is meant for smoke testing
the only plausible orthogonal branch on top of the clean #1953 lineage.
"""

from __future__ import annotations

import sys
from pathlib import Path


BIGRAM_CLASS = r'''

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        if bigram_vocab_size < 2:
            raise ValueError(f"bigram_vocab_size must be >=2, got {bigram_vocab_size}")
        self.bigram_vocab_size = int(bigram_vocab_size)
        self.embed = nn.Embedding(self.bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        if bigram_dim != model_dim:
            self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
            nn.init.orthogonal_(self.proj.weight, gain=1.0)
        else:
            self.proj = None
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "class BigramHashEmbedding" in text:
        print(f"{path}: BigramHash patch already present")
        return

    text = replace_once(
        text,
        '    embed_bits = int(os.environ.get("EMBED_BITS", 8))\n',
        '    embed_bits = int(os.environ.get("EMBED_BITS", 8))\n'
        '    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))\n'
        '    bigram_dim = int(os.environ.get("BIGRAM_DIM", 32))\n'
        '    bigram_bits = int(os.environ.get("BIGRAM_BITS", 6))\n',
        "Hyperparameters embed_bits",
    )

    text = replace_once(
        text,
        "\nclass GPT(nn.Module):\n",
        BIGRAM_CLASS + "\nclass GPT(nn.Module):\n",
        "GPT class",
    )

    text = replace_once(
        text,
        "        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)\n",
        "        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)\n"
        "        self.bigram = (\n"
        "            BigramHashEmbedding(h.bigram_vocab_size, h.bigram_dim, h.model_dim)\n"
        "            if h.bigram_vocab_size > 0\n"
        "            else None\n"
        "        )\n",
        "GPT tok_emb",
    )

    # Two occurrences: _forward_hidden and forward_ttt.
    replaced = text.count("        x = self.tok_emb(input_ids)\n")
    if replaced != 2:
        raise SystemExit(f"expected two token embedding anchors, found {replaced}")
    text = text.replace(
        "        x = self.tok_emb(input_ids)\n",
        "        x = self.tok_emb(input_ids)\n"
        "        if self.bigram is not None:\n"
        "            x = x + self.bigram(input_ids)\n",
    )

    text = replace_once(
        text,
        '"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,parallel_post_lambdas,parallel_resid_lambdas,attn_gate_proj,attn_gate_w,smear_gate,smear_lambda",\n',
        '"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,parallel_post_lambdas,parallel_resid_lambdas,attn_gate_proj,attn_gate_w,smear_gate,smear_lambda,bigram.scale",\n',
        "CONTROL_TENSOR_NAME_PATTERNS",
    )

    text = replace_once(
        text,
        "        # SmearGate params live on GPT root (not in .blocks), so add them by hand.\n",
        "        if getattr(base_model, \"bigram\", None) is not None:\n"
        "            if base_model.bigram.proj is not None:\n"
        "                matrix_params.append(base_model.bigram.proj.weight)\n"
        "            scalar_params.append(base_model.bigram.scale)\n"
        "        # SmearGate params live on GPT root (not in .blocks), so add them by hand.\n",
        "optimizer bigram params",
    )

    text = replace_once(
        text,
        '        tok_params = [\n'
        '            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}\n'
        '        ]\n',
        '        tok_params = [\n'
        '            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}\n'
        '        ]\n'
        '        if getattr(base_model, "bigram", None) is not None:\n'
        '            tok_params.append(\n'
        '                {"params": [base_model.bigram.embed.weight], "lr": h.embed_lr, "base_lr": h.embed_lr}\n'
        '            )\n',
        "optimizer token params",
    )

    text = replace_once(
        text,
        "        # Dedicated int8-per-row path for attn_gate_w (bypasses both GPTQ and\n",
        "        if t.is_floating_point() and name == \"bigram.embed.weight\":\n"
        "            bits = int(getattr(h, \"bigram_bits\", 6))\n"
        "            qmax = 2 ** (bits - 1) - 1\n"
        "            row_max = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)\n"
        "            s = (row_max / qmax).squeeze(-1).to(torch.float16)\n"
        "            q = torch.clamp(torch.round(t / s.float().view(-1, 1)), -qmax, qmax).to(torch.int8)\n"
        "            result[name + \".q\"] = q\n"
        "            result[name + \".scale\"] = s\n"
        "            meta[name] = f\"bigram_embed_int{bits}\"\n"
        "            continue\n"
        "        if t.is_floating_point() and name == \"bigram.proj.weight\" and t.ndim == 2:\n"
        "            gq, gs = _quantize_gate_int8_row(t)\n"
        "            result[name + \".gq\"] = gq\n"
        "            result[name + \".gs\"] = gs\n"
        "            meta[name] = \"bigram_proj_int8_row\"\n"
        "            continue\n"
        "        # Dedicated int8-per-row path for attn_gate_w (bypasses both GPTQ and\n",
        "quantization bigram hooks",
    )

    if '        if info in ("gate_int8_row", "small_control_int8_row"):\n' in text:
        text = replace_once(
            text,
            '        if info in ("gate_int8_row", "small_control_int8_row"):\n',
            '        if info in ("gate_int8_row", "small_control_int8_row", "bigram_proj_int8_row"):\n',
            "dequant int8 row",
        )
    else:
        text = replace_once(
            text,
            '        if info == "gate_int8_row":\n',
            '        if info in ("gate_int8_row", "bigram_proj_int8_row"):\n',
            "dequant int8 row",
        )

    path.write_text(text, encoding="utf-8")
    print(f"{path}: applied BigramHash patch")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_bigramhash_1953.py /path/to/train_gpt.py")
    patch(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
