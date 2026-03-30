"""
Gravitational Lensing Probe — Does information flow bend around high-gravity tokens?

Hypothesis: High-leverage tokens act as gravitational lenses in attention space.
Inserting a massive token between subject and object should collapse the direct
attention path and force information to route through the massive token.

The falsifiable signature: A(object -> subject) drops when a high-gravity token
is inserted, while A(object -> massive) + A(massive -> subject) absorbs the mass.
Low-gravity control tokens should show significantly less deflection.
"""

import io
import math
import os
import sys
import zlib
import json

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

import sentencepiece as spm

SCRIPT_DIR = os.path.dirname(__file__)
GOLF_ROOT = os.path.join(SCRIPT_DIR, "..", "parameter-golf")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


# ── Model (from generate.py, modified to capture attention weights) ──

class CastedLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__(in_f, out_f, bias=bias)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.inv_freq = nn.Parameter(
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        # Storage for captured attention weights
        self.last_attn_weights = None

    def forward(self, x, capture_attn=False):
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

        if self.num_kv_heads < self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask[None, None], float('-inf'))
        attn_weights = F.softmax(attn, dim=-1, dtype=torch.float32)

        if capture_attn:
            self.last_attn_weights = attn_weights.detach().cpu()

        y = (attn_weights.to(x.dtype) @ v).transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.proj = CastedLinear(mlp_mult * dim, dim, bias=False)
    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0, capture_attn=False):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x), capture_attn=capture_attn)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base, qk_gain_init):
        super().__init__()
        self.tie_embeddings = tie_embeddings
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

    def forward_with_attention(self, input_ids):
        """Forward pass that captures attention weights from all layers."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, capture_attn=True)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, capture_attn=True)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        # Collect attention from all layers
        all_attn = []
        for block in self.blocks:
            all_attn.append(block.attn.last_attn_weights)
        return logits, all_attn


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = obj.get("passthrough_orig_dtypes", {}).get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def load_model(checkpoint_path, tokenizer_path, num_layers=12, device="cuda"):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    model = GPT(
        vocab_size=1024, num_layers=num_layers, model_dim=384,
        num_heads=6, num_kv_heads=2, mlp_mult=3,
        tie_embeddings=True, logit_softcap=30.0,
        rope_base=10000.0, qk_gain_init=1.5,
    )

    with open(checkpoint_path, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)),
                             map_location="cpu", weights_only=True)
    state_dict = dequantize_state_dict_int8(quant_state)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, sp


def tokenize_and_locate(sp, text, target_tokens):
    """
    Tokenize text and find token SPANS covering each target word.
    Returns positions as (start_idx, end_idx) tuples — the range of token
    indices that reconstruct the target surface form.
    """
    ids = sp.encode(text)
    pieces = [sp.id_to_piece(i) for i in ids]

    # Reconstruct character offsets for each token
    # SentencePiece uses \u2581 for space
    reconstructed = ""
    token_char_spans = []
    for idx, piece in enumerate(pieces):
        display = piece.replace('\u2581', ' ')
        # Byte tokens like <0x6D> represent single bytes
        if piece.startswith('<0x') and piece.endswith('>'):
            byte_val = int(piece[3:-1], 16)
            try:
                display = bytes([byte_val]).decode('utf-8', errors='replace')
            except:
                display = '?'
        start = len(reconstructed)
        reconstructed += display
        end = len(reconstructed)
        token_char_spans.append((start, end))

    positions = {}
    for name, surface in target_tokens.items():
        # Find the surface form in the reconstructed text
        # Try with and without leading space
        for search in [' ' + surface, surface]:
            char_idx = reconstructed.find(search)
            if char_idx >= 0:
                char_start = char_idx
                char_end = char_idx + len(search)
                # Find which tokens span this character range
                tok_start = None
                tok_end = None
                for tidx, (cs, ce) in enumerate(token_char_spans):
                    if cs < char_end and ce > char_start:
                        if tok_start is None:
                            tok_start = tidx
                        tok_end = tidx + 1
                if tok_start is not None:
                    # Use the LAST token in the span as the representative position
                    # (that's where the full word meaning has been assembled)
                    positions[name] = {
                        "span": (tok_start, tok_end),
                        "repr": tok_end - 1,  # last token = most assembled
                        "tokens": pieces[tok_start:tok_end],
                    }
                break

    return ids, pieces, positions


@torch.no_grad()
def measure_attention(model, sp, text, target_tokens, device="cuda"):
    """
    Run text through model, capture attention, and extract attention weights
    between specified token positions.
    """
    ids, pieces, positions = tokenize_and_locate(sp, text, target_tokens)

    missing = [k for k in target_tokens if k not in positions]
    if missing:
        return None, pieces, positions, f"Missing positions for: {missing}"

    # Reset rotary caches to handle variable sequence lengths
    for block in model.blocks:
        block.attn.rotary._seq_len_cached = 0

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits, all_attn = model.forward_with_attention(input_ids)

    # all_attn is list of [1, num_heads, seq_len, seq_len] per layer
    return all_attn, pieces, positions, None


def run_lensing_probe(model, sp, device="cuda"):
    """Run the full gravitational lensing experiment."""

    # Load leverage scores to identify high/low gravity tokens
    scored_path = os.path.join(DATA_DIR, "candidates_scored.jsonl")
    leverage_map = {}
    if os.path.exists(scored_path):
        with open(scored_path, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if c.get("ablation_leverage", 0) != 0:
                    leverage_map[c["readable"]] = c["ablation_leverage"]

    print(f"Loaded {len(leverage_map)} leverage scores")
    if leverage_map:
        top = sorted(leverage_map.items(), key=lambda x: -x[1])[:10]
        bottom = sorted(leverage_map.items(), key=lambda x: x[1])[:10]
        print(f"\nHighest gravity tokens:")
        for tok, lev in top:
            print(f"  {lev:+.3f}  {tok!r}")
        print(f"\nLowest gravity tokens:")
        for tok, lev in bottom:
            print(f"  {lev:+.3f}  {tok!r}")

    # === EXPERIMENT SENTENCES ===
    # Base: clean subject-verb-object with measurable attention path
    # Massive: insert high-gravity token between S and O
    # Control: insert low-gravity token between S and O

    experiments = [
        {
            "name": "Negation lensing",
            "base": "The government announced the policy",
            "massive": "The government not announced the policy",
            "control": "The government or announced the policy",
            "subject": "government",
            "object": "policy",
            "lens": "not",
            "ctrl": "or",
        },
        {
            "name": "Conditional lensing",
            "base": "The system produced the result",
            "massive": "The system if produced the result",
            "control": "The system an produced the result",
            "subject": "system",
            "object": "result",
            "lens": "if",
            "ctrl": "an",
        },
        {
            "name": "Causal lensing",
            "base": "The water caused the damage",
            "massive": "The water because caused the damage",
            "control": "The water so caused the damage",
            "subject": "water",
            "object": "damage",
            "lens": "because",
            "ctrl": "so",
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"{'='*70}")

        for condition, label in [
            ("base", "BASE (no insertion)"),
            ("massive", f"MASSIVE ({exp['lens']})"),
            ("control", f"CONTROL ({exp['ctrl']})")
        ]:
            text = exp[condition]
            print(f"\n--- {label} ---")
            print(f"Text: {text!r}")

            # Identify target tokens to locate
            targets = {"subject": exp["subject"], "object": exp["object"]}
            if condition == "massive":
                targets["lens"] = exp["lens"]
            elif condition == "control":
                targets["lens"] = exp["ctrl"]

            all_attn, pieces, positions, err = measure_attention(
                model, sp, text, targets, device
            )

            print(f"Tokens: {pieces}")
            print(f"Positions: {positions}")

            if err:
                print(f"ERROR: {err}")
                continue

            subj_pos = positions["subject"]["repr"]
            obj_pos = positions["object"]["repr"]
            lens_info = positions.get("lens", None)
            lens_pos = lens_info["repr"] if lens_info else None

            print(f"Subject '{exp['subject']}': tokens {positions['subject']['tokens']} -> repr pos {subj_pos}")
            print(f"Object '{exp['object']}': tokens {positions['object']['tokens']} -> repr pos {obj_pos}")
            if lens_info:
                lens_word = exp.get('lens', exp.get('ctrl', '?'))
                print(f"Lens '{lens_word}': tokens {lens_info['tokens']} -> repr pos {lens_pos}")

            # Attention from object -> subject (the direct geodesic)
            # Average across all layers and heads
            direct_attn_per_layer = []
            lens_attn_per_layer = []  # object -> lens, lens -> subject

            for layer_idx, attn in enumerate(all_attn):
                # attn shape: [1, num_heads, seq_len, seq_len]
                # attn[0, head, query_pos, key_pos] = how much query_pos attends to key_pos
                a = attn[0]  # [num_heads, seq_len, seq_len]

                # Direct: object attends to subject
                direct = a[:, obj_pos, subj_pos].mean().item()
                direct_attn_per_layer.append(direct)

                if lens_pos is not None:
                    # Lensed path: object -> lens + lens -> subject
                    obj_to_lens = a[:, obj_pos, lens_pos].mean().item()
                    lens_to_subj = a[:, lens_pos, subj_pos].mean().item()
                    lens_attn_per_layer.append((obj_to_lens, lens_to_subj))

            mean_direct = np.mean(direct_attn_per_layer)
            print(f"\nA(object->subject) mean across layers: {mean_direct:.4f}")
            print(f"A(object->subject) per layer: {['%.4f' % x for x in direct_attn_per_layer]}")

            if lens_pos is not None and lens_attn_per_layer:
                obj_lens = [x[0] for x in lens_attn_per_layer]
                lens_subj = [x[1] for x in lens_attn_per_layer]
                print(f"A(object->lens) mean: {np.mean(obj_lens):.4f}")
                print(f"A(lens->subject) mean: {np.mean(lens_subj):.4f}")
                print(f"Indirect path product mean: {np.mean([a*b for a,b in lens_attn_per_layer]):.6f}")

            result = {
                "experiment": exp["name"],
                "condition": condition,
                "label": label,
                "text": text,
                "pieces": pieces,
                "positions": positions,
                "direct_attn_mean": mean_direct,
                "direct_attn_per_layer": direct_attn_per_layer,
            }
            if lens_pos is not None and lens_attn_per_layer:
                result["obj_to_lens_mean"] = np.mean(obj_lens)
                result["lens_to_subj_mean"] = np.mean(lens_subj)
                result["obj_to_lens_per_layer"] = obj_lens
                result["lens_to_subj_per_layer"] = lens_subj

            # Gravity scores if available
            if condition == "massive" and exp["lens"] in leverage_map:
                result["lens_leverage"] = leverage_map[exp["lens"]]
                print(f"Lens token gravity: {leverage_map[exp['lens']]:.3f}")
            if condition == "control" and exp["ctrl"] in leverage_map:
                result["lens_leverage"] = leverage_map[exp["ctrl"]]
                print(f"Control token gravity: {leverage_map[exp['ctrl']]:.3f}")

            results.append(result)

        # Summary comparison
        base_r = [r for r in results if r["experiment"] == exp["name"] and r["condition"] == "base"]
        mass_r = [r for r in results if r["experiment"] == exp["name"] and r["condition"] == "massive"]
        ctrl_r = [r for r in results if r["experiment"] == exp["name"] and r["condition"] == "control"]

        if base_r and mass_r and ctrl_r:
            base_direct = base_r[0]["direct_attn_mean"]
            mass_direct = mass_r[0]["direct_attn_mean"]
            ctrl_direct = ctrl_r[0]["direct_attn_mean"]

            print(f"\n{'─'*50}")
            print(f"LENSING SUMMARY: {exp['name']}")
            print(f"{'─'*50}")
            print(f"Direct A(obj->subj) BASE:    {base_direct:.4f}")
            print(f"Direct A(obj->subj) MASSIVE: {mass_direct:.4f}  (delta: {mass_direct - base_direct:+.4f})")
            print(f"Direct A(obj->subj) CONTROL: {ctrl_direct:.4f}  (delta: {ctrl_direct - base_direct:+.4f})")

            if mass_direct < base_direct and mass_direct < ctrl_direct:
                deflection_ratio = (base_direct - mass_direct) / max(base_direct - ctrl_direct, 1e-8)
                print(f"\n** LENSING DETECTED ** Massive token deflects {deflection_ratio:.1f}x more than control")
            elif mass_direct < base_direct:
                print(f"\nDeflection present but not stronger than control.")
            else:
                print(f"\nNo deflection detected.")

    # Save results
    output_path = os.path.join(DATA_DIR, "lensing_probe_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = json.loads(json.dumps(results, default=convert))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    # Fix Windows console encoding
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default to the fully trained 12L seed 137 checkpoint
    checkpoint = os.path.join(GOLF_ROOT, "logs", "gravity_12L_seed137.int8.ptz")
    if not os.path.exists(checkpoint):
        # Fall back to smoke test checkpoint
        checkpoint = os.path.join(GOLF_ROOT, "final_model.int8.ptz")
        num_layers = 13
        print("WARNING: Using 13L smoke test checkpoint (100 steps)")
    else:
        num_layers = 12

    tokenizer = os.path.join(DATA_DIR, "tokenizers", "gravity_beta_1.0.model")

    print(f"Loading model ({num_layers}L)...")
    model, sp = load_model(checkpoint, tokenizer, num_layers=num_layers, device=device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")

    results = run_lensing_probe(model, sp, device)
