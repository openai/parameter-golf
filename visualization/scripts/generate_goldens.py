#!/usr/bin/env python3
"""Generate parity-check goldens for the visualization browser demo.

For a fixed text input (the Transformer abstract), this script produces
deterministic, byte-reproducible reference outputs for each model so the
in-browser implementation can verify it matches Python exactly.

Stages (run incrementally, each writes its own files):

  1. tokenize     -> tokens.json
  2. masks        -> masks.bin   (K x n_tokens packed bools)
  3. mdlm         -> mdlm_logits.bin, mdlm_topk.json, mdlm_per_token_nelbo.json,
                     mdlm_per_byte_bpb.json
  4. ar           -> (deferred until AR artifact is available locally)

Usage:
  python generate_goldens.py tokenize
  python generate_goldens.py masks --K 64
  python generate_goldens.py mdlm  --artifact artifacts/diffusion-mdlm-9L-512d-0d2984.ptz
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the model definitions and artifact loaders from bpb_compare.py
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "text_diffusion"))
import sentencepiece as spm
from bpb_compare import (  # noqa: E402
    DiffusionTransformer,
    GPT,
    apply_rotary_emb,
    detect_architecture,
    load_artifact,
    sp_token_byte_lengths,
)

GOLDENS_DIR = Path(__file__).resolve().parent.parent / "goldens"
TEXT_PATH = GOLDENS_DIR / "transformer_abstract.txt"
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"


# ---------------------------------------------------------------------------
# Stage 1: tokenize
# ---------------------------------------------------------------------------

def stage_tokenize() -> None:
    text = TEXT_PATH.read_text(encoding="utf-8")
    text_bytes = text.encode("utf-8")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    token_ids = sp.encode_as_ids(text)

    pieces = [sp.id_to_piece(t) for t in token_ids]
    byte_lens = sp_token_byte_lengths(sp, token_ids)

    # Cumulative byte offset for each token (start position in UTF-8 byte stream).
    # This is what the visualization needs to map per-token bits onto byte cells.
    byte_starts: list[int] = []
    cursor = 0
    for n in byte_lens:
        byte_starts.append(cursor)
        cursor += n

    out = {
        "text_n_bytes": len(text_bytes),
        "n_tokens": len(token_ids),
        "vocab_size": sp.get_piece_size(),
        "token_ids": token_ids,
        "pieces": pieces,
        "byte_lens": byte_lens,
        "byte_starts": byte_starts,
        "covered_bytes": cursor,
    }
    out_path = GOLDENS_DIR / "tokens.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(f"  text bytes:   {len(text_bytes)}")
    print(f"  tokens:       {len(token_ids)}")
    print(f"  vocab_size:   {sp.get_piece_size()}")
    print(f"  covered bytes: {cursor}  (matches text_n_bytes: {cursor == len(text_bytes)})")
    print(f"  first 8 pieces: {pieces[:8]!r}")


# ---------------------------------------------------------------------------
# Stage 2: deterministic masks
# ---------------------------------------------------------------------------
#
# The original per_byte_bits_diffusion uses torch.rand for masks and a special
# rule that forces position 0 to be masked when t < 0.01.  We materialise that
# logic here once with a fixed seed so both Python and the JS port consume the
# same K x n_tokens bool tensor.
#
# File format (masks.bin):
#   header (24 bytes):  magic "PGMASK\0\0" (8B), uint32 K, uint32 n_tokens,
#                       float32 eps, float32 _reserved
#   t_values:           K float32 values (linspace(eps, 1.0, K))
#   masks:              K * n_tokens uint8 (one byte per bool — simple & cache-friendly)

MASK_MAGIC = b"PGMASK\0\0"


def stage_masks(K: int, eps: float, seed: int) -> None:
    tokens = json.loads((GOLDENS_DIR / "tokens.json").read_text())
    n = tokens["n_tokens"]

    g = torch.Generator().manual_seed(seed)
    t_values = torch.linspace(eps, 1.0, K)
    masks = torch.zeros((K, n), dtype=torch.bool)
    for k, t in enumerate(t_values):
        tv = float(t.item())
        masks[k] = torch.rand(n, generator=g) < tv
        if tv < 0.01:
            masks[k, 0] = True

    out_path = GOLDENS_DIR / "masks.bin"
    with out_path.open("wb") as f:
        f.write(MASK_MAGIC)
        f.write(struct.pack("<II", K, n))
        f.write(struct.pack("<ff", float(eps), 0.0))
        f.write(t_values.numpy().astype(np.float32).tobytes())
        f.write(masks.numpy().astype(np.uint8).tobytes())

    n_set = int(masks.sum().item())
    print(f"wrote {out_path}  ({out_path.stat().st_size} bytes)")
    print(f"  K:        {K}")
    print(f"  n_tokens: {n}")
    print(f"  eps:      {eps}")
    print(f"  seed:     {seed}")
    print(f"  total masked positions: {n_set} / {K * n}  ({n_set / (K * n):.3f})")
    print(f"  first row mask sum:    {int(masks[0].sum().item())} / {n}  (t={t_values[0]:.4f})")
    print(f"  last  row mask sum:    {int(masks[-1].sum().item())} / {n}  (t={t_values[-1]:.4f})")


def load_masks() -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load masks.bin → (t_values [K], masks [K,n] bool, eps)."""
    raw = (GOLDENS_DIR / "masks.bin").read_bytes()
    assert raw[:8] == MASK_MAGIC, f"bad magic: {raw[:8]!r}"
    K, n = struct.unpack("<II", raw[8:16])
    eps, _ = struct.unpack("<ff", raw[16:24])
    cursor = 24
    t_values = np.frombuffer(raw[cursor:cursor + 4 * K], dtype=np.float32).copy()
    cursor += 4 * K
    masks = np.frombuffer(raw[cursor:cursor + K * n], dtype=np.uint8).reshape(K, n).copy()
    return torch.from_numpy(t_values), torch.from_numpy(masks).bool(), float(eps)


# ---------------------------------------------------------------------------
# Stage 3: MDLM forward
# ---------------------------------------------------------------------------

def build_diffusion_model(artifact_path: str) -> tuple[DiffusionTransformer, str]:
    sd = load_artifact(artifact_path)
    arch = detect_architecture(sd)
    assert arch["is_diffusion"], "expected a diffusion artifact"
    model = DiffusionTransformer(
        vocab_size=1024,
        num_layers=arch["num_layers"],
        model_dim=arch["model_dim"],
        num_heads=8, num_kv_heads=4, mlp_mult=2,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    model.load_state_dict(sd, strict=False)
    model.float()
    model.eval()
    label = f"MDLM {arch['num_layers']}L-{arch['model_dim']}d"
    return model, label


@torch.inference_mode()
def stage_mdlm(artifact_path: str, top_k: int, save_full_logits: bool) -> None:
    tokens = json.loads((GOLDENS_DIR / "tokens.json").read_text())
    token_ids = tokens["token_ids"]
    n = tokens["n_tokens"]
    n_bytes = tokens["text_n_bytes"]

    t_values, masks, eps = load_masks()
    K = t_values.shape[0]
    assert masks.shape == (K, n), f"mask shape {masks.shape} != ({K},{n})"

    model, label = build_diffusion_model(artifact_path)
    print(f"loaded {label} from {artifact_path}")

    ids = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0)  # [1, n]
    V = 1024

    topk_path = GOLDENS_DIR / "mdlm_topk.json"
    nelbo_path = GOLDENS_DIR / "mdlm_per_token_nelbo.json"
    bpb_path = GOLDENS_DIR / "mdlm_per_byte_bpb.json"
    logits_path = GOLDENS_DIR / "mdlm_logits.bin"

    fout = None
    if save_full_logits:
        # Layout: magic "PGLOGIT\0" (8B), uint32 K, uint32 n, uint32 V, uint32 _reserved,
        #         K*n*V float32, row-major [k, pos, vocab]
        LOGIT_MAGIC = b"PGLOGIT\0"
        fout = logits_path.open("wb")
        fout.write(LOGIT_MAGIC)
        fout.write(struct.pack("<IIII", K, n, V, 0))

    token_nelbo = np.zeros(n, dtype=np.float64)
    scale = (1 - eps) / K

    topk_per_pass: list[dict] = []

    for k in range(K):
        tv = float(t_values[k].item())
        mask_row = masks[k:k + 1]  # [1, n]
        logits = model.forward_logits(ids, mask_row)  # [1, n, V]
        logits_np = logits[0].float().numpy()
        if fout is not None:
            fout.write(logits_np.astype(np.float32).tobytes())

        log_probs = F.log_softmax(logits[0].float(), dim=-1)
        nll = -log_probs.gather(-1, ids[0].unsqueeze(-1)).squeeze(-1).numpy()  # [n]

        masked_positions = mask_row[0].nonzero(as_tuple=True)[0].tolist()
        weight = scale / tv
        for pos in masked_positions:
            token_nelbo[pos] += weight * float(nll[pos])

        # Top-K capture for masked positions only
        if masked_positions:
            top_vals, top_ids = torch.topk(logits[0], k=top_k, dim=-1)  # [n, top_k]
            top_vals_np = top_vals.float().numpy()
            top_ids_np = top_ids.numpy()
            entries = []
            for pos in masked_positions:
                entries.append({
                    "pos": int(pos),
                    "ids": top_ids_np[pos].tolist(),
                    "values": [float(v) for v in top_vals_np[pos]],
                    "true_id": int(token_ids[pos]),
                    "true_logit": float(logits_np[pos, token_ids[pos]]),
                    "nll_nats": float(nll[pos]),
                })
            topk_per_pass.append({"k": k, "t": tv, "entries": entries})

        if k < 3 or k == K - 1:
            print(f"  pass k={k:2d}  t={tv:.4f}  masked={int(mask_row.sum().item()):3d}/{n}  "
                  f"sum_nll={float(nll[mask_row[0]].sum()):.4f}")

    if fout is not None:
        fout.close()
        print(f"wrote {logits_path}  ({logits_path.stat().st_size:,} bytes)")

    # Per-token bits and per-byte BPB
    token_bits = (token_nelbo / math.log(2)).tolist()
    byte_lens = tokens["byte_lens"]
    per_byte = []
    for bits, blen in zip(token_bits, byte_lens):
        if blen > 0:
            per_byte.extend([bits / blen] * blen)
    if len(per_byte) < n_bytes:
        per_byte.extend([0.0] * (n_bytes - len(per_byte)))
    per_byte = per_byte[:n_bytes]
    bpb = sum(per_byte) / n_bytes if n_bytes > 0 else 0.0

    nelbo_path.write_text(json.dumps({
        "label": label,
        "n_tokens": n,
        "token_nelbo_nats": token_nelbo.tolist(),
        "token_bits": token_bits,
    }, indent=2))
    bpb_path.write_text(json.dumps({
        "label": label,
        "bpb": bpb,
        "n_bytes": n_bytes,
        "per_byte": per_byte,
    }, indent=2))
    topk_path.write_text(json.dumps({
        "label": label,
        "top_k": top_k,
        "passes": topk_per_pass,
    }))

    print(f"wrote {nelbo_path}")
    print(f"wrote {bpb_path}")
    print(f"wrote {topk_path}  ({topk_path.stat().st_size:,} bytes)")
    print()
    print(f"MDLM BPB on Transformer abstract: {bpb:.4f}")


# ---------------------------------------------------------------------------
# Stage 3b: AR forward
# ---------------------------------------------------------------------------
#
# The AR baseline (`GPT` in bpb_compare.py) is a causal LM. The eval recipe is
# straightforward: feed all tokens, take per-position softmax of logits[i] to
# predict token i+1. We assign 10 bits (log2(vocab=1024)) to position 0 because
# there is no left context — same as `per_byte_bits_ar` in bpb_compare.py.

def build_ar_model(artifact_path: str) -> tuple[GPT, str]:
    sd = load_artifact(artifact_path)
    arch = detect_architecture(sd)
    assert not arch["is_diffusion"], "expected an AR (causal) artifact"
    model = GPT(
        vocab_size=1024,
        num_layers=arch["num_layers"],
        model_dim=arch["model_dim"],
        num_heads=8, num_kv_heads=4, mlp_mult=2,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    model.load_state_dict(sd, strict=False)
    model.float()
    model.eval()
    label = f"AR {arch['num_layers']}L-{arch['model_dim']}d"
    return model, label


@torch.inference_mode()
def stage_ar(artifact_path: str, top_k: int, save_full_logits: bool) -> None:
    tokens = json.loads((GOLDENS_DIR / "tokens.json").read_text())
    token_ids = tokens["token_ids"]
    n = tokens["n_tokens"]
    n_bytes = tokens["text_n_bytes"]

    model, label = build_ar_model(artifact_path)
    print(f"loaded {label} from {artifact_path}")

    # Match bpb_compare.per_byte_bits_ar exactly: feed tokens[:n-1] to predict
    # tokens[1:n]. The very last token has no successor; the very first token
    # has no predecessor. Forwarding only n-1 tokens (vs n) makes our logits
    # bit-identical to bpb_compare even though SDPA's reduction order differs
    # between seq_len=n and seq_len=n-1.
    ids_input = torch.tensor(token_ids[:n - 1], dtype=torch.int64).unsqueeze(0)  # [1, n-1]
    V = 1024
    logits = model.forward_logits(ids_input)  # [1, n-1, V]
    logits_np = logits[0].float().numpy()  # [n-1, V]

    # NLL: position i in {1..n-1} is predicted from prefix [0..i-1] using logits[i-1]
    # Match bpb_compare exactly: do `nll / log(2)` on a fp32 tensor then `.tolist()`,
    # rather than per-element `.item() / math.log(2)` (which would cast to fp64
    # before the divide and drift by ~1 fp32 ULP per byte).
    log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [n-1, V]
    targets = torch.tensor(token_ids[1:], dtype=torch.int64)  # [n-1]
    nll_tensor = -log_probs[torch.arange(n - 1), targets]      # [n-1] fp32
    token_bits: list[float] = [math.log2(V)] + (nll_tensor / math.log(2)).tolist()
    nll_per_token: list[float] = [math.log2(V)] + nll_tensor.tolist()

    # Top-K per position. Position i's top-K is computed from logits[i-1]
    # (the predictions FOR position i). Position 0 has no predictor — emit None.
    top_vals, top_ids = torch.topk(logits[0], k=top_k, dim=-1)  # [n-1, top_k]
    top_vals_np = top_vals.float().numpy()
    top_ids_np = top_ids.numpy()
    topk: list[dict] = []
    for i in range(n):
        if i == 0:
            topk.append({"pos": 0, "predicted_from": None,
                         "true_id": int(token_ids[0]),
                         "ids": None, "values": None, "bits": math.log2(V)})
        else:
            row = i - 1
            topk.append({
                "pos": i,
                "predicted_from": row,  # logits[row] predicts token at pos i
                "true_id": int(token_ids[i]),
                "true_logit": float(logits_np[row, token_ids[i]]),
                "ids": top_ids_np[row].tolist(),
                "values": [float(v) for v in top_vals_np[row]],
                "bits": token_bits[i],
            })

    # Per-byte spread (same recipe as bpb_compare.spread_token_bits_to_bytes)
    byte_lens = tokens["byte_lens"]
    per_byte: list[float] = []
    for bits, blen in zip(token_bits, byte_lens):
        if blen > 0:
            per_byte.extend([bits / blen] * blen)
    if len(per_byte) < n_bytes:
        per_byte.extend([0.0] * (n_bytes - len(per_byte)))
    per_byte = per_byte[:n_bytes]
    bpb = sum(per_byte) / n_bytes if n_bytes > 0 else 0.0

    # Save
    (GOLDENS_DIR / "ar_topk.json").write_text(json.dumps({
        "label": label, "top_k": top_k, "tokens": topk,
    }))
    (GOLDENS_DIR / "ar_per_token_bits.json").write_text(json.dumps({
        "label": label, "n_tokens": n,
        "token_bits": token_bits,
        "token_nll_nats": nll_per_token,
    }, indent=2))
    (GOLDENS_DIR / "ar_per_byte_bpb.json").write_text(json.dumps({
        "label": label, "bpb": bpb, "n_bytes": n_bytes, "per_byte": per_byte,
    }, indent=2))

    if save_full_logits:
        # Layout: magic "PGARLOG\0" (8B), uint32 n_pred (=n-1), uint32 V, uint32 _,
        #         (n-1)*V float32. Row i predicts token at position (i+1).
        LOGIT_MAGIC = b"PGARLOG\0"
        out = (GOLDENS_DIR / "ar_logits.bin").open("wb")
        out.write(LOGIT_MAGIC)
        out.write(struct.pack("<III", n - 1, V, 0))
        out.write(logits_np.astype(np.float32).tobytes())
        out.close()
        print(f"wrote {GOLDENS_DIR / 'ar_logits.bin'}  "
              f"({(GOLDENS_DIR / 'ar_logits.bin').stat().st_size:,} bytes)")

    print(f"wrote ar_topk.json, ar_per_token_bits.json, ar_per_byte_bpb.json")
    print()
    print(f"AR BPB on Transformer abstract: {bpb:.4f}")


# ---------------------------------------------------------------------------
# Stage 3c: AR trace (causal version of the diffusion trace)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def traced_ar_forward(model: GPT, input_ids: torch.Tensor
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mirror of GPT.forward_logits with intermediate capture (causal=True)."""
    out: dict[str, torch.Tensor] = {}

    def cap(name: str, t: torch.Tensor) -> None:
        out[name] = t.detach().float().clone().contiguous()

    cap("input_ids", input_ids.float())
    x = model.tok_emb(input_ids)
    cap("tok_emb_out", x)
    x = F.rms_norm(x, (x.size(-1),))
    cap("init_rmsnorm_out", x)
    x0 = x
    cap("x0", x0)

    skips: list[torch.Tensor] = []

    def run_block(block, x_in, x0_in, prefix):
        mix = block.resid_mix.to(dtype=x_in.dtype)
        x_mixed = mix[0][None, None, :] * x_in + mix[1][None, None, :] * x0_in
        cap(f"{prefix}/resid_mix_out", x_mixed)
        x_attn_norm = block.attn_norm(x_mixed)
        cap(f"{prefix}/attn_norm_out", x_attn_norm)
        attn = block.attn
        bsz, seqlen, dim = x_attn_norm.shape
        q = attn.c_q(x_attn_norm).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.c_k(x_attn_norm).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = attn.c_v(x_attn_norm).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        cap(f"{prefix}/q_proj", q); cap(f"{prefix}/k_proj", k); cap(f"{prefix}/v_proj", v)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cap(f"{prefix}/q_after_rmsnorm", q); cap(f"{prefix}/k_after_rmsnorm", k)
        cos, sin = attn.rotary(seqlen, x_attn_norm.device, q.dtype)
        cap(f"{prefix}/rope_cos", cos); cap(f"{prefix}/rope_sin", sin)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        cap(f"{prefix}/q_after_rotary", q); cap(f"{prefix}/k_after_rotary", k)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, :, None, None]
        cap(f"{prefix}/q_after_gain", q)
        attn_y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=attn.is_causal,
            enable_gqa=(attn.num_kv_heads != attn.num_heads),
        )
        cap(f"{prefix}/attn_sdpa_out", attn_y)
        attn_y_flat = attn_y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        cap(f"{prefix}/attn_sdpa_flat", attn_y_flat)
        attn_proj = attn.proj(attn_y_flat)
        cap(f"{prefix}/attn_proj_out", attn_proj)
        x_after_attn = x_mixed + block.attn_scale.to(dtype=x_mixed.dtype)[None, None, :] * attn_proj
        cap(f"{prefix}/post_attn_residual", x_after_attn)
        x_mlp_norm = block.mlp_norm(x_after_attn)
        cap(f"{prefix}/mlp_norm_out", x_mlp_norm)
        mlp_fc = block.mlp.fc(x_mlp_norm)
        cap(f"{prefix}/mlp_fc_out", mlp_fc)
        mlp_act = torch.relu(mlp_fc).square()
        cap(f"{prefix}/mlp_relu_sq", mlp_act)
        mlp_proj = block.mlp.proj(mlp_act)
        cap(f"{prefix}/mlp_proj_out", mlp_proj)
        x_out = x_after_attn + block.mlp_scale.to(dtype=x_after_attn.dtype)[None, None, :] * mlp_proj
        cap(f"{prefix}/block_out", x_out)
        return x_out

    for i in range(model.num_encoder_layers):
        x = run_block(model.blocks[i], x, x0, f"enc{i:02d}")
        skips.append(x)
        cap(f"enc{i:02d}/skip_pushed", x)

    for i in range(model.num_decoder_layers):
        if skips:
            sk = skips.pop()
            cap(f"dec{i:02d}/skip_popped", sk)
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * sk
            cap(f"dec{i:02d}/post_skip_add", x)
        x = run_block(model.blocks[model.num_encoder_layers + i], x, x0, f"dec{i:02d}")

    x = model.final_norm(x)
    cap("final_norm_out", x)
    logits_pre = F.linear(x, model.tok_emb.weight)
    cap("logits_pre_softcap", logits_pre)
    logits = model.logit_softcap * torch.tanh(logits_pre / model.logit_softcap)
    cap("logits", logits)
    return logits, out


@torch.inference_mode()
def stage_ar_trace(artifact_path: str) -> None:
    tokens = json.loads((GOLDENS_DIR / "tokens.json").read_text())
    token_ids = tokens["token_ids"]

    model, label = build_ar_model(artifact_path)
    print(f"model: {label}")
    # Match stage_ar (n-1 inputs predict positions 1..n-1).
    n = len(token_ids)
    ids = torch.tensor(token_ids[:n - 1], dtype=torch.int64).unsqueeze(0)

    logits_ref = model.forward_logits(ids).float()
    logits_traced, intermediates = traced_ar_forward(model, ids)
    diff = (logits_ref - logits_traced).abs().max().item()
    print(f"trace self-check max abs diff vs forward_logits: {diff:.3e}")
    assert diff == 0.0, f"trace diverged from forward_logits by {diff}"

    bin_path = GOLDENS_DIR / "ar_trace.bin"
    json_path = GOLDENS_DIR / "ar_trace.json"
    manifest = {"label": label, "tensors": []}
    offset = 0
    with bin_path.open("wb") as f:
        for name, t in intermediates.items():
            arr = t.cpu().numpy().astype(np.float32, copy=False)
            n_elems = int(arr.size)
            f.write(arr.tobytes())
            manifest["tensors"].append({
                "name": name, "shape": list(arr.shape),
                "dtype": "float32", "offset": offset, "nelems": n_elems,
            })
            offset += n_elems * 4
    json_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {bin_path}  ({bin_path.stat().st_size:,} bytes, "
          f"{len(manifest['tensors'])} tensors)")
    print(f"wrote {json_path}")


# ---------------------------------------------------------------------------
# Stage 4: trace one MDLM forward pass with every intermediate activation
# ---------------------------------------------------------------------------
#
# This is the parity sledgehammer: we manually replay DiffusionTransformer
# .forward_logits for one chosen mask k, capturing every tensor between ops.
# The JS port can load these and assert each intermediate matches before the
# final logits — that way any divergence is localised to a single op.
#
# We also assert (in this very script) that the traced reproduction's final
# logits are bit-identical to the original forward_logits — a self-check that
# this trace function hasn't drifted from the model class.
#
# File layout (mdlm_trace_k{k}.bin / .json):
#   .bin  =  concatenated raw fp32, little-endian, in the order listed in .json
#   .json =  {"k": int, "tensors": [{"name": str, "shape": [..], "offset": int,
#                                    "nelems": int}, ...]}


@torch.inference_mode()
def traced_diffusion_forward(model: DiffusionTransformer,
                             input_ids: torch.Tensor,
                             mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mirror of DiffusionTransformer.forward_logits with intermediate capture.

    Returns (final_logits, {name: tensor}). All captured tensors are detached
    fp32 clones. The recipe must remain a line-for-line mirror of the original
    method; we self-check the final logits below.
    """
    out: dict[str, torch.Tensor] = {}

    def cap(name: str, t: torch.Tensor) -> None:
        out[name] = t.detach().float().clone().contiguous()

    # --- Input embedding & initial RMSNorm ---
    masked_ids = torch.where(mask, model.mask_id, input_ids)
    cap("masked_ids", masked_ids.float())
    x = model.tok_emb(masked_ids)
    cap("tok_emb_out", x)
    x = F.rms_norm(x, (x.size(-1),))
    cap("init_rmsnorm_out", x)
    x0 = x
    cap("x0", x0)

    skips: list[torch.Tensor] = []

    def run_block(block, x_in: torch.Tensor, x0_in: torch.Tensor, prefix: str) -> torch.Tensor:
        # === residual mix ===
        mix = block.resid_mix.to(dtype=x_in.dtype)
        x_mixed = mix[0][None, None, :] * x_in + mix[1][None, None, :] * x0_in
        cap(f"{prefix}/resid_mix_out", x_mixed)

        # === attention ===
        x_attn_norm = block.attn_norm(x_mixed)
        cap(f"{prefix}/attn_norm_out", x_attn_norm)

        attn = block.attn
        bsz, seqlen, dim = x_attn_norm.shape
        q = attn.c_q(x_attn_norm).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.c_k(x_attn_norm).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = attn.c_v(x_attn_norm).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        cap(f"{prefix}/q_proj", q); cap(f"{prefix}/k_proj", k); cap(f"{prefix}/v_proj", v)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cap(f"{prefix}/q_after_rmsnorm", q); cap(f"{prefix}/k_after_rmsnorm", k)
        cos, sin = attn.rotary(seqlen, x_attn_norm.device, q.dtype)
        # cos/sin are shared across all blocks but easy to capture once per block for sanity
        cap(f"{prefix}/rope_cos", cos); cap(f"{prefix}/rope_sin", sin)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        cap(f"{prefix}/q_after_rotary", q); cap(f"{prefix}/k_after_rotary", k)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, :, None, None]
        cap(f"{prefix}/q_after_gain", q)
        attn_y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=attn.is_causal,
            enable_gqa=(attn.num_kv_heads != attn.num_heads),
        )
        cap(f"{prefix}/attn_sdpa_out", attn_y)
        attn_y_flat = attn_y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        cap(f"{prefix}/attn_sdpa_flat", attn_y_flat)
        attn_proj = attn.proj(attn_y_flat)
        cap(f"{prefix}/attn_proj_out", attn_proj)

        x_after_attn = x_mixed + block.attn_scale.to(dtype=x_mixed.dtype)[None, None, :] * attn_proj
        cap(f"{prefix}/post_attn_residual", x_after_attn)

        # === MLP ===
        x_mlp_norm = block.mlp_norm(x_after_attn)
        cap(f"{prefix}/mlp_norm_out", x_mlp_norm)
        mlp_fc = block.mlp.fc(x_mlp_norm)
        cap(f"{prefix}/mlp_fc_out", mlp_fc)
        mlp_act = torch.relu(mlp_fc).square()
        cap(f"{prefix}/mlp_relu_sq", mlp_act)
        mlp_proj = block.mlp.proj(mlp_act)
        cap(f"{prefix}/mlp_proj_out", mlp_proj)

        x_out = x_after_attn + block.mlp_scale.to(dtype=x_after_attn.dtype)[None, None, :] * mlp_proj
        cap(f"{prefix}/block_out", x_out)
        return x_out

    # --- Encoder ---
    for i in range(model.num_encoder_layers):
        x = run_block(model.blocks[i], x, x0, f"enc{i:02d}")
        skips.append(x)
        cap(f"enc{i:02d}/skip_pushed", x)

    # --- Decoder ---
    for i in range(model.num_decoder_layers):
        if skips:
            sk = skips.pop()
            cap(f"dec{i:02d}/skip_popped", sk)
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * sk
            cap(f"dec{i:02d}/post_skip_add", x)
        x = run_block(model.blocks[model.num_encoder_layers + i], x, x0, f"dec{i:02d}")

    # --- Output head ---
    x = model.final_norm(x)
    cap("final_norm_out", x)
    logits_pre = F.linear(x, model.tok_emb.weight[:model.vocab_size])
    cap("logits_pre_softcap", logits_pre)
    logits = model.logit_softcap * torch.tanh(logits_pre / model.logit_softcap)
    cap("logits", logits)

    return logits, out


@torch.inference_mode()
def stage_trace(artifact_path: str, k: int) -> None:
    tokens = json.loads((GOLDENS_DIR / "tokens.json").read_text())
    token_ids = tokens["token_ids"]
    n = tokens["n_tokens"]

    t_values, masks, eps = load_masks()
    K = t_values.shape[0]
    assert 0 <= k < K, f"k={k} out of range [0,{K})"
    print(f"tracing pass k={k}  t={float(t_values[k]):.4f}  "
          f"masked={int(masks[k].sum().item())}/{n}")

    model, label = build_diffusion_model(artifact_path)
    print(f"model: {label}")

    ids = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0)
    mask_row = masks[k:k + 1]

    # Self-check: traced output must equal the model's own forward_logits
    logits_ref = model.forward_logits(ids, mask_row).float()
    logits_traced, intermediates = traced_diffusion_forward(model, ids, mask_row)
    diff = (logits_ref - logits_traced).abs().max().item()
    print(f"trace self-check max abs diff vs forward_logits: {diff:.3e}")
    assert diff == 0.0, f"trace diverged from forward_logits by {diff}"

    # Cross-check against the matching pass in mdlm_logits.bin (if present)
    logits_bin = GOLDENS_DIR / "mdlm_logits.bin"
    if logits_bin.exists():
        raw = logits_bin.read_bytes()
        assert raw[:8] == b"PGLOGIT\0"
        K_b, n_b, V_b, _ = struct.unpack("<IIII", raw[8:24])
        assert (K_b, n_b, V_b) == (K, n, 1024)
        offset = 24 + k * n * 1024 * 4
        ref = np.frombuffer(raw[offset:offset + n * 1024 * 4],
                            dtype=np.float32).reshape(n, 1024).copy()
        diff_bin = float(np.abs(ref - logits_traced[0].numpy()).max())
        print(f"trace vs mdlm_logits.bin[k={k}] max abs diff: {diff_bin:.3e}")
        assert diff_bin == 0.0

    # Write trace
    bin_path = GOLDENS_DIR / f"mdlm_trace_k{k:02d}.bin"
    json_path = GOLDENS_DIR / f"mdlm_trace_k{k:02d}.json"

    manifest = {"k": k, "t": float(t_values[k]), "label": label, "tensors": []}
    offset = 0
    with bin_path.open("wb") as f:
        for name, t in intermediates.items():
            arr = t.cpu().numpy().astype(np.float32, copy=False)
            n_elems = int(arr.size)
            f.write(arr.tobytes())
            manifest["tensors"].append({
                "name": name,
                "shape": list(arr.shape),
                "dtype": "float32",
                "offset": offset,
                "nelems": n_elems,
            })
            offset += n_elems * 4
    json_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {bin_path}  ({bin_path.stat().st_size:,} bytes, "
          f"{len(manifest['tensors'])} tensors)")
    print(f"wrote {json_path}")
    # Print a few highlights so we know what we have
    print("\nfirst 6 tensors:")
    for entry in manifest["tensors"][:6]:
        print(f"  {entry['name']:30s}  shape={entry['shape']}  nelems={entry['nelems']}")
    print(f"... ({len(manifest['tensors'])} total) ...")
    print("last 4 tensors:")
    for entry in manifest["tensors"][-4:]:
        print(f"  {entry['name']:30s}  shape={entry['shape']}  nelems={entry['nelems']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    sub.add_parser("tokenize")

    pm = sub.add_parser("masks")
    pm.add_argument("--K", type=int, default=64)
    pm.add_argument("--eps", type=float, default=1e-3)
    pm.add_argument("--seed", type=int, default=0xC0FFEE)

    pd = sub.add_parser("mdlm")
    pd.add_argument("--artifact", required=True)
    pd.add_argument("--top-k", type=int, default=10)
    pd.add_argument("--save-full-logits", action="store_true",
                    help="Also dump full K*n*V logits (~125MB) for byte-level diff")

    pt = sub.add_parser("trace")
    pt.add_argument("--artifact", required=True)
    pt.add_argument("--k", type=int, default=0,
                    help="Which mask index (0..K-1) to trace. Default 0 = smallest mask.")

    pa = sub.add_parser("ar")
    pa.add_argument("--artifact", required=True)
    pa.add_argument("--top-k", type=int, default=10)
    pa.add_argument("--save-full-logits", action="store_true",
                    help="Also dump full n*V AR logits for byte-level diff")

    pat = sub.add_parser("ar-trace")
    pat.add_argument("--artifact", required=True)

    args = p.parse_args()

    if args.stage == "tokenize":
        stage_tokenize()
    elif args.stage == "masks":
        stage_masks(args.K, args.eps, args.seed)
    elif args.stage == "mdlm":
        stage_mdlm(args.artifact, args.top_k, args.save_full_logits)
    elif args.stage == "trace":
        stage_trace(args.artifact, args.k)
    elif args.stage == "ar":
        stage_ar(args.artifact, args.top_k, args.save_full_logits)
    elif args.stage == "ar-trace":
        stage_ar_trace(args.artifact)


if __name__ == "__main__":
    main()
