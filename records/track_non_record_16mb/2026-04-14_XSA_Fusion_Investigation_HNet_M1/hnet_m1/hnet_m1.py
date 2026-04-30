"""H-Net Milestone 1 model: hierarchical byte-level stack with a FIXED chunker.

Architecture:
    bytes --> byte_emb (256 x D_enc)
          --> byte_encoder (2 blocks at D_enc=256)
          --> enc_to_main projection (D_enc -> D_main)
          --> fixed chunker: x[:, ::CHUNK_STRIDE, :]
          --> main_network (11 blocks at D_main=512)
          --> main_to_dec projection (D_main -> D_enc)
          --> upsampler: x.repeat_interleave(CHUNK_STRIDE, dim=1)  (truncate/pad to T)
          --> byte_decoder (1 block at D_enc=256)
          --> final_norm + byte_head (D_enc -> 256)
          --> per-byte logits

Reuses bigbag's Block / RMSNorm / CastedLinear / Rotary / apply_rotary_emb via
namespace injection from an already-exec'd baseline module.

No learned chunker. No ratio loss. No depth recurrence / parallel residuals in
M1 (can add for M2/M3). Plain AdamW on everything.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_hnet_m1(ns: dict, *, byte_seq_len: int, chunk_stride: int,
                  d_enc: int = 256, d_main: int = 512,
                  enc_num_layers: int = 2, main_num_layers: int = 11, dec_num_layers: int = 1,
                  enc_num_heads: int = 8, enc_num_kv_heads: int = 4, enc_mlp_mult: float = 3.0,
                  main_num_heads: int = 8, main_num_kv_heads: int = 4, main_mlp_mult: float = 4.0,
                  dec_num_heads: int = 8, dec_num_kv_heads: int = 4, dec_mlp_mult: float = 3.0,
                  rope_base: float = 10000.0,
                  enc_rope_dims: int = 16, main_rope_dims: int = 16, dec_rope_dims: int = 16,
                  enc_qk_gain_init: float = 1.5, main_qk_gain_init: float = 5.0, dec_qk_gain_init: float = 1.5):
    """Factory that builds an HNetM1 module using classes from baseline `ns`."""
    Block            = ns["Block"]
    RMSNorm          = ns["RMSNorm"]
    CastedLinear     = ns["CastedLinear"]
    Rotary           = ns["Rotary"]
    apply_rotary_emb = ns["apply_rotary_emb"]  # noqa: F841 (used inside Block)

    class HNetM1(nn.Module):
        def __init__(self):
            super().__init__()
            self.byte_seq_len = byte_seq_len
            self.chunk_stride = chunk_stride
            self.d_enc = d_enc
            self.d_main = d_main
            self.main_seq_len = (byte_seq_len + chunk_stride - 1) // chunk_stride

            # byte embedding (256 vocab)
            self.byte_emb = nn.Embedding(256, d_enc)
            nn.init.normal_(self.byte_emb.weight, mean=0.0, std=0.01)

            # byte encoder
            self.byte_encoder = nn.ModuleList([
                Block(dim=d_enc, num_heads=enc_num_heads, num_kv_heads=enc_num_kv_heads,
                      mlp_mult=enc_mlp_mult, rope_base=rope_base, qk_gain_init=enc_qk_gain_init,
                      train_seq_len=byte_seq_len, layer_idx=i, ln_scale=True)
                for i in range(enc_num_layers)
            ])
            for blk in self.byte_encoder:
                blk.attn.rope_dims = enc_rope_dims
                blk.attn.rotary = Rotary(d_enc // enc_num_heads, base=rope_base,
                                          train_seq_len=byte_seq_len, rope_dims=enc_rope_dims)

            # enc -> main projection
            self.enc_to_main = CastedLinear(d_enc, d_main, bias=False)

            # main network
            self.main_blocks = nn.ModuleList([
                Block(dim=d_main, num_heads=main_num_heads, num_kv_heads=main_num_kv_heads,
                      mlp_mult=main_mlp_mult, rope_base=rope_base, qk_gain_init=main_qk_gain_init,
                      train_seq_len=self.main_seq_len, layer_idx=i, ln_scale=True)
                for i in range(main_num_layers)
            ])
            for blk in self.main_blocks:
                blk.attn.rope_dims = main_rope_dims
                blk.attn.rotary = Rotary(d_main // main_num_heads, base=rope_base,
                                          train_seq_len=self.main_seq_len, rope_dims=main_rope_dims)

            # main -> dec projection
            self.main_to_dec = CastedLinear(d_main, d_enc, bias=False)

            # byte decoder
            self.byte_decoder = nn.ModuleList([
                Block(dim=d_enc, num_heads=dec_num_heads, num_kv_heads=dec_num_kv_heads,
                      mlp_mult=dec_mlp_mult, rope_base=rope_base, qk_gain_init=dec_qk_gain_init,
                      train_seq_len=byte_seq_len, layer_idx=i, ln_scale=True)
                for i in range(dec_num_layers)
            ])
            for blk in self.byte_decoder:
                blk.attn.rope_dims = dec_rope_dims
                blk.attn.rotary = Rotary(d_enc // dec_num_heads, base=rope_base,
                                          train_seq_len=byte_seq_len, rope_dims=dec_rope_dims)

            # final norm + head
            self.final_norm = RMSNorm()
            self.byte_head = CastedLinear(d_enc, 256, bias=False)
            nn.init.zeros_(self.byte_head.weight)  # zero-init per baseline convention

        def forward_logits(self, input_bytes: torch.Tensor) -> torch.Tensor:
            B, T = input_bytes.shape
            assert T == self.byte_seq_len, f"byte_seq_len mismatch {T} vs {self.byte_seq_len}"

            x = self.byte_emb(input_bytes)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x
            for blk in self.byte_encoder:
                x = blk(x, x0)
            x_enc_final = x   # (B, T, D_enc) — fine-grained byte representation; skip source

            # project + downsample (fixed stride)
            x_main = self.enc_to_main(x_enc_final)
            x_main = x_main[:, :: self.chunk_stride, :].contiguous()   # (B, T_main, D_main)
            x0_main = x_main
            for blk in self.main_blocks:
                x_main = blk(x_main, x0_main)

            # project + upsample (repeat each chunk by stride)
            x_dec = self.main_to_dec(x_main)                               # (B, T_main, D_enc)
            x_dec = x_dec.repeat_interleave(self.chunk_stride, dim=1)      # (B, T_main*stride, D_enc)
            if x_dec.size(1) < T:
                x_dec = F.pad(x_dec, (0, 0, 0, T - x_dec.size(1)))
            x_dec = x_dec[:, :T, :].contiguous()

            # H-Net-style byte skip: combine main-output + byte-encoder-output so the byte
            # decoder has BOTH coarse (main) and fine (byte encoder) representations per byte.
            x_dec = x_dec + x_enc_final
            x0_dec = x_enc_final   # initial residual lane = byte-level info directly
            for blk in self.byte_decoder:
                x_dec = blk(x_dec, x0_dec)

            x_dec = self.final_norm(x_dec)
            logits = self.byte_head(x_dec)                                 # (B, T, 256)
            return logits

        def forward(self, input_bytes: torch.Tensor, target_bytes: torch.Tensor) -> torch.Tensor:
            logits = self.forward_logits(input_bytes)
            logits = logits.reshape(-1, logits.size(-1))
            targets = target_bytes.reshape(-1)
            return F.cross_entropy(logits.float(), targets, reduction="mean")

    return HNetM1()


def count_params(model: nn.Module, exclude_embeddings: bool = False) -> tuple[int, int]:
    total = 0; nonembed = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if not any(k in name for k in ("byte_emb", "byte_head")):
            nonembed += p.numel()
    return total, nonembed
