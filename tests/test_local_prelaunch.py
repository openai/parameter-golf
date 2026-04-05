"""Pre-launch local validation — everything we can test before GPU credits

Run with: python3 -m pytest tests/test_local_prelaunch.py -v -s

Tests:
  1. Real FineWeb data loading + tokenizer BPB evaluation
  2. Training on real data (model learns actual language patterns)
  3. Production-size model param count + artifact size
  4. Autoregressive calibration generation (GPTQ)
  5. Complete serialize → deserialize → eval roundtrip
  6. Sliding window eval code path
"""
from __future__ import annotations

import io
import lzma
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm
import torch

# ---- Mock GPU-only modules ----
def _fake_flash_attn(q, k, v, causal=False):
    bsz, seqlen, nqh, hd = q.shape
    nkvh = k.shape[2]
    if nqh != nkvh:
        reps = nqh // nkvh
        k = k.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
        v = v.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
    scale = hd ** -0.5
    q2 = q.transpose(1, 2).float()
    k2 = k.transpose(1, 2).float()
    v2 = v.transpose(1, 2).float()
    attn = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v2)
    return out.transpose(1, 2).to(q.dtype)

for mod_name in ("flash_attn_interface", "flash_attn", "mamba_ssm", "causal_conv1d"):
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        fake.flash_attn_func = _fake_flash_attn
        sys.modules[mod_name] = fake
sys.modules["flash_attn_interface"].flash_attn_func = _fake_flash_attn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt
train_gpt.flash_attn_3_func = _fake_flash_attn
from train_gpt import (
    GPT, MambaBlock, CastedLinear, _HessianGPT,
    _unbank_state_dict, _rebank_state_dict,
    mixed_quantize_int6, dequantize_mixed_int6,
    load_data_shard, build_sentencepiece_luts,
    restore_low_dim_params_to_fp32,
)

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "datasets" / "fineweb10B_sp1024"
TOKENIZER_PATH = ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
TRAIN_SHARD = DATA_DIR / "fineweb_train_000000.bin"
VAL_SHARD = DATA_DIR / "fineweb_val_000000.bin"

# Skip all tests if data not downloaded
pytestmark = pytest.mark.skipif(
    not TRAIN_SHARD.exists() or not VAL_SHARD.exists(),
    reason="Dataset not downloaded. Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1"
)


def _make_model(num_layers=6, mamba_layers="0,1,2", model_dim=64, **kw):
    defaults = dict(
        vocab_size=1024, num_heads=4, num_kv_heads=2, mlp_mult=3.0,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        mamba_d_state=8, mamba_d_conv=4, mamba_expand=1.5,
    )
    defaults.update(kw)
    return GPT(num_layers=num_layers, mamba_layers=mamba_layers,
               model_dim=model_dim, **defaults)


# ===== Test 1: Real data loading =====

class TestRealDataLoading:
    def test_load_train_shard(self):
        tokens = load_data_shard(TRAIN_SHARD)
        assert tokens.ndim == 1
        assert tokens.numel() > 1_000_000  # at least 1M tokens
        assert tokens.long().max().item() < 1024  # all within vocab
        print(f"\n  Train shard: {tokens.numel():,} tokens")

    def test_load_val_shard(self):
        tokens = load_data_shard(VAL_SHARD)
        assert tokens.ndim == 1
        assert tokens.numel() > 1_000_000
        assert tokens.long().max().item() < 1024
        print(f"\n  Val shard: {tokens.numel():,} tokens")

    def test_tokenizer_and_bpb_luts(self):
        sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
        assert sp.vocab_size() == 1024
        base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, 1024, torch.device("cpu"))
        assert base_bytes.shape == (1024,)
        # Most tokens should map to 1+ bytes
        assert (base_bytes > 0).sum().item() > 500
        print(f"\n  Tokens with >0 bytes: {(base_bytes > 0).sum().item()}/1024")


# ===== Test 2: Train on real FineWeb data =====

class TestRealDataTraining:
    def test_learn_real_language_patterns(self):
        """Train on actual FineWeb data — loss should decrease faster than random."""
        torch.manual_seed(42)
        model = _make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Load real training data
        tokens = load_data_shard(TRAIN_SHARD)
        seq_len = 64

        losses = []
        for step in range(15):
            start = step * 4 * seq_len  # non-overlapping batches
            batch = tokens[start:start + 4 * (seq_len + 1)].long()
            ids = batch[:4 * seq_len].reshape(4, seq_len)
            tgt = batch[1:1 + 4 * seq_len].reshape(4, seq_len)
            optimizer.zero_grad()
            loss = model(ids, tgt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Should decrease meaningfully on real data
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        improvement = avg_first - avg_last
        print(f"\n  Real data training: first3_avg={avg_first:.4f} last3_avg={avg_last:.4f} improvement={improvement:.4f}")
        assert avg_last < avg_first, f"No improvement on real data: {losses}"


# ===== Test 3: BPB evaluation on real val data =====

class TestBPBEvaluation:
    def test_eval_bpb_on_real_val(self):
        """Compute actual BPB on validation data — should be in reasonable range."""
        torch.manual_seed(42)
        model = _make_model()
        model.eval()

        sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
        base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, 1024, torch.device("cpu"))

        # Load a chunk of val data
        val_tokens = load_data_shard(VAL_SHARD)
        seq_len = 64
        num_seqs = 32  # evaluate on 32 sequences

        total_loss = 0.0
        total_tokens = 0
        total_bytes = 0.0

        with torch.no_grad():
            for i in range(num_seqs):
                start = i * seq_len
                chunk = val_tokens[start:start + seq_len + 1].long()
                x = chunk[:-1].unsqueeze(0)  # (1, seq_len)
                y = chunk[1:].unsqueeze(0)   # (1, seq_len)
                loss = model(x, y)
                total_loss += loss.item() * seq_len
                total_tokens += seq_len

                # BPB calculation
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes_val = base_bytes[tgt_ids].to(dtype=torch.int16)
                token_bytes_val += (has_space[tgt_ids] & ~is_boundary[prev_ids]).to(dtype=torch.int16)
                total_bytes += token_bytes_val.float().sum().item()

        avg_loss = total_loss / total_tokens
        bits_per_token = avg_loss / math.log(2.0)
        tokens_per_byte = total_tokens / total_bytes
        bpb = bits_per_token * tokens_per_byte

        print(f"\n  Untrained model val: loss={avg_loss:.4f} bpb={bpb:.4f}")
        print(f"  (For reference: SOTA = 1.1147 BPB, random baseline ~ 6.93 nats)")

        # Untrained model should have high but finite BPB
        assert 1.0 < bpb < 20.0, f"BPB out of range: {bpb}"
        assert math.isfinite(bpb)


# ===== Test 4: Production-size model =====

class TestProductionSize:
    def test_full_512_model_param_count(self):
        """Instantiate the actual production config to get exact param count."""
        model = GPT(
            vocab_size=1024, num_layers=18, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=3.0,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            mamba_layers="0,1,2,3,4,5,6,7,8,9,10,11,15,16,17",
            mamba_d_state=32, mamba_d_conv=4, mamba_expand=1.5,
            bigram_vocab_size=2048, bigram_dim=128,
        )
        total = sum(p.numel() for p in model.parameters())
        mamba = sum(p.numel() for p in model.mamba_blocks.parameters())
        attn = total - mamba
        print(f"\n  Production model (d=512, 18L, 15M+3A):")
        print(f"    Total params: {total:,}")
        print(f"    Mamba params: {mamba:,}")
        print(f"    Non-Mamba params: {attn:,}")
        print(f"    Estimated int6 size: {total * 6 / 8 / 1024 / 1024:.1f} MB (raw)")

        # Should be around 27.8M as estimated
        assert 20_000_000 < total < 40_000_000

    def test_full_512_model_artifact_size(self):
        """Quantize the production model and measure actual LZMA artifact size."""
        torch.manual_seed(42)
        model = GPT(
            vocab_size=1024, num_layers=18, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=3.0,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            mamba_layers="0,1,2,3,4,5,6,7,8,9,10,11,15,16,17",
            mamba_d_state=32, mamba_d_conv=4, mamba_expand=1.5,
            bigram_vocab_size=2048, bigram_dim=128,
        )
        sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        n_attn = model.n_attn
        num_layers = 18

        unbanked = _unbank_state_dict(sd, num_layers, n_attn=n_attn)
        quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn", "mamba"})

        buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, buf)
        raw_bytes = len(buf.getvalue())
        compressed = lzma.compress(buf.getvalue(), preset=9)
        compressed_bytes = len(compressed)

        # Estimate code size (read train_gpt.py)
        code_bytes = (ROOT / "train_gpt.py").stat().st_size

        total_artifact = compressed_bytes + code_bytes
        print(f"\n  Production artifact estimation:")
        print(f"    Model raw (int6): {raw_bytes / 1024 / 1024:.2f} MB")
        print(f"    Model LZMA9:      {compressed_bytes / 1024 / 1024:.2f} MB")
        print(f"    Code (train_gpt): {code_bytes / 1024:.1f} KB")
        print(f"    Total artifact:   {total_artifact / 1024 / 1024:.2f} MB")
        print(f"    Budget remaining: {(16_000_000 - total_artifact) / 1024 / 1024:.2f} MB")

        if total_artifact <= 16_000_000:
            print(f"    STATUS: FITS in 16MB budget!")
        else:
            overage = total_artifact - 16_000_000
            print(f"    STATUS: OVER BUDGET by {overage / 1024:.1f} KB")
            print(f"    Action: Reduce expand, d_state, or enable selective pruning")

        # Even random weights should compress well with int6
        assert raw_bytes > 0
        assert compressed_bytes < raw_bytes


# ===== Test 5: Autoregressive calibration generation =====

class TestCalibrationGeneration:
    def test_autoregressive_generation(self):
        """Test the GPTQ calibration data generation (CPU version)."""
        torch.manual_seed(42)
        model = _make_model()
        model.eval()

        # CPU version of generate_autoregressive_calib
        # (original uses torch.autocast(cuda) which doesn't work on CPU)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42)
        num_seqs, seq_len = 4, 32

        all_tokens = []
        with torch.inference_mode():
            for batch_start in range(0, num_seqs, 2):
                bs = min(2, num_seqs - batch_start)
                tokens = torch.randint(0, 1024, (bs, 1), generator=rng)
                for pos in range(seq_len - 1):
                    logits = model.forward_logits(tokens)
                    next_logit = logits[:, -1, :]
                    probs = torch.softmax(next_logit / 0.8, dim=-1)
                    next_tok = torch.multinomial(probs, 1, generator=rng)
                    tokens = torch.cat([tokens, next_tok], dim=1)
                for i in range(bs):
                    all_tokens.append(tokens[i:i+1])

        assert len(all_tokens) == num_seqs
        for seq in all_tokens:
            assert seq.shape == (1, seq_len)
            assert seq.max().item() < 1024
            assert seq.min().item() >= 0
        print(f"\n  Generated {num_seqs} calibration sequences of length {seq_len}")
        print(f"  Sample tokens: {all_tokens[0][0, :10].tolist()}")


# ===== Test 6: Complete Hessian-based GPTQ pipeline =====

class TestGPTQPipeline:
    def test_hessian_collection_and_quantize(self):
        """Full Hessian collection → GPTQ quantization on CPU."""
        from train_gpt import collect_hessians_from_tokens

        torch.manual_seed(42)
        model = _make_model(num_layers=4, mamba_layers="0,1")
        model.eval()

        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        n_attn = model.n_attn
        unbanked = _unbank_state_dict(sd, 4, n_attn=n_attn)

        # Build Hessian model
        hessian_model = _HessianGPT(
            vocab_size=1024, num_layers=4, model_dim=64,
            num_heads=4, num_kv_heads=2, mlp_mult=3.0,
            tie_embeddings=True, logit_softcap=30.0,
            rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=2048, bigram_dim=128,
            xsa_last_n=4, rope_dims=16, ln_scale=True,
            ve_enabled=True, ve_dim=16, ve_layers="",
            mamba_layers="0,1", mamba_d_state=8,
            mamba_d_conv=4, mamba_expand=1.5,
        )
        loadable = {k: v for k, v in unbanked.items() if k in hessian_model.state_dict()}
        hessian_model.load_state_dict(loadable, strict=False)

        # Generate calibration tokens on CPU
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42)
        calib_tokens = []
        with torch.inference_mode():
            for _ in range(4):
                tokens = torch.randint(0, 1024, (1, 1), generator=rng)
                for pos in range(31):
                    logits = model.forward_logits(tokens)
                    probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                    tokens = torch.cat([tokens, torch.multinomial(probs, 1, generator=rng)], dim=1)
                calib_tokens.append(tokens)

        # Collect Hessians
        hessians = collect_hessians_from_tokens(hessian_model, calib_tokens, torch.device("cpu"))
        print(f"\n  Hessians collected for {len(hessians)} layers")
        assert len(hessians) > 0

        # Quantize with Hessians
        quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn", "mamba"}, hessians=hessians)
        assert len(quant_result) > 0

        # Verify dequantized model still works
        deq = dequantize_mixed_int6(quant_result, quant_meta, unbanked)
        rebanked = _rebank_state_dict(deq, 4, sd, n_attn=n_attn)

        model2 = _make_model(num_layers=4, mamba_layers="0,1")
        model2.load_state_dict(rebanked, strict=True)
        model2.eval()

        ids = torch.randint(0, 1024, (1, 16))
        tgt = torch.randint(0, 1024, (1, 16))
        with torch.no_grad():
            loss = model2(ids, tgt)
        assert torch.isfinite(loss)
        print(f"  Post-GPTQ loss: {loss.item():.4f}")


# ===== Test 7: forward_logits consistency =====

class TestForwardLogits:
    def test_forward_logits_shape(self):
        model = _make_model()
        model.eval()
        ids = torch.randint(0, 1024, (2, 32))
        with torch.no_grad():
            logits = model.forward_logits(ids)
        assert logits.shape == (2, 32, 1024)
        assert torch.isfinite(logits).all()

    def test_forward_logits_with_real_data(self):
        model = _make_model()
        model.eval()
        tokens = load_data_shard(TRAIN_SHARD)
        ids = tokens[:64].long().unsqueeze(0)  # (1, 64)
        with torch.no_grad():
            logits = model.forward_logits(ids)
        assert logits.shape == (1, 64, 1024)
        # Top prediction should be a valid token
        top_tokens = logits[0].argmax(dim=-1)
        assert top_tokens.max().item() < 1024
