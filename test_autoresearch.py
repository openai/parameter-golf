"""
tests for the autoresearch pipeline and train_gpt.py components.
run with: pytest test_autoresearch.py -v
"""

import io
import math
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------

class TestHyperparameters:
    def test_defaults(self):
        # import fresh each time to pick up env
        from train_gpt import Hyperparameters
        args = Hyperparameters()
        assert args.vocab_size == 1024
        assert args.num_layers == 9
        assert args.model_dim == 512
        assert args.num_heads == 8
        assert args.num_kv_heads == 4
        assert args.tie_embeddings is True
        assert args.max_wallclock_seconds == 600.0

    def test_env_override(self):
        with patch.dict(os.environ, {"VOCAB_SIZE": "2048", "NUM_LAYERS": "12"}):
            # re-import to pick up patched env
            import importlib
            import train_gpt
            importlib.reload(train_gpt)
            args = train_gpt.Hyperparameters()
            assert args.vocab_size == 2048
            assert args.num_layers == 12
        # reload back to defaults
        import importlib
        import train_gpt
        importlib.reload(train_gpt)


# ---------------------------------------------------------------------------
# model architecture
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    @pytest.fixture
    def small_model(self):
        from train_gpt import GPT
        return GPT(
            vocab_size=64,
            num_layers=2,
            model_dim=32,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
        )

    def test_forward_runs(self, small_model):
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        loss = small_model(x, y)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_tied_embeddings(self, small_model):
        assert small_model.lm_head is None
        assert small_model.tie_embeddings is True

    def test_untied_embeddings(self):
        from train_gpt import GPT
        model = GPT(
            vocab_size=64, num_layers=2, model_dim=32,
            num_heads=4, num_kv_heads=2, mlp_mult=2,
            tie_embeddings=False, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        assert model.lm_head is not None

    def test_encoder_decoder_split(self, small_model):
        # 2 layers -> 1 encoder + 1 decoder
        assert small_model.num_encoder_layers == 1
        assert small_model.num_decoder_layers == 1

    def test_skip_weights_shape(self, small_model):
        expected = min(small_model.num_encoder_layers, small_model.num_decoder_layers)
        assert small_model.skip_weights.shape == (expected, 32)

    def test_logit_softcap_positive(self):
        from train_gpt import GPT
        with pytest.raises(ValueError, match="logit_softcap must be positive"):
            GPT(
                vocab_size=64, num_layers=2, model_dim=32,
                num_heads=4, num_kv_heads=2, mlp_mult=2,
                tie_embeddings=True, tied_embed_init_std=0.005,
                logit_softcap=-1.0, rope_base=10000.0, qk_gain_init=1.5,
            )

    def test_param_count_reasonable(self, small_model):
        n_params = sum(p.numel() for p in small_model.parameters())
        # small model should have some params but not too many
        assert 1000 < n_params < 100_000


# ---------------------------------------------------------------------------
# individual modules
# ---------------------------------------------------------------------------

class TestModules:
    def test_rms_norm(self):
        from train_gpt import RMSNorm
        norm = RMSNorm()
        x = torch.randn(2, 4, 32)
        out = norm(x)
        assert out.shape == x.shape
        # rms norm should roughly normalize the last dim
        rms = (out ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_casted_linear(self):
        from train_gpt import CastedLinear
        layer = CastedLinear(32, 64, bias=False)
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (2, 64)
        assert out.dtype == torch.bfloat16

    def test_rotary(self):
        from train_gpt import Rotary
        rot = Rotary(16, base=10000.0)
        cos, sin = rot(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
        assert cos.shape == (1, 1, 8, 8)  # half of dim=16
        assert sin.shape == (1, 1, 8, 8)

    def test_rotary_caching(self):
        from train_gpt import Rotary
        rot = Rotary(16)
        cos1, sin1 = rot(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
        cos2, sin2 = rot(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
        assert cos1 is cos2  # should be cached

    def test_apply_rotary_emb(self):
        from train_gpt import apply_rotary_emb
        x = torch.randn(1, 1, 4, 8)
        cos = torch.ones(1, 1, 4, 4)
        sin = torch.zeros(1, 1, 4, 4)
        # with cos=1 sin=0, rotary should be identity
        out = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(out, x)

    def test_mlp(self):
        from train_gpt import MLP
        mlp = MLP(dim=32, mlp_mult=2)
        x = torch.randn(2, 4, 32)
        out = mlp(x)
        assert out.shape == (2, 4, 32)

    def test_block(self):
        from train_gpt import Block
        block = Block(dim=32, num_heads=4, num_kv_heads=2, mlp_mult=2,
                      rope_base=10000.0, qk_gain_init=1.5)
        x = torch.randn(2, 4, 32)
        x0 = torch.randn(2, 4, 32)
        out = block(x, x0)
        assert out.shape == (2, 4, 32)


# ---------------------------------------------------------------------------
# quantization roundtrip
# ---------------------------------------------------------------------------

class TestQuantization:
    def test_int8_roundtrip_small(self):
        from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8
        state = {"weight": torch.randn(8, 8)}
        obj, stats = quantize_state_dict_int8(state)
        restored = dequantize_state_dict_int8(obj)
        assert "weight" in restored
        # int8 quantization loses precision but should be close
        assert torch.allclose(state["weight"], restored["weight"], atol=0.1)

    def test_int8_roundtrip_large_matrix(self):
        from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8
        # large enough to trigger per-row quantization (> INT8_KEEP_FLOAT_MAX_NUMEL)
        w = torch.randn(512, 512)
        state = {"big_weight": w}
        obj, stats = quantize_state_dict_int8(state)
        restored = dequantize_state_dict_int8(obj)
        # per-row int8 should preserve reasonable accuracy
        cos_sim = torch.nn.functional.cosine_similarity(
            w.flatten().unsqueeze(0),
            restored["big_weight"].flatten().unsqueeze(0),
        )
        assert cos_sim.item() > 0.99

    def test_int8_passthrough_nonfloat(self):
        from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8
        state = {"indices": torch.tensor([1, 2, 3], dtype=torch.int64)}
        obj, stats = quantize_state_dict_int8(state)
        restored = dequantize_state_dict_int8(obj)
        assert torch.equal(state["indices"], restored["indices"])

    def test_int8_stats(self):
        from train_gpt import quantize_state_dict_int8
        state = {"w": torch.randn(4, 4), "b": torch.randn(4)}
        obj, stats = quantize_state_dict_int8(state)
        assert stats["num_tensors"] == 2
        assert stats["param_count"] == 20

    def test_zlib_compression(self):
        import zlib
        from train_gpt import quantize_state_dict_int8
        # a real model's quantized state should compress well
        from train_gpt import GPT
        model = GPT(
            vocab_size=64, num_layers=2, model_dim=32,
            num_heads=4, num_kv_heads=2, mlp_mult=2,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        obj, stats = quantize_state_dict_int8(model.state_dict())
        buf = io.BytesIO()
        torch.save(obj, buf)
        raw = buf.getvalue()
        compressed = zlib.compress(raw, 9)
        # compressed should be smaller
        assert len(compressed) < len(raw)


# ---------------------------------------------------------------------------
# artifact size constraint
# ---------------------------------------------------------------------------

class TestArtifactSize:
    def test_baseline_under_16mb(self):
        """the default baseline config must produce an artifact under 16mb."""
        import zlib
        from train_gpt import GPT, quantize_state_dict_int8
        model = GPT(
            vocab_size=1024, num_layers=9, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=2,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        obj, stats = quantize_state_dict_int8(model.state_dict())
        buf = io.BytesIO()
        torch.save(obj, buf)
        compressed = zlib.compress(buf.getvalue(), 9)
        code_size = Path("train_gpt.py").stat().st_size
        total = len(compressed) + code_size
        assert total < 16_000_000, f"artifact {total} bytes exceeds 16MB limit"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

class TestDataLoading:
    def _make_shard(self, path: Path, num_tokens: int):
        """create a minimal valid shard file."""
        header = np.zeros(256, dtype="<i4")
        header[0] = 20240520  # magic
        header[1] = 1         # version
        header[2] = num_tokens
        tokens = np.random.randint(0, 1024, size=num_tokens, dtype="<u2")
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())

    def test_load_data_shard(self, tmp_path):
        from train_gpt import load_data_shard
        shard = tmp_path / "test_shard.bin"
        self._make_shard(shard, 1000)
        tokens = load_data_shard(shard)
        assert tokens.shape == (1000,)
        assert tokens.dtype == torch.uint16

    def test_load_data_shard_bad_magic(self, tmp_path):
        from train_gpt import load_data_shard
        shard = tmp_path / "bad_shard.bin"
        header = np.zeros(256, dtype="<i4")
        header[0] = 12345  # wrong magic
        header[1] = 1
        header[2] = 100
        tokens = np.zeros(100, dtype="<u2")
        with open(shard, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())
        with pytest.raises(ValueError, match="Unexpected shard header"):
            load_data_shard(shard)

    def test_token_stream(self, tmp_path):
        from train_gpt import TokenStream
        shard = tmp_path / "fineweb_train_000000.bin"
        self._make_shard(shard, 500)
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        chunk = stream.take(100)
        assert chunk.shape == (100,)

    def test_token_stream_wraps(self, tmp_path):
        from train_gpt import TokenStream
        shard = tmp_path / "fineweb_train_000000.bin"
        self._make_shard(shard, 50)
        stream = TokenStream(str(tmp_path / "fineweb_train_*.bin"))
        # take more than one shard's worth — should wrap around
        chunk = stream.take(120)
        assert chunk.shape == (120,)

    def test_token_stream_no_files(self, tmp_path):
        from train_gpt import TokenStream
        with pytest.raises(FileNotFoundError):
            TokenStream(str(tmp_path / "nonexistent_*.bin"))


# ---------------------------------------------------------------------------
# newton-schulz orthogonalization (muon core)
# ---------------------------------------------------------------------------

class TestNewtonSchulz:
    def test_output_shape(self):
        from train_gpt import zeropower_via_newtonschulz5
        g = torch.randn(64, 32)
        out = zeropower_via_newtonschulz5(g, steps=5)
        assert out.shape == g.shape

    def test_approximately_orthogonal(self):
        from train_gpt import zeropower_via_newtonschulz5
        # newton-schulz runs in bf16 so precision is limited.
        # check that singular values are roughly 1 (the "zeropower" property)
        g = torch.randn(16, 16)
        out = zeropower_via_newtonschulz5(g, steps=10)
        svs = torch.linalg.svdvals(out.float())
        assert torch.allclose(svs, torch.ones_like(svs), atol=0.35)

    def test_transposed_case(self):
        from train_gpt import zeropower_via_newtonschulz5
        # tall matrix (rows > cols) triggers transposed path
        g = torch.randn(64, 16)
        out = zeropower_via_newtonschulz5(g, steps=5)
        assert out.shape == (64, 16)


# ---------------------------------------------------------------------------
# program.md contract
# ---------------------------------------------------------------------------

class TestProgramMd:
    def test_exists(self):
        assert Path("program.md").is_file()

    def test_has_required_sections(self):
        content = Path("program.md").read_text()
        assert "## Setup" in content
        assert "## Experimentation" in content
        assert "## Reasoning" in content
        assert "## Backtracking" in content
        assert "## The Experiment Loop" in content
        assert "NEVER STOP" in content

    def test_no_push(self):
        content = Path("program.md").read_text()
        assert "NEVER push" in content or "NEVER run `git push`" in content

    def test_artifact_limit_mentioned(self):
        content = Path("program.md").read_text()
        assert "16MB" in content or "16,000,000" in content

    def test_modal_launch_command(self):
        content = Path("program.md").read_text()
        assert "modal run modal_train.py" in content


# ---------------------------------------------------------------------------
# modal_train.py
# ---------------------------------------------------------------------------

class TestModalTrain:
    def test_file_exists(self):
        assert Path("modal_train.py").is_file()

    def test_mounts_local_train_gpt(self):
        content = Path("modal_train.py").read_text()
        assert "train_gpt.py" in content
        assert "Mount" in content or "mount" in content

    def test_has_single_and_multi_gpu(self):
        content = Path("modal_train.py").read_text()
        assert "H100" in content
        assert "H100:8" in content
