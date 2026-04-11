"""Tests for checkpoint saving, scoring script, and adversarial masking.

Verifies:
1. Checkpoint saving creates per-experiment directories
2. score_and_reorder_data.py shard I/O functions
3. Adversarial embedding masking math and integration
4. Backward compatibility (all new features disabled by default)

Run: .venv/bin/python -m pytest tests/test_checkpoint_and_scoring.py -v
"""
import sys
import os
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_module(path):
    import importlib.util
    name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def r2():
    return _load_module("train_gpt_r2.py")


@pytest.fixture(scope="module")
def scoring():
    return _load_module("scripts/score_and_reorder_data.py")


SMALL = dict(
    vocab_size=64, num_layers=4, model_dim=128, num_heads=4,
    num_kv_heads=2, mlp_mult=2, tie_embeddings=True,
    tied_embed_init_std=0.005, logit_softcap=30.0,
    rope_base=10000.0, qk_gain_init=1.5,
)


# =====================================================
# Checkpoint Saving Tests
# =====================================================

class TestCheckpointSaving:
    def test_checkpoint_dir_structure(self, r2):
        """Verify checkpoint save code creates correct directory structure."""
        # We can't run the full training loop, but we can verify the code pattern
        # exists in the script by checking the module has the right constructs
        import ast
        tree = ast.parse(open("train_gpt_r2.py").read())
        # Find the checkpoint saving code
        source = open("train_gpt_r2.py").read()
        assert 'ckpt_dir = os.path.join("checkpoints", args.run_id)' in source
        assert 'os.makedirs(ckpt_dir, exist_ok=True)' in source
        assert 'torch.save(base_model.state_dict(), ckpt_path)' in source

    def test_checkpoint_backward_compat(self, r2):
        """Verify default final_model.pt is still saved."""
        source = open("train_gpt_r2.py").read()
        assert 'torch.save(base_model.state_dict(), "final_model.pt")' in source

    def test_all_scripts_have_checkpoint_saving(self):
        """All 3 training scripts must have per-experiment checkpoint saving."""
        for script in ["train_gpt_r1.py", "train_gpt_r2.py", "train_gpt_r3.py"]:
            source = open(script).read()
            assert 'ckpt_dir = os.path.join("checkpoints", args.run_id)' in source, \
                f"{script} missing checkpoint saving"


# =====================================================
# Shard I/O Tests (for scoring script)
# =====================================================

class TestShardIO:
    def test_write_and_read_roundtrip(self, scoring):
        """Write a shard, read it back, verify contents match."""
        tokens = np.array([1, 2, 3, 4, 5, 100, 200, 500], dtype="<u2")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            scoring.write_data_shard(path, tokens)
            loaded = scoring.load_data_shard(path)
            assert len(loaded) == len(tokens)
            assert (loaded.numpy() == tokens).all(), "Tokens don't match after roundtrip"
        finally:
            os.unlink(path)

    def test_shard_header_format(self, scoring):
        """Verify shard header has correct magic and version."""
        tokens = np.array([10, 20, 30], dtype="<u2")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            scoring.write_data_shard(path, tokens)
            header = np.fromfile(path, dtype="<i4", count=3)
            assert header[0] == 20240520, f"Magic should be 20240520, got {header[0]}"
            assert header[1] == 1, f"Version should be 1, got {header[1]}"
            assert header[2] == 3, f"Token count should be 3, got {header[2]}"
        finally:
            os.unlink(path)

    def test_empty_shard(self, scoring):
        """Writing an empty shard should work."""
        tokens = np.array([], dtype="<u2")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            scoring.write_data_shard(path, tokens)
            loaded = scoring.load_data_shard(path)
            assert len(loaded) == 0
        finally:
            os.unlink(path)

    def test_large_token_values(self, scoring):
        """uint16 supports values up to 65535."""
        tokens = np.array([0, 1023, 65535], dtype="<u2")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            scoring.write_data_shard(path, tokens)
            loaded = scoring.load_data_shard(path)
            assert (loaded.numpy() == tokens).all()
        finally:
            os.unlink(path)


# =====================================================
# Adversarial Masking Math Tests
# =====================================================

class TestAdversarialMaskMath:
    def test_sigmoid_mapping(self):
        """Low loss should map to high mask probability via sigmoid."""
        prev_loss = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        base, max_rate, threshold = 0.05, 0.30, 1.0
        mask_prob = base + (max_rate - base) * torch.sigmoid(-(prev_loss - threshold))

        # At threshold (loss=1.0): sigmoid(0) = 0.5 → mask_prob = 0.05 + 0.25*0.5 = 0.175
        assert abs(mask_prob[2].item() - 0.175) < 0.01, f"At threshold, mask_prob should be ~0.175, got {mask_prob[2]}"

        # Low loss (0.1) → high mask prob
        assert mask_prob[0] > mask_prob[4], "Low loss should have higher mask prob than high loss"

        # Monotonically decreasing with loss
        for i in range(len(prev_loss) - 1):
            assert mask_prob[i] >= mask_prob[i + 1], f"mask_prob should decrease with loss at index {i}"

    def test_mask_prob_range(self):
        """Mask probability should be bounded by [base, max_rate]."""
        prev_loss = torch.tensor([0.001, 100.0])  # extreme values
        base, max_rate, threshold = 0.05, 0.30, 1.0
        mask_prob = base + (max_rate - base) * torch.sigmoid(-(prev_loss - threshold))

        assert mask_prob.min() >= base - 0.01, f"Min mask_prob should be >= base, got {mask_prob.min()}"
        assert mask_prob.max() <= max_rate + 0.01, f"Max mask_prob should be <= max_rate, got {mask_prob.max()}"

    def test_embedding_mask_shape(self):
        """Embedding mask should be [T] and broadcastable to [B, T, D]."""
        T = 32
        mask_prob = torch.full((T,), 0.2)
        embedding_mask = (torch.rand(T) > mask_prob).float()
        embedding_mask[0] = 1.0

        assert embedding_mask.shape == (T,)
        assert embedding_mask[0] == 1.0

        # Broadcast test
        B, D = 2, 128
        x = torch.randn(B, T, D)
        masked_x = x * embedding_mask.unsqueeze(-1)  # [T, 1] broadcast
        # Hmm, need [B, T, 1] — but mask is [T], not [B, T]
        # Actually in the code it's applied as: x * embedding_mask.unsqueeze(-1)
        # where embedding_mask is [T] → unsqueeze(-1) → [T, 1]
        # But x is [B, T, D] — broadcasting: [T, 1] broadcasts with [B, T, D]? No!
        # [T, 1] doesn't broadcast with [B, T, D] because leading dims don't match.
        # Need [1, T, 1] or the code needs to handle batching.
        # Let me check what the actual code does...
        pass  # Flagged for review below

    def test_prev_loss_cache_detached(self):
        """Cached loss must be detached from computation graph."""
        loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        cached = loss.detach().cpu()
        assert not cached.requires_grad, "Cached loss should not require grad"
        assert cached.device == torch.device("cpu"), "Cached loss should be on CPU"


# =====================================================
# Adversarial Masking Integration Tests
# =====================================================

class TestAdversarialMaskingIntegration:
    def test_embedding_mask_applied_correctly(self, r2):
        """Verify embedding_mask zeros out the right positions in forward pass."""
        model = r2.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))

        # All-ones mask should match no mask
        loss_no_mask = model(x, y)
        loss_ones = model(x, y, embedding_mask=torch.ones(16))
        assert torch.allclose(loss_no_mask, loss_ones, atol=1e-5), \
            f"All-ones mask should be identity: {loss_no_mask} vs {loss_ones}"

    def test_masking_changes_loss(self, r2):
        """Masking many positions should measurably change the loss."""
        model = r2.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        torch.manual_seed(42)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))

        loss_no_mask = model(x, y)
        # Mask 50% of positions — strong enough signal to change loss
        mask = torch.ones(16)
        mask[1::2] = 0.0  # mask all odd positions
        mask[0] = 1.0  # keep first
        loss_masked = model(x, y, embedding_mask=mask)

        # With 50% context removed, loss should differ noticeably
        assert abs(loss_no_mask.item() - loss_masked.item()) > 1e-4, \
            f"Masking 50% positions should change loss: {loss_no_mask.item():.6f} vs {loss_masked.item():.6f}"

    def test_per_position_loss_with_mask(self, r2):
        """reduction='none' with embedding_mask should return per-position losses."""
        model = r2.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        mask = torch.ones(16)
        mask[5] = 0.0

        loss = model(x, y, reduction="none", embedding_mask=mask)
        assert loss.shape == (2 * 16,), f"Expected shape (32,), got {loss.shape}"
        assert torch.isfinite(loss).all()

    def test_mask_gradient_to_embedding(self, r2):
        """Gradient should flow to embedding weights through non-masked positions."""
        model = r2.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        mask = torch.ones(16)
        mask[3] = 0.0

        loss = model(x, y, embedding_mask=mask)
        loss.backward()
        assert model.tok_emb.weight.grad is not None
        assert model.tok_emb.weight.grad.abs().sum() > 0, "Should have non-zero gradients"

    def test_adversarial_with_all_features(self, r2):
        """Adversarial masking should work with BigramHash, XSA, Value residual, DML-Gated."""
        model = r2.GPT(
            **SMALL, bigram_vocab_size=32, bigram_dim=16,
            xsa_last_n=2, value_residual=True, mlp_type="dml_gated",
        )
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        mask = torch.ones(16)
        mask[5] = 0.0

        loss = model(x, y, reduction="none", embedding_mask=mask)
        assert loss.shape == (32,)
        assert torch.isfinite(loss).all()
        loss.mean().backward()


# =====================================================
# Backward Compatibility Tests
# =====================================================

class TestBackwardCompatibility:
    def test_default_env_vars(self, r2):
        """All new features should be disabled by default."""
        h = r2.Hyperparameters()
        assert h.causal_probe == False, "CAUSAL_PROBE should default to False"
        assert h.causal_probe_mode == "adaptive", "CAUSAL_PROBE_MODE should default to 'adaptive'"
        assert h.token_drop_rate == 0.0, "TOKEN_DROP_RATE should default to 0"
        assert h.corrupt_rate == 0.0, "CORRUPT_RATE should default to 0"
        assert h.cross_layer_bt == False, "CROSS_LAYER_BT should default to False"

    def test_model_forward_default_signature(self, r2):
        """model(x, y) without extra args should work (backward compat)."""
        model = r2.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        loss = model(x, y)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_reduction_default_is_mean(self, r2):
        """Default reduction should be 'mean' (scalar loss)."""
        model = r2.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        loss = model(x, y)
        assert loss.ndim == 0, f"Default should return scalar, got ndim={loss.ndim}"


# =====================================================
# Embedding Mask Broadcasting Bug Check
# =====================================================

class TestEmbeddingMaskBroadcasting:
    def test_mask_broadcasts_correctly_in_forward(self, r2):
        """Verify the unsqueeze(-1) broadcast works for [T] mask with [B, T, D] embeddings."""
        # The code does: x = x * embedding_mask.unsqueeze(-1)
        # embedding_mask is [T], unsqueeze(-1) makes it [T, 1]
        # x is [B, T, D]
        # Broadcasting: [T, 1] with [B, T, D] — this SHOULD work because
        # PyTorch broadcasts from the right: [T, 1] → [1, T, 1] → [B, T, 1] → [B, T, D]
        B, T, D = 2, 16, 128
        x = torch.randn(B, T, D)
        mask = torch.ones(T)
        mask[3] = 0.0

        result = x * mask.unsqueeze(-1)
        assert result.shape == (B, T, D)
        assert (result[:, 3, :] == 0).all(), "Masked position should be all zeros"
        assert (result[:, 0, :] != 0).any(), "Non-masked position should have non-zero values"
