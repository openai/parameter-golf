# tests/test_artifact_size.py
"""Tests to verify submission artifact stays under 16MB."""
import os
import pytest

MAX_ARTIFACT_BYTES = 16_000_000  # 16MB decimal (challenge rule)


def get_artifact_size(model_path: str, code_path: str = "train_gpt.py") -> dict:
    """Calculate artifact size as defined by challenge rules."""
    if not os.path.exists(model_path):
        return None
    if not os.path.exists(code_path):
        return None
    model_bytes = os.path.getsize(model_path)
    code_bytes = len(open(code_path, "r", encoding="utf-8").read().encode("utf-8"))
    return {
        "model_bytes": model_bytes,
        "code_bytes": code_bytes,
        "total_bytes": model_bytes + code_bytes,
        "under_limit": (model_bytes + code_bytes) < MAX_ARTIFACT_BYTES,
        "headroom_bytes": MAX_ARTIFACT_BYTES - (model_bytes + code_bytes),
    }


class TestArtifactSize:
    def test_code_alone_under_limit(self):
        """train_gpt.py code alone must be < 15MB (leave room for weights)."""
        code_path = "train_gpt.py"
        if not os.path.exists(code_path):
            pytest.skip("train_gpt.py not found — run from repo root")
        code_bytes = len(open(code_path, encoding="utf-8").read().encode("utf-8"))
        assert code_bytes < 15_000_000, \
            f"Code alone is {code_bytes:,} bytes — too large for a real model to fit"

    def test_artifact_under_16mb(self):
        """Full artifact (code + compressed model) must be under 16MB."""
        model_path = "final_model.int8.ptz"
        if not os.path.exists(model_path):
            pytest.skip("final_model.int8.ptz not found — run a training first")
        result = get_artifact_size(model_path)
        assert result["under_limit"], (
            f"Artifact OVER LIMIT: {result['total_bytes']:,} bytes "
            f"({result['total_bytes'] - MAX_ARTIFACT_BYTES:,} bytes over)"
        )

    def test_artifact_size_checker_function(self):
        """Verify get_artifact_size returns correct structure."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ptz", delete=False) as mf:
            mf.write(b"x" * 1000)
            model_path = mf.name
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as cf:
            cf.write("# test code\n" * 100)
            code_path = cf.name
        try:
            result = get_artifact_size(model_path, code_path)
            assert "total_bytes" in result
            assert "under_limit" in result
            assert "headroom_bytes" in result
            assert result["total_bytes"] == result["model_bytes"] + result["code_bytes"]
        finally:
            os.unlink(model_path)
            os.unlink(code_path)

    def test_zstd_smaller_than_zlib(self):
        """zstd-22 should produce smaller output than zlib-9 for model weights."""
        import zlib
        try:
            import zstandard as zstd
        except ImportError:
            pytest.skip("zstandard not installed")
        import torch
        data = torch.randn(1000, 1000).numpy().tobytes()
        zlib_size = len(zlib.compress(data, level=9))
        zstd_size = len(zstd.ZstdCompressor(level=22).compress(data))
        assert zstd_size <= zlib_size * 1.05, \
            f"zstd-22 ({zstd_size:,}) should be <= zlib-9 ({zlib_size:,})"
