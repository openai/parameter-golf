"""Offline input-fidelity tests for the variance-analysis command."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure_variance_ratio as analysis


def write_shard(path, tokens, *, magic=20240520, version=1, declared=None, suffix=b""):
    values = np.asarray(tokens, dtype="<u2")
    header = np.zeros(256, dtype="<i4")
    header[:3] = [magic, version, len(values) if declared is None else declared]
    path.write_bytes(header.tobytes() + values.tobytes() + suffix)


def invoke(tmp_path, monkeypatch, vocab_size=32):
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({}, checkpoint)
    model = torch.nn.Module()
    captured = []
    monkeypatch.setattr(
        analysis,
        "load_module_from_path",
        lambda path: SimpleNamespace(Hyperparameters=SimpleNamespace(vocab_size=vocab_size)),
    )
    monkeypatch.setattr(analysis, "build_model", lambda *args: model)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)

    def measure(model, tokens, seq_len, device, model_type):
        captured.append(np.asarray(tokens).copy())
        return {0: {"layer": 0, "alpha": 0.0, "var_diff": 1.0, "var_raw": 2.0, "r_ell": 0.5}}

    monkeypatch.setattr(analysis, "measure_variance_ratio", measure)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_variance_ratio.py",
            "--model-type",
            "standard",
            "--script",
            "synthetic.py",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(tmp_path),
            "--seq-len",
            "4",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    analysis.main()
    return captured


@pytest.mark.parametrize("length", [8, 512, 8192, 8200])
def test_analysis_receives_tokens_without_header_or_clamping(tmp_path, monkeypatch, length):
    tokens = np.arange(length, dtype=np.uint16) % 32
    write_shard(tmp_path / "fineweb_val_000.bin", tokens)
    captured = invoke(tmp_path, monkeypatch)
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], tokens[:8192])
    assert captured[0].dtype == np.int32


@pytest.mark.parametrize(
    "kwargs", [{"magic": 7}, {"version": 2}, {"declared": 20}, {"declared": -1}, {"suffix": b"x"}]
)
def test_malformed_shards_are_rejected_before_measurement(tmp_path, monkeypatch, kwargs):
    write_shard(tmp_path / "fineweb_val_000.bin", [1, 2, 3, 4], **kwargs)
    with pytest.raises(ValueError):
        invoke(tmp_path, monkeypatch)
    assert not (tmp_path / "result.json").exists()


def test_short_header_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "fineweb_val_000.bin").write_bytes(b"\0" * 12)
    with pytest.raises(ValueError):
        invoke(tmp_path, monkeypatch)


@pytest.mark.parametrize("invalid", [32, 65535])
def test_out_of_vocabulary_tokens_fail_instead_of_being_rewritten(tmp_path, monkeypatch, invalid):
    write_shard(tmp_path / "fineweb_val_000.bin", [0, 1, invalid, 2])
    with pytest.raises(ValueError, match="vocab"):
        invoke(tmp_path, monkeypatch)
    assert not (tmp_path / "result.json").exists()


def test_missing_validation_files_keep_existing_error(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        invoke(tmp_path, monkeypatch)


def test_command_variance_uses_only_synthetic_token_activations(tmp_path, monkeypatch):
    import json

    class ToyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_v = torch.nn.Identity()

    class ToyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ToyAttention()

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([ToyBlock()])

        def forward(self, input_ids, target_ids):
            return self.blocks[0].attn.c_v(input_ids.float().unsqueeze(-1)).sum()

    tokens = np.array([0, 2, 4, 6, 3, 2, 1, 0], dtype=np.uint16)
    write_shard(tmp_path / "fineweb_val_000.bin", tokens)
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({}, checkpoint)
    monkeypatch.setattr(
        analysis,
        "load_module_from_path",
        lambda path: SimpleNamespace(Hyperparameters=SimpleNamespace(vocab_size=32)),
    )
    monkeypatch.setattr(analysis, "build_model", lambda *args: ToyModel())
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_variance_ratio.py",
            "--model-type",
            "standard",
            "--script",
            "synthetic.py",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(tmp_path),
            "--seq-len",
            "4",
            "--output",
            str(output),
        ],
    )
    analysis.main()
    result = json.loads(output.read_text())["0"]
    activations = tokens.reshape(2, 4)[:, :-1].astype(np.float64)
    var_raw = np.var(activations[:, 1:], ddof=1)
    var_diff = np.var(np.diff(activations, axis=1), ddof=1)
    assert result["var_raw"] == pytest.approx(var_raw)
    assert result["var_diff"] == pytest.approx(var_diff)
    assert result["r_ell"] == pytest.approx(round(var_diff / var_raw, 4))
