from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "records"
    / "track_10min_16mb"
    / "2026-03-17_LoRA_TTT"
    / "train_gpt.py"
)
_spec = spec_from_file_location("track_lora_ttt_train_gpt", MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load module from {MODULE_PATH}")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
_find_docs = _module._find_docs


def test_find_docs_falls_back_when_no_bos_tokens_present():
    tokens = torch.tensor([42, 7, 9, 11], dtype=torch.int64)
    assert _find_docs(tokens) == [(0, 4)]


def test_find_docs_skips_trailing_bos_only_fragment():
    tokens = torch.tensor([1, 2, 1], dtype=torch.int64)
    assert _find_docs(tokens) == [(0, 3)]


def test_find_docs_raises_for_too_short_sequence_without_bos():
    tokens = torch.tensor([99], dtype=torch.int64)
    try:
        _find_docs(tokens)
    except ValueError as exc:
        assert "at least 2 tokens" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid short sequence")
