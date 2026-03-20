from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import _find_docs


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
