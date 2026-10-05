"""Compare both byte lookup tables with a real SentencePiece decoder."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import build_sentencepiece_luts as torch_luts


@pytest.fixture(params=[False, True], ids=["multiword", "word-boundaries"])
def tokenizer(request):
    model = io.BytesIO()
    sentences = ["alpha  beta gamma", "alpha beta  gamma", "alpha beta gamma", "café  beta"] * 60
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_writer=model,
        vocab_size=80,
        model_type="bpe",
        hard_vocab_limit=False,
        split_by_whitespace=request.param,
        allow_whitespace_only_pieces=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        minloglevel=2,
    )
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


def lookup(sp, backend, extra=0):
    if backend == "torch":
        return tuple(
            value.numpy() for value in torch_luts(sp, sp.vocab_size() + extra, torch.device("cpu"))
        )
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    from train_gpt_mlx import build_sentencepiece_luts as mlx_luts

    return mlx_luts(sp, sp.vocab_size() + extra)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize(
    "text", ["alpha beta gamma", "alpha  beta gamma", "alpha beta  gamma", "café  beta", "alpha"]
)
def test_decoded_bytes_match_the_actual_tokenizer(tokenizer, backend, text):
    base, leading, boundary = lookup(tokenizer, backend)
    tokens = np.array(tokenizer.encode(text, add_bos=True))
    previous, targets = tokens[:-1], tokens[1:]
    counted = int(np.sum(base[targets].astype(np.int64) + (leading[targets] & ~boundary[previous])))
    decoded = tokenizer.decode(tokens.tolist())
    assert decoded == text
    assert counted == len(decoded.encode("utf-8"))
    # For fixed likelihoods, the BPB denominator must use decoded bytes.
    nll = np.arange(1, len(targets) + 1, dtype=np.float64).sum()
    assert nll / (counted * np.log(2)) == pytest.approx(nll / (len(text.encode()) * np.log(2)))


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_piece_bytes_and_boundary_flags(tokenizer, backend):
    base, leading, boundary = lookup(tokenizer, backend, extra=3)
    for token in range(tokenizer.vocab_size()):
        if tokenizer.is_control(token) or tokenizer.is_unknown(token) or tokenizer.is_unused(token):
            assert base[token] == 0 and boundary[token]
            continue
        piece = tokenizer.id_to_piece(token)
        assert int(base[token]) == len(tokenizer.decode([token]).encode("utf-8"))
        assert bool(leading[token]) == piece.startswith("▁")
        assert not boundary[token]
    np.testing.assert_array_equal(base[-3:], 0)
    np.testing.assert_array_equal(leading[-3:], False)
    np.testing.assert_array_equal(boundary[-3:], True)
