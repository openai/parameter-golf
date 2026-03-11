"""USER-EDITABLE tokenizer builder demos.

Copy these functions into your own module, or point tokenizer specs at a
different file/module entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

from pure_byte_tokenizer import default_pure_byte_tokenizer


def _iter_sentencepiece_text(docs_jsonl: Path):
    with Path(docs_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"].replace("\x00", " ").strip()
            if text:
                yield text


def build_pure_byte_tokenizer(*, spec, docs_jsonl, tokenizers_dir):
    del docs_jsonl
    tok = default_pure_byte_tokenizer()
    path = Path(tokenizers_dir) / spec.get("filename", "fineweb_pure_byte_260.json")
    tok.save_json(path)
    print(f"Wrote byte tokenizer: {path}")
    return {
        "name": spec.get("name", "pure_byte_260"),
        "kind": "byte",
        "dataset_suffix": spec.get("dataset_suffix", "byte260"),
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "encode": tok.encode,
        "manifest": {"path": str(path), "pad_id": tok.pad_id, "unk_id": tok.unk_id},
    }


def build_sentencepiece_tokenizer(*, spec, docs_jsonl, tokenizers_dir):
    vocab_size = int(spec["vocab_size"])
    prefix = Path(tokenizers_dir) / spec.get("model_prefix", f"fineweb_{vocab_size}_bpe")
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    if not (model_path.exists() and vocab_path.exists()):
        print(f"Training SentencePiece tokenizer name={spec.get('name', f'sp_bpe_{vocab_size}')} vocab={vocab_size}")
        kwargs = {
            "sentence_iterator": _iter_sentencepiece_text(Path(docs_jsonl)),
            "model_prefix": str(prefix),
            "model_type": "bpe",
            "vocab_size": vocab_size,
            "character_coverage": 0.999,
            "byte_fallback": True,
            "split_digits": True,
            "normalization_rule_name": "nmt_nfkc",
            "add_dummy_prefix": False,
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "hard_vocab_limit": False,
        }
        kwargs.update(spec.get("trainer_overrides") or {})
        spm.SentencePieceTrainer.train(**kwargs)
    else:
        print(f"Reusing existing SentencePiece model: {model_path}")

    tok = spm.SentencePieceProcessor(model_file=str(model_path))
    return {
        "name": spec.get("name", f"sp_bpe_{vocab_size}"),
        "kind": "sentencepiece_bpe",
        "dataset_suffix": spec.get("dataset_suffix", f"sp{vocab_size}"),
        "vocab_size": int(tok.vocab_size()),
        "bos_id": int(tok.bos_id()),
        "eos_id": int(tok.eos_id()),
        "encode": lambda text, tok=tok: tok.encode(text, out_type=int),
        "manifest": {"model_path": str(model_path), "vocab_path": str(vocab_path)},
    }
