"""USER-EDITABLE tokenizer builder demos.

Copy these functions into your own module, or point tokenizer specs at a
different file/module entirely.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from pure_byte_tokenizer import default_pure_byte_tokenizer


TOKENIZER_THREADS = max(1, int(os.environ.get("MATCHED_FINEWEB_TOKENIZER_THREADS", str(os.cpu_count() or 8))))


def _iter_sentencepiece_text(docs_jsonl: Path, *, max_docs: int | None = None):
    with Path(docs_jsonl).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
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
        "encode_batch": tok.encode_batch,
        "manifest": {"path": str(path), "pad_id": tok.pad_id, "unk_id": tok.unk_id},
    }


def build_sentencepiece_tokenizer(*, spec, docs_jsonl, tokenizers_dir):
    import sentencepiece as spm

    vocab_size = int(spec["vocab_size"])
    prefix = Path(tokenizers_dir) / spec.get("model_prefix", f"fineweb_{vocab_size}_bpe")
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for artifact in (model_path, vocab_path):
        if artifact.exists():
            artifact.unlink()

    reuse_model_path = spec.get("reuse_model_path")
    if reuse_model_path is not None:
        reuse_model_path = Path(reuse_model_path).expanduser().resolve()
        if not reuse_model_path.is_file():
            raise FileNotFoundError(reuse_model_path)
        shutil.copy2(reuse_model_path, model_path)
        reuse_vocab_path = reuse_model_path.with_suffix(".vocab")
        if reuse_vocab_path.is_file():
            shutil.copy2(reuse_vocab_path, vocab_path)
        print(
            f"Reusing SentencePiece tokenizer name={spec.get('name', f'sp_bpe_{vocab_size}')} "
            f"vocab={vocab_size} model={reuse_model_path}"
        )
    else:
        print(f"Training SentencePiece tokenizer name={spec.get('name', f'sp_bpe_{vocab_size}')} vocab={vocab_size}")
        kwargs = {
            "sentence_iterator": _iter_sentencepiece_text(
                Path(docs_jsonl),
                max_docs=(
                    None
                    if spec.get("tokenizer_train_docs") is None
                    else int(spec["tokenizer_train_docs"])
                ),
            ),
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

    tok = spm.SentencePieceProcessor(model_file=str(model_path))
    return {
        "name": spec.get("name", f"sp_bpe_{vocab_size}"),
        "kind": "sentencepiece_bpe",
        "dataset_suffix": spec.get("dataset_suffix", f"sp{vocab_size}"),
        "vocab_size": int(tok.vocab_size()),
        "bos_id": int(tok.bos_id()),
        "eos_id": int(tok.eos_id()),
        "encode": lambda text, tok=tok: tok.encode(text, out_type=int),
        "encode_batch": lambda texts, tok=tok: tok.encode(texts, out_type=int, num_threads=TOKENIZER_THREADS),
        "manifest": {"model_path": str(model_path), "vocab_path": str(vocab_path)},
    }
