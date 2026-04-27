from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - dependency availability varies by environment
    spm = None


DATA_MANIFEST_PATH = Path("data/manifest.json")
TOKENIZER_SPECS_PATH = Path("data/tokenizer_specs.json")


@dataclass(frozen=True)
class TokenizerVariantSpec:
    name: str
    dataset_suffix: str
    vocab_size: int
    tokenizer_kind: str = "sentencepiece_bpe"
    model_path: str | None = None
    dataset_path: str | None = None
    notes: tuple[str, ...] = ()


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def sentencepiece_tokenizer_stats(model_path: str) -> dict[str, object]:
    if spm is None:
        raise ImportError("sentencepiece is required to inspect tokenizer stats")
    tokenizer_file = Path(model_path)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    pieces = [sp.id_to_piece(i) for i in range(int(sp.vocab_size()))]
    byte_pieces = sum(1 for i in range(int(sp.vocab_size())) if sp.is_byte(i))
    control_pieces = sum(1 for i in range(int(sp.vocab_size())) if sp.is_control(i))
    avg_piece_utf8_bytes = float(np.mean([len(piece.encode("utf-8")) for piece in pieces])) if pieces else 0.0
    leading_space_pieces = sum(1 for piece in pieces if piece.startswith("\u2581"))
    return {
        "model_path": str(tokenizer_file),
        "vocab_size": int(sp.vocab_size()),
        "byte_pieces": byte_pieces,
        "control_pieces": control_pieces,
        "leading_space_pieces": leading_space_pieces,
        "avg_piece_utf8_bytes": round(avg_piece_utf8_bytes, 4),
    }


def tokenizer_variant_specs(path: str | Path = TOKENIZER_SPECS_PATH) -> list[TokenizerVariantSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = payload.get("tokenizers", payload.get("tokenizer_specs", []))
    variants: list[TokenizerVariantSpec] = []
    for item in specs:
        variants.append(
            TokenizerVariantSpec(
                name=str(item["name"]),
                dataset_suffix=str(item["dataset_suffix"]),
                vocab_size=int(item["vocab_size"]),
                tokenizer_kind=str(item.get("tokenizer_kind", "sentencepiece_bpe")),
                model_path=item.get("model_path"),
                dataset_path=item.get("dataset_path"),
                notes=tuple(item.get("notes", [])),
            )
        )
    return variants


def append_tokenizer_variant_spec(
    payload: dict[str, object],
    spec: TokenizerVariantSpec,
) -> dict[str, object]:
    updated = json.loads(json.dumps(payload))
    tokenizers = list(updated.get("tokenizers") or updated.get("tokenizer_specs") or [])
    tokenizers = [item for item in tokenizers if item.get("name") != spec.name]
    tokenizers.append(
        {
            "name": spec.name,
            "dataset_suffix": spec.dataset_suffix,
            "vocab_size": spec.vocab_size,
            "tokenizer_kind": spec.tokenizer_kind,
            "model_path": spec.model_path,
            "dataset_path": spec.dataset_path,
            "notes": list(spec.notes),
        }
    )
    if "tokenizers" in updated:
        updated["tokenizers"] = tokenizers
    else:
        updated["tokenizer_specs"] = tokenizers
    return updated


def recommended_bigram_vocab_size(vocab_size: int) -> int:
    return max(2048, int(round(vocab_size * 1.25 / 256.0) * 256))


def dataset_name_for_suffix(dataset_suffix: str) -> str:
    return f"fineweb10B_{dataset_suffix}"


def dataset_path_for_suffix(dataset_suffix: str) -> str:
    return f"datasets/{dataset_name_for_suffix(dataset_suffix)}"


def tokenizer_model_path_for_suffix(dataset_suffix: str, vocab_size: int) -> str:
    return f"tokenizers/fineweb_{vocab_size}_bpe.model"
