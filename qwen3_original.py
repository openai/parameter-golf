"""Evaluate Qwen/Qwen3-1.7B-Base on the FineWeb validation set.

The validation set is the fixed first 50k-doc FineWeb split used by the
parameter-golf challenge. We reuse the repo's published docs cache when
available, tokenize the validation docs with the Qwen tokenizer, and report
both token cross-entropy loss and tokenizer-agnostic bits-per-byte.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from time import time
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = os.environ.get("NEMO_MODEL_ID", "Qwen/Qwen3-1.7B-FP8")
DEFAULT_DOCS_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DEFAULT_REMOTE_ROOT = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DEFAULT_DOCS_JSONL = ROOT / "data" / "docs_selected.jsonl"
DEFAULT_DOCS_SIDECAR = ROOT / "data" / "docs_selected.source_manifest.json"
DEFAULT_MAX_DOCS = int(os.environ.get("NEMO_MAX_DOCS", "50000"))
DEFAULT_DOC_BATCH_SIZE = int(os.environ.get("NEMO_DOC_BATCH_SIZE", "256"))
DEFAULT_SEQ_LEN = int(os.environ.get("NEMO_SEQ_LEN", "2048"))
DEFAULT_EVAL_BATCH_SEQS = int(os.environ.get("NEMO_EVAL_BATCH_SEQS", "1"))
DEFAULT_BYTE_LUT_BATCH = int(os.environ.get("NEMO_BYTE_LUT_BATCH", "4096"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen/Qwen3-1.7B-FP8 on FineWeb validation.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to load.")
    parser.add_argument(
        "--docs-jsonl",
        default=str(DEFAULT_DOCS_JSONL),
        help="Path to docs_selected.jsonl. If missing, the script falls back to downloading it from HF.",
    )
    parser.add_argument(
        "--docs-sidecar",
        default=str(DEFAULT_DOCS_SIDECAR),
        help="Path to docs_selected.source_manifest.json. Used for the default validation doc count.",
    )
    parser.add_argument(
        "--docs-repo-id",
        default=DEFAULT_DOCS_REPO_ID,
        help="Hugging Face dataset repo id used to fetch docs_selected.jsonl if needed.",
    )
    parser.add_argument(
        "--docs-remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help="Remote root prefix inside the dataset repo.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of validation documents to evaluate.",
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=DEFAULT_DOC_BATCH_SIZE,
        help="Number of documents to tokenize together per batch.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Sequence length used for evaluation windows.",
    )
    parser.add_argument(
        "--eval-batch-seqs",
        type=int,
        default=DEFAULT_EVAL_BATCH_SEQS,
        help="Number of evaluation sequences to score per forward pass.",
    )
    parser.add_argument(
        "--byte-lut-batch",
        type=int,
        default=DEFAULT_BYTE_LUT_BATCH,
        help="Chunk size used to precompute tokenizer byte lengths.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the model repository to provide custom code.",
    )
    return parser


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_hf_dataset_file(repo_id: str, remote_root: str, filename: str, destination: Path) -> Path:
    if destination.is_file():
        return destination

    remote_path = Path(remote_root) / filename if remote_root else Path(filename)
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def load_sidecar(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"docs sidecar must be a JSON object: {path}")
    return payload


def resolve_docs_path(args: argparse.Namespace) -> tuple[Path, dict[str, object] | None, int]:
    docs_path = Path(args.docs_jsonl)
    sidecar_path = Path(args.docs_sidecar)
    if not docs_path.is_file():
        try:
            fetch_hf_dataset_file(args.docs_repo_id, args.docs_remote_root, docs_path.name, docs_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not find docs_selected.jsonl at {docs_path} and HF download failed. "
                "Run `python3 data/cached_challenge_fineweb.py --variant sp1024 --with-docs` first "
                "or provide `--docs-jsonl` with a local path."
            ) from exc
    sidecar = load_sidecar(sidecar_path)
    if sidecar is None:
        try:
            fetch_hf_dataset_file(args.docs_repo_id, args.docs_remote_root, sidecar_path.name, sidecar_path)
        except Exception:
            sidecar = None
        else:
            sidecar = load_sidecar(sidecar_path)

    default_max_docs = int(sidecar.get("num_val_docs", DEFAULT_MAX_DOCS)) if sidecar is not None else DEFAULT_MAX_DOCS
    max_docs = args.max_docs if args.max_docs is not None else default_max_docs
    if max_docs <= 0:
        raise ValueError(f"max_docs must be positive, got {max_docs}")
    return docs_path, sidecar, max_docs


def iter_validation_docs(docs_path: Path, max_docs: int) -> Iterator[str]:
    with docs_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            yield json.loads(line)["text"].replace("\x00", " ").strip()


def batched_docs(docs: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for text in docs:
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_token_byte_luts(tokenizer, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocab_size = len(tokenizer)
    base_bytes_lut = np.zeros((vocab_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((vocab_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((vocab_size,), dtype=np.bool_)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for start in range(0, vocab_size, batch_size):
        stop = min(start + batch_size, vocab_size)
        token_ids = list(range(start, stop))
        token_texts = tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id, raw_text in zip(token_ids, token_texts, strict=True):
            if token_id in special_ids:
                continue
            is_boundary_token_lut[token_id] = False
            piece = raw_text or ""
            if piece.startswith(("▁", "Ġ")):
                has_leading_space_lut[token_id] = True
                piece = piece[1:]
            elif piece.startswith(" "):
                has_leading_space_lut[token_id] = True
                # Match transfer.py semantics: track exactly one synthetic leading-space byte.
                piece = piece[1:]
            base_bytes_lut[token_id] = len(piece.encode("utf-8")) if piece else 0
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def tokenized_doc_batch(tokenizer, texts: list[str], bos_token_id: int | None) -> list[np.ndarray]:
    if not texts:
        return []
    encoded_docs = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if encoded_docs and isinstance(encoded_docs[0], int):
        encoded_docs = [encoded_docs]
    out: list[np.ndarray] = []
    for ids in encoded_docs:
        ids_arr = np.asarray(ids, dtype=np.int64)
        if bos_token_id is not None:
            ids_arr = np.concatenate((np.asarray([bos_token_id], dtype=np.int64), ids_arr))
        out.append(ids_arr)
    return out


def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    docs_path: Path,
    max_docs: int,
    doc_batch_size: int,
    seq_len: int,
    eval_batch_seqs: int,
) -> tuple[float, float, int, int, int]:
    device = next(model.parameters()).device
    autocast_enabled = device.type == "cuda"
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    docs_seen = 0
    tokens_seen = 0
    pending_ids = np.empty((0,), dtype=np.int64)
    bos_token_id = tokenizer.bos_token_id

    model.eval()
    with torch.inference_mode():
        for docs_batch in batched_docs(iter_validation_docs(docs_path, max_docs), doc_batch_size):
            docs_seen += len(docs_batch)
            ids_batch = tokenized_doc_batch(tokenizer, docs_batch, bos_token_id)
            if not ids_batch:
                continue
            flat_ids = np.concatenate(ids_batch) if len(ids_batch) > 1 else ids_batch[0]
            tokens_seen += int(flat_ids.size)
            if pending_ids.size:
                flat_ids = np.concatenate((pending_ids, flat_ids))

            if flat_ids.size < seq_len + 1:
                pending_ids = flat_ids
                continue

            num_chunks = 1 + (flat_ids.size - (seq_len + 1)) // seq_len
            for chunk_start in range(0, num_chunks, eval_batch_seqs):
                batch_starts = [i * seq_len for i in range(chunk_start, min(chunk_start + eval_batch_seqs, num_chunks))]
                batch_ids = [flat_ids[start : start + seq_len + 1] for start in batch_starts]
                batch_inputs = np.stack([chunk[:-1] for chunk in batch_ids])
                batch_targets = np.stack([chunk[1:] for chunk in batch_ids])
                inputs = torch.from_numpy(batch_inputs).to(device=device, dtype=torch.long, non_blocking=True)
                targets = torch.from_numpy(batch_targets).to(device=device, dtype=torch.long, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    outputs = model(input_ids=inputs, use_cache=False)
                    logits = outputs.logits
                    batch_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="mean",
                    )
                batch_tokens = int(targets.numel())
                loss_sum += batch_loss.to(torch.float64) * batch_tokens
                token_count += batch_tokens
                prev_ids = batch_inputs.reshape(-1)
                tgt_ids = batch_targets.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].astype(np.int64, copy=False)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int64, copy=False)
                byte_count += int(token_bytes.sum())

            next_start = num_chunks * seq_len
            pending_ids = flat_ids[next_start:]

    if token_count.item() == 0 or byte_count.item() == 0:
        raise ValueError("Validation corpus produced no scored tokens or bytes")

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    val_bpb = bits_per_token * tokens_per_byte
    return float(val_loss.item()), float(val_bpb), docs_seen, tokens_seen, int(byte_count.item())


def main() -> None:
    args = build_parser().parse_args()
    docs_path, sidecar, max_docs = resolve_docs_path(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        dtype=torch.bfloat16,
    )

    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for evaluation")
    model = model.to(device)

    model_vocab_size = getattr(model.config, "vocab_size", None)
    tokenizer_vocab_size = len(tokenizer)
    if model_vocab_size is not None and tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer_vocab_size} exceeds model vocab size {model_vocab_size}"
        )

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_token_byte_luts(tokenizer, args.byte_lut_batch)
    log(f"model_id:{args.model_id}")
    log(f"tokenizer_vocab_size:{tokenizer_vocab_size} model_vocab_size:{model_vocab_size}")
    log(f"docs_jsonl:{docs_path}")
    if sidecar is not None:
        log(
            "docs_manifest:"
            f" num_docs={sidecar.get('num_docs', 'unknown')} num_val_docs={sidecar.get('num_val_docs', 'unknown')}"
        )
    log(
        f"eval_config docs:{max_docs} doc_batch_size:{args.doc_batch_size} seq_len:{args.seq_len} "
        f"eval_batch_seqs:{args.eval_batch_seqs}"
    )
    log(f"validation_stream:bos_token_id:{tokenizer.bos_token_id} doc_boundaries:preserved")

    t0 = time.performance_counter()
    val_loss, val_bpb, docs_seen, tokens_seen, byte_count = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        docs_path=docs_path,
        max_docs=max_docs,
        doc_batch_size=args.doc_batch_size,
        seq_len=args.seq_len,
        eval_batch_seqs=args.eval_batch_seqs,
    )
    t1 = time.performance_counter()
    elapsed = t1 - t0

    log(
        f"val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f} docs_seen:{docs_seen} "
        f"tokens_seen:{tokens_seen} bytes_seen:{byte_count}"
        f" eval_time_sec:{elapsed:.2f}"
    )


if __name__ == "__main__":
    main()