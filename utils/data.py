"""FineWeb data loading utilities for byte-level language modeling."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "fineweb_bytes"


def download_fineweb(target_bytes: int = 500_000_000, split: str = "train"):
    """Download FineWeb sample and cache raw bytes locally."""
    cache_file = CACHE_DIR / f"fineweb_{split}.bin"
    if cache_file.exists() and cache_file.stat().st_size >= target_bytes:
        print(f"[data] Using cached {cache_file} ({cache_file.stat().st_size:,} bytes)")
        return cache_file
    elif cache_file.exists():
        print(f"[data] Cached file too small ({cache_file.stat().st_size:,} < {target_bytes:,}), re-downloading...")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading FineWeb {split} split (~{target_bytes // 1_000_000}MB)...")

    from datasets import load_dataset

    if split == "train":
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        collected = bytearray()
        for example in ds:
            text = example["text"]
            collected.extend(text.encode("utf-8"))
            if len(collected) >= target_bytes:
                break
        data = bytes(collected[:target_bytes])
    else:
        # Validation: use a separate slice via skip/take (fast, no 500MB scan)
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        collected = bytearray()
        n_skip = 50_000  # skip first 50k docs (used for training)
        n_seen = 0
        for example in ds:
            n_seen += 1
            if n_seen <= n_skip:
                continue
            collected.extend(example["text"].encode("utf-8"))
            if len(collected) >= 10_000_000:  # 10MB validation
                break
        data = bytes(collected[:10_000_000])

    with open(cache_file, "wb") as f:
        f.write(data)
    print(f"[data] Saved {len(data):,} bytes to {cache_file}")
    return cache_file


def load_bytes(path: Path) -> np.ndarray:
    """Load a binary file as a uint8 numpy array."""
    return np.fromfile(str(path), dtype=np.uint8)


def load_fineweb_train(target_bytes: int = 500_000_000) -> np.ndarray:
    """Load training data as byte array."""
    cache_file = download_fineweb(target_bytes, split="train")
    return load_bytes(cache_file)


def load_fineweb_valid() -> np.ndarray:
    """Load validation data as byte array."""
    cache_file = download_fineweb(10_000_000, split="valid")
    return load_bytes(cache_file)


class ByteDataset(IterableDataset):
    """Infinite random-offset byte sequence dataset for training."""

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            idx = random.randint(0, len(self.data) - self.seq_len - 2)
            chunk = self.data[idx: idx + self.seq_len + 1]
            x = torch.from_numpy(chunk[:-1].copy()).long()
            y = torch.from_numpy(chunk[1:].copy()).long()
            yield x, y


def make_byte_batches(data: np.ndarray, batch_size: int, seq_len: int):
    """Generator that yields random (input, target) batches of byte sequences."""
    max_start = len(data) - seq_len - 1
    while True:
        starts = [random.randint(0, max_start) for _ in range(batch_size)]
        xs, ys = [], []
        for s in starts:
            chunk = data[s: s + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        yield (
            torch.from_numpy(np.stack(xs)).long(),
            torch.from_numpy(np.stack(ys)).long(),
        )
