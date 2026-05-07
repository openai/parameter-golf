from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    targets = targets.astype(np.int64, copy=False)
    probs = softmax(logits, axis=-1)
    row_idx = np.arange(targets.shape[0], dtype=np.int64)
    chosen = np.clip(probs[row_idx, targets], 1e-12, 1.0)
    return -np.log(chosen)


def cross_entropy_from_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    targets = targets.astype(np.int64, copy=False)
    probs = np.asarray(probabilities, dtype=np.float64)
    row_idx = np.arange(targets.shape[0], dtype=np.int64)
    chosen = np.clip(probs[row_idx, targets], 1e-12, 1.0)
    return -np.log(chosen)


def bits_per_byte_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    losses = cross_entropy_from_logits(logits, targets)
    return float(np.mean(losses) / np.log(2.0))


def bits_per_byte_from_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> float:
    losses = cross_entropy_from_probabilities(probabilities, targets)
    return float(np.mean(losses) / np.log(2.0))


def bits_per_token_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    return bits_per_byte_from_logits(logits, targets)


def bits_per_token_from_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return bits_per_byte_from_probabilities(probabilities, targets)
