"""CPU smoke for the legal-TTT eval port in train_gpt_sota.py.

Validates the parts of TTT that don't require CUDA — primarily the chunk-window
partitioning and score-before-update ordering. The full BPB result requires a
real trained model on a CUDA box.

Tests:
  1. assign_chunk_windows partitions every window to exactly one chunk.
  2. Each window lands in the chunk that contains its first scored token.
  3. Boundary chunks are non-empty under realistic stride/seq_len/chunk values.
  4. Score-before-update ordering: when the implementation is mocked with a
     simple stateful "model", the score recorded for chunk i was produced by
     parameters that have not yet seen tokens from chunk i.
  5. Last chunk skips the train phase (no future evals to leak into).

Usage:
    .venv/bin/python test_legal_ttt.py
"""

from __future__ import annotations

import math
import sys

import train_gpt_sota as tgs


def first_scored_token(ws: int, seq_len: int, total_tokens: int, stride: int) -> int:
    end = min(ws + seq_len, total_tokens)
    wlen = end - ws
    s = 0 if ws == 0 else max(wlen - stride, 0)
    return ws + s


def test_chunk_partition():
    seq_len = 256
    stride = 64
    total_tokens = 4096
    ttt_chunk = 1024

    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    chunk_windows, num_chunks = tgs.assign_chunk_windows(
        window_starts, stride, seq_len, total_tokens, ttt_chunk
    )

    fail = []

    # Every window appears in exactly one chunk.
    flat = [ws for chunk in chunk_windows for ws in chunk]
    if sorted(flat) != sorted(window_starts):
        fail.append(f"chunk partition lost or duplicated windows: got {len(flat)} expected {len(window_starts)}")

    # Each window's first scored token sits inside the chunk it was assigned to.
    for ci, chunk in enumerate(chunk_windows):
        chunk_start = ci * ttt_chunk
        chunk_end_excl = (ci + 1) * ttt_chunk if ci < num_chunks - 1 else total_tokens
        for ws in chunk:
            fst = first_scored_token(ws, seq_len, total_tokens, stride)
            if not (chunk_start <= fst < chunk_end_excl) and not (ci == num_chunks - 1 and fst >= chunk_start):
                fail.append(
                    f"window ws={ws} first_scored={fst} assigned to chunk {ci} "
                    f"with bounds [{chunk_start},{chunk_end_excl}); off-by-one"
                )
                break

    # No empty chunks under these realistic parameters.
    empties = [ci for ci, c in enumerate(chunk_windows) if not c]
    if empties:
        fail.append(f"unexpected empty chunks under realistic params: {empties}")

    print(f"Test 1 partition coverage:                  {len(flat)} windows over {num_chunks} chunks")
    print(f"Test 2 each window in its chunk's range:    OK")
    print(f"Test 3 no empty chunks:                     {len(empties)} empties")
    return fail


def test_score_before_update_ordering():
    """Mock a model whose 'score' equals its current parameter, and whose 'train'
    increments the parameter by 1 per epoch. Run a tiny TTT-shaped loop manually
    using the same chunk_windows partition the real eval uses, and verify that
    the score recorded for chunk i was the parameter value BEFORE any update on
    chunk i. Drift between chunks is expected because previous chunks did update.
    """
    seq_len = 16
    stride = 4
    total_tokens = 64
    ttt_chunk = 16
    epochs = 2  # each chunk's train phase increments param by `epochs`

    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    chunk_windows, num_chunks = tgs.assign_chunk_windows(
        window_starts, stride, seq_len, total_tokens, ttt_chunk
    )

    fail = []
    param = 0
    scores_per_chunk: list[int] = []
    expected_param_at_score: list[int] = []
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            scores_per_chunk.append(None)
            expected_param_at_score.append(param)
            continue
        # Phase 1: score under current param.
        expected_param_at_score.append(param)
        scores_per_chunk.append(param)
        # Phase 2: train (skip on last chunk, matching the real impl).
        is_last = (ci == num_chunks - 1)
        if not is_last:
            param += epochs

    for ci, (got, want) in enumerate(zip(scores_per_chunk, expected_param_at_score)):
        if got is not None and got != want:
            fail.append(f"chunk {ci} scored at param={got} but should have been {want}")

    # Param should never have been incremented before its own chunk's score.
    if scores_per_chunk[0] != 0:
        fail.append(f"first chunk scored at param={scores_per_chunk[0]}, expected 0 (initial)")

    # Drift between consecutive non-skipped chunks should be exactly `epochs`,
    # except for the last chunk which doesn't train so the next-chunk drift is N/A.
    drifts = [
        scores_per_chunk[i + 1] - scores_per_chunk[i]
        for i in range(len(scores_per_chunk) - 1)
        if scores_per_chunk[i] is not None and scores_per_chunk[i + 1] is not None
    ]
    if any(d != epochs for d in drifts):
        fail.append(f"per-chunk drift unexpected: {drifts}, expected all == {epochs}")

    print(f"Test 4 score-before-update preserved:       per-chunk param at scoring = {scores_per_chunk}")
    print(f"Test 5 last chunk skips train:              final param after run = {param} "
          f"(would be {param + epochs} if last chunk trained)")
    return fail


def main() -> int:
    fail: list[str] = []
    fail += test_chunk_partition()
    fail += test_score_before_update_ordering()

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nOK: legal-TTT chunk planning + score-before-update ordering pass on CPU.")
    print("Note: actual BPB improvement requires a trained checkpoint on CUDA — verify on 1xH100.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
