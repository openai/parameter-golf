import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from train_gpt_mlx import build_sliding_eval_windows, eval_val_non_overlapping, eval_val_sliding


class TrainGptMlxSlidingEvalTest(unittest.TestCase):
    def test_build_sliding_eval_windows_covers_each_token_once(self) -> None:
        windows = build_sliding_eval_windows(total_tokens=20, seq_len=8, stride=3)

        self.assertEqual(windows[0], (0, 8, 0, 8))

        scored = []
        for window_start, window_len, score_start, score_end in windows:
            self.assertGreaterEqual(window_len, 3)
            self.assertLessEqual(window_len, 8)
            self.assertGreaterEqual(score_start, 0)
            self.assertLessEqual(score_end, window_len)
            scored.extend(range(window_start + score_start, window_start + score_end))

        self.assertEqual(scored, list(range(20)))

    def test_build_sliding_eval_windows_matches_non_overlapping_when_stride_equals_seq_len(self) -> None:
        windows = build_sliding_eval_windows(total_tokens=16, seq_len=8, stride=8)

        self.assertEqual(
            windows,
            [
                (0, 8, 0, 8),
                (8, 8, 0, 8),
            ],
        )

    def test_sliding_eval_scores_fewer_low_context_positions(self) -> None:
        args = SimpleNamespace(
            val_batch_size=8,
            grad_accum_steps=1,
            train_seq_len=8,
            eval_stride=4,
            effective_eval_batch_seqs=2,
        )
        val_tokens = np.zeros((17,), dtype=np.int32)
        base_bytes_lut = np.ones((4,), dtype=np.int16)
        has_leading_space_lut = np.zeros((4,), dtype=bool)
        is_boundary_token_lut = np.zeros((4,), dtype=bool)

        def fake_forward_logits(x: mx.array) -> mx.array:
            x_np = np.array(x, dtype=np.int32, copy=False)
            logits = np.full((x_np.shape[0], x_np.shape[1], 4), -8.0, dtype=np.float32)
            for batch_idx in range(x_np.shape[0]):
                for token_idx in range(x_np.shape[1]):
                    predicted = 1 if token_idx == 0 else int(x_np[batch_idx, token_idx])
                    logits[batch_idx, token_idx, predicted] = 8.0
            return mx.array(logits)

        def fake_loss(x: mx.array, y: mx.array) -> mx.array:
            logits = fake_forward_logits(x)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        regular_loss, regular_bpb = eval_val_non_overlapping(
            args,
            fake_loss,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        sliding_loss, sliding_bpb = eval_val_sliding(
            args,
            fake_forward_logits,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )

        self.assertLess(sliding_loss, regular_loss)
        self.assertLess(sliding_bpb, regular_bpb)


if __name__ == "__main__":
    unittest.main()
