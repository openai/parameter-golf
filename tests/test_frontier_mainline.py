from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from train_gpt_frontier_mainline import GPT, Hyperparameters, mainline_surface_record


class FrontierMainlineTest(unittest.TestCase):
    def test_surface_record_resolves_recurrence_and_track(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
                "VOCAB_SIZE": "8192",
                "NUM_LAYERS": "8",
                "RECUR_LAYERS": "2,3,4",
                "NUM_LOOPS": "2",
                "PARALLEL_START_LAYER": "5",
                "TTT_ENABLED": "1",
                "COMPRESSOR": "zlib",
            },
            clear=False,
        ):
            args = Hyperparameters.from_env()
            surface = mainline_surface_record(args)
        self.assertEqual(args.loop_start, 2)
        self.assertEqual(args.loop_end, 4)
        self.assertEqual(args.parallel_residual_start, 5)
        self.assertEqual(surface["track"], "score_first_ttt")

    def test_gpt_forward_on_cpu_with_loop_and_parallel_controls(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "VOCAB_SIZE": "64",
                "NUM_LAYERS": "6",
                "MODEL_DIM": "32",
                "EMBEDDING_DIM": "32",
                "NUM_HEADS": "4",
                "NUM_KV_HEADS": "2",
                "MLP_MULT": "2.0",
                "XSA_LAST_N": "0",
                "RECUR_LAYERS": "1,2",
                "NUM_LOOPS": "1",
                "PARALLEL_START_LAYER": "4",
                "PER_PASS_EMBEDDINGS": "1",
                "TIE_EMBEDDINGS": "1",
                "COMPRESSOR": "zlib",
            },
            clear=False,
        ):
            args = Hyperparameters.from_env()
        model = GPT(args)
        model.looping_active = True
        tokens = torch.randint(0, args.vocab_size, (2, 8), dtype=torch.int64)
        logits = model.forward_logits(tokens)
        arch = model.describe_architecture()
        self.assertEqual(tuple(logits.shape), (2, 8, args.vocab_size))
        self.assertEqual(arch["parallel_residual_layers"], [4, 5])
        self.assertTrue(arch["per_pass_embeddings"])


if __name__ == "__main__":
    unittest.main()
