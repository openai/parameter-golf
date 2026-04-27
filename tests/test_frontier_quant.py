from __future__ import annotations

import unittest

import torch

from frontier_quant import dequantize_state_dict, quant_config_from_env, quantize_state_dict


class FrontierQuantTest(unittest.TestCase):
    def test_quant_config_from_env_resolves_overrides(self) -> None:
        config = quant_config_from_env(
            {
                "QUANT_POLICY": "grouped_sdclip",
                "GPTQ_CALIBRATION_STRATEGY": "mixed",
                "GPTQ_MODE": "full",
                "GPTQ_CALIBRATION_BATCHES": "48",
                "MATRIX_BITS": "5",
                "EMBED_CLIP_SIGMAS": "21.5",
                "COMPRESSOR": "zlib",
            }
        )
        self.assertEqual(config.policy_name, "grouped_sdclip")
        self.assertEqual(config.calibration_strategy, "mixed")
        self.assertEqual(config.calibration_batches, 48)
        self.assertEqual(config.bits_by_class["attn"], 5)
        self.assertEqual(config.bits_by_class["mlp"], 5)
        self.assertEqual(config.clip_sigmas_by_class["embed"], 21.5)

    def test_quantize_state_dict_uses_gptq_when_hessian_present(self) -> None:
        config = quant_config_from_env(
            {
                "QUANT_POLICY": "sdclip",
                "GPTQ_MODE": "full",
                "GPTQ_CALIBRATION_BATCHES": "8",
                "QUANT_PASSTHROUGH_MAX_NUMEL": "0",
                "COMPRESSOR": "zlib",
            }
        )
        state_dict = {
            "blocks.0.attn.c_q.weight": torch.randn(16, 16, dtype=torch.float32),
            "blocks.0.attn.q_gain": torch.ones(4, dtype=torch.float32),
        }
        hessians = {
            "blocks.0.attn.c_q.weight": torch.eye(16, dtype=torch.float32),
        }
        payload, summary = quantize_state_dict(state_dict, config=config, hessians=hessians)
        qmeta = payload["qmeta"]["blocks.0.attn.c_q.weight"]
        self.assertEqual(qmeta["method"], "gptq")
        self.assertEqual(qmeta["tensor_class"], "attn")
        self.assertEqual(summary["tensor_classes"]["attn"]["bits"], 6)

        restored = dequantize_state_dict(payload)
        self.assertEqual(restored["blocks.0.attn.c_q.weight"].shape, state_dict["blocks.0.attn.c_q.weight"].shape)
        self.assertTrue(torch.equal(restored["blocks.0.attn.q_gain"], state_dict["blocks.0.attn.q_gain"]))


if __name__ == "__main__":
    unittest.main()
