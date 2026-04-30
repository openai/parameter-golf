import os
import unittest

import mlx.core as mx

from train_gpt_mlx import dequantize_state_dict_int8, quantize_state_dict_int8


class TrainGptMlxQuantizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_large_patterns = os.environ.get("INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS")

    def tearDown(self) -> None:
        if self.original_large_patterns is None:
            os.environ.pop("INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS", None)
        else:
            os.environ["INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS"] = self.original_large_patterns

    def test_large_embedding_is_quantized_by_default(self) -> None:
        os.environ.pop("INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS", None)
        flat_state = {"tok_emb.weight": mx.ones((300, 300), dtype=mx.bfloat16)}

        quant_obj, _ = quantize_state_dict_int8(flat_state)

        self.assertIn("tok_emb.weight", quant_obj["quantized"])
        self.assertNotIn("tok_emb.weight", quant_obj["passthrough"])

    def test_large_embedding_can_be_kept_in_fp16_when_requested(self) -> None:
        os.environ["INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS"] = "tok_emb.weight"
        flat_state = {"tok_emb.weight": mx.ones((300, 300), dtype=mx.bfloat16)}

        quant_obj, _ = quantize_state_dict_int8(flat_state)
        restored = dequantize_state_dict_int8(quant_obj)

        self.assertNotIn("tok_emb.weight", quant_obj["quantized"])
        self.assertIn("tok_emb.weight", quant_obj["passthrough"])
        self.assertEqual(restored["tok_emb.weight"].dtype, mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
