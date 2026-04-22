from __future__ import annotations

import ast
import importlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_MLX_RUNTIME = importlib.util.find_spec("mlx") is not None


class TrainGptMlxWrapperTests(unittest.TestCase):
    def test_wrapper_imports_runner_main(self) -> None:
        wrapper_path = ROOT / "train_gpt_mlx.py"
        module = ast.parse(wrapper_path.read_text(encoding="utf-8"))

        import_from = next(node for node in module.body if isinstance(node, ast.ImportFrom))
        self.assertEqual(import_from.module, "train_gpt_mlx_lib.runner")
        self.assertEqual([alias.name for alias in import_from.names], ["main"])

        if_stmt = next(node for node in module.body if isinstance(node, ast.If))
        compare = if_stmt.test
        self.assertIsInstance(compare, ast.Compare)
        self.assertEqual(compare.left.id, "__name__")
        self.assertEqual(compare.comparators[0].value, "__main__")
        self.assertEqual(if_stmt.body[0].value.func.id, "main")

    def test_internal_package_imports(self) -> None:
        if not HAS_MLX_RUNTIME:
            self.skipTest("mlx is not installed in this python environment")
        package = importlib.import_module("train_gpt_mlx_lib")
        runner = importlib.import_module("train_gpt_mlx_lib.runner")
        model = importlib.import_module("train_gpt_mlx_lib.model")

        self.assertTrue(callable(package.main))
        self.assertTrue(callable(runner.main))
        self.assertTrue(hasattr(model, "GPT"))


class TrainGptMlxHelperTests(unittest.TestCase):
    def test_token_chunks_preserves_original_chunking(self) -> None:
        if not HAS_MLX_RUNTIME:
            self.skipTest("mlx is not installed in this python environment")
        from train_gpt_mlx_lib.optim import token_chunks

        self.assertEqual(token_chunks(8192, 1024, 4096), [4096, 4096])
        self.assertEqual(token_chunks(9216, 1024, 3072), [3072, 3072, 3072])

    def test_quantization_roundtrip_preserves_shapes_and_dtypes(self) -> None:
        if not HAS_MLX_RUNTIME:
            self.skipTest("mlx is not installed in this python environment")
        import mlx.core as mx
        import numpy as np

        from train_gpt_mlx_lib.quantization import dequantize_state_dict_int8, quantize_state_dict_int8

        flat_state = {
            "big_matrix": mx.array(np.arange(70_000, dtype=np.float32).reshape(100, 700)),
            "control_vector": mx.array(np.linspace(-1.0, 1.0, 1024, dtype=np.float32)),
            "int_ids": mx.array(np.arange(16, dtype=np.int32)),
        }

        quant_obj, stats = quantize_state_dict_int8(flat_state)
        restored = dequantize_state_dict_int8(quant_obj)

        self.assertGreater(stats["int8_payload_bytes"], 0)
        self.assertEqual(restored["big_matrix"].shape, flat_state["big_matrix"].shape)
        self.assertEqual(restored["control_vector"].shape, flat_state["control_vector"].shape)
        self.assertEqual(restored["int_ids"].dtype, flat_state["int_ids"].dtype)


if __name__ == "__main__":
    unittest.main()
