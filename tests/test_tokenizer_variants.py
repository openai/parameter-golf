from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frontier_tokenizer import TokenizerVariantSpec, append_tokenizer_variant_spec, tokenizer_variant_specs


class TokenizerVariantTest(unittest.TestCase):
    def test_append_variant_spec_replaces_existing_name(self) -> None:
        payload = {"tokenizers": [{"name": "sp_bpe_8192", "dataset_suffix": "sp8192", "vocab_size": 8192}]}
        updated = append_tokenizer_variant_spec(
            payload,
            TokenizerVariantSpec(
                name="sp_bpe_8192",
                dataset_suffix="sp8192",
                vocab_size=8192,
                model_path="tokenizers/fineweb_8192_bpe.model",
                dataset_path="datasets/fineweb10B_sp8192",
                notes=("stable",),
            ),
        )
        self.assertEqual(len(updated["tokenizers"]), 1)
        self.assertEqual(updated["tokenizers"][0]["model_path"], "tokenizers/fineweb_8192_bpe.model")

    def test_tokenizer_variant_specs_parses_current_spec_file_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "tokenizer_specs.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "tokenizers": [
                            {
                                "name": "sp_bpe_7680",
                                "dataset_suffix": "sp7680",
                                "vocab_size": 7680,
                                "model_path": "tokenizers/fineweb_7680_bpe.model",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            specs = tokenizer_variant_specs(spec_path)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].name, "sp_bpe_7680")
            self.assertEqual(specs[0].vocab_size, 7680)


if __name__ == "__main__":
    unittest.main()

