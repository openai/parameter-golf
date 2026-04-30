import unittest

from train_gpt_mlx import GPT


class TrainGptMlxModelConfigTest(unittest.TestCase):
    def test_gpt_uses_explicit_mlp_hidden(self) -> None:
        model = GPT(
            vocab_size=32,
            num_layers=2,
            dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            mlp_hidden=24,
            logit_chunk_tokens=0,
            logit_softcap=30.0,
            rope_base=10_000.0,
            tied_embed_init_std=0.005,
            qk_gain_init=1.5,
        )

        self.assertEqual(model.blocks[0].mlp.fc.weight.shape, (24, 16))
        self.assertEqual(model.blocks[0].mlp.proj.weight.shape, (16, 24))

    def test_gpt_defaults_to_multiplier_for_mlp_hidden(self) -> None:
        model = GPT(
            vocab_size=32,
            num_layers=2,
            dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            mlp_hidden=0,
            logit_chunk_tokens=0,
            logit_softcap=30.0,
            rope_base=10_000.0,
            tied_embed_init_std=0.005,
            qk_gain_init=1.5,
        )

        self.assertEqual(model.blocks[0].mlp.fc.weight.shape, (32, 16))
        self.assertEqual(model.blocks[0].mlp.proj.weight.shape, (16, 32))


if __name__ == "__main__":
    unittest.main()
