import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "records"
    / "track_non_record_16mb"
    / "2026-03-26_DiffusionNoisedTeacher_AR"
    / "train_gpt.py"
)


def load_submission_module():
    spec = importlib.util.spec_from_file_location("diffusion_submission_train_gpt", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiffusionHelperTests(unittest.TestCase):
    def test_noise_ratio_schedule_interpolates_from_min_to_max(self):
        module = load_submission_module()
        self.assertAlmostEqual(module.diffusion_noise_ratio_for_step(0, 100, 0.1, 0.5), 0.1)
        self.assertAlmostEqual(module.diffusion_noise_ratio_for_step(100, 100, 0.1, 0.5), 0.5)
        self.assertAlmostEqual(module.diffusion_noise_ratio_for_step(50, 100, 0.1, 0.5), 0.3)

    def test_corrupt_input_ids_changes_only_non_bos_tokens(self):
        module = load_submission_module()
        x = torch.tensor([[1, 11, 12, 13, 14], [1, 21, 22, 23, 24]], dtype=torch.int64)
        generator = torch.Generator().manual_seed(123)
        corrupted, noisy_mask = module.corrupt_input_ids(
            x,
            mask_token_id=2,
            vocab_size=1024,
            noise_ratio=1.0,
            random_replace_prob=0.0,
            generator=generator,
        )
        self.assertTrue(torch.equal(corrupted[:, 0], x[:, 0]))
        self.assertTrue(torch.equal(noisy_mask[:, 0], torch.zeros(2, dtype=torch.bool)))
        self.assertTrue(torch.equal(corrupted[:, 1:], torch.full_like(x[:, 1:], 2)))
        self.assertTrue(torch.equal(noisy_mask[:, 1:], torch.ones_like(noisy_mask[:, 1:], dtype=torch.bool)))


if __name__ == "__main__":
    unittest.main()
