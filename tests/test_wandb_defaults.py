from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "records"
    / "track_10min_16mb"
    / "2026-03-19_TrainingOptSeq4096"
    / "train_gpt.py"
)


class WandbDefaultsTest(unittest.TestCase):
    def test_training_script_enables_wandb_by_default_and_reads_local_keys(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('wandb_enabled = os.environ.get("WANDB", "1").lower() not in', source)
        self.assertIn('wandb_project = os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf")', source)
        self.assertIn('wandb_entity = os.environ.get("WANDB_ENTITY", "amelie-iska-math")', source)
        self.assertIn("find_wandb_api_key", source)
        self.assertIn('Path("keys.txt")', source)
        self.assertIn("wandb.init", source)
        self.assertIn("wandb.define_metric", source)
        self.assertIn("wandb_run.log", source)
        self.assertIn('"val/bpb": val_bpb', source)
        self.assertIn('"val_bpb": val_bpb', source)
        self.assertIn('"bpb": val_bpb', source)
        self.assertIn('"openai_parameter_golf/bpb": val_bpb', source)
        self.assertIn('"fineweb/val_bpb": val_bpb', source)
        self.assertIn('"fineweb/best_val_bpb": best_val_bpb', source)
        self.assertIn('"bpb/gap_to_target": target_gap', source)
        self.assertIn('"progress/fraction": step / max(args.iterations, 1)', source)
        self.assertIn('"perf/train_tokens_per_second": tokens_per_second', source)
        self.assertIn('"system/vram_allocated_gb": torch.cuda.memory_allocated(device) / 1e9', source)
        self.assertIn('"checkpoint/bytes": checkpoint_bytes', source)


if __name__ == "__main__":
    unittest.main()
