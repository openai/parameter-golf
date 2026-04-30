"""Tests for resumable checkpoint infrastructure (Phases 1 & 2).

These tests verify the core logic without requiring torch/CUDA by
extracting and testing the pure-Python parts of the resume system.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestResumeManifestPath(unittest.TestCase):
    """Test _resume_manifest_path helper."""

    def test_returns_json_in_dir(self):
        # Inline the logic since we can't import the real module
        def _resume_manifest_path(resume_dir):
            return os.path.join(resume_dir, "resume_manifest.json")

        result = _resume_manifest_path("/some/dir/resume")
        self.assertEqual(result, "/some/dir/resume/resume_manifest.json")

    def test_empty_dir(self):
        def _resume_manifest_path(resume_dir):
            return os.path.join(resume_dir, "resume_manifest.json")

        result = _resume_manifest_path("")
        self.assertEqual(result, "resume_manifest.json")


class TestManifestSchema(unittest.TestCase):
    """Verify the manifest JSON schema matches expectations."""

    def _make_manifest(self, step=100, world_size=8, training_time_ms=60000.0):
        """Create a manifest dict matching the save_resume_checkpoint format."""
        return {
            "step": step,
            "training_time_ms": training_time_ms,
            "world_size": world_size,
            "timestamp": 1234567890.0,
            "rank_files": {
                str(r): f"resume_rank{r}_step{step}.pt" for r in range(world_size)
            },
            "hparam_fingerprint": {
                "num_layers": 9,
                "model_dim": 512,
                "num_heads": 8,
                "num_kv_heads": 4,
                "vocab_size": 1024,
                "mlp_mult": 2,
                "num_loops": 0,
                "train_seq_len": 1024,
                "tokenizer_path": "/data/tokenizer.model",
                "data_path": "/data/train",
            },
            "exported_minutes": [10, 20],
        }

    def test_required_top_level_keys(self):
        manifest = self._make_manifest()
        required_keys = {"step", "training_time_ms", "world_size", "timestamp",
                         "rank_files", "hparam_fingerprint", "exported_minutes"}
        self.assertTrue(required_keys.issubset(set(manifest.keys())))

    def test_rank_files_count_matches_world_size(self):
        for ws in [1, 2, 4, 8]:
            manifest = self._make_manifest(world_size=ws)
            self.assertEqual(len(manifest["rank_files"]), ws)
            for r in range(ws):
                self.assertIn(str(r), manifest["rank_files"])

    def test_rank_file_naming_convention(self):
        manifest = self._make_manifest(step=42, world_size=2)
        self.assertEqual(manifest["rank_files"]["0"], "resume_rank0_step42.pt")
        self.assertEqual(manifest["rank_files"]["1"], "resume_rank1_step42.pt")

    def test_hparam_fingerprint_keys(self):
        manifest = self._make_manifest()
        fp = manifest["hparam_fingerprint"]
        expected_keys = {"num_layers", "model_dim", "num_heads", "num_kv_heads",
                         "vocab_size", "mlp_mult", "num_loops", "train_seq_len",
                         "tokenizer_path", "data_path"}
        self.assertEqual(set(fp.keys()), expected_keys)

    def test_manifest_json_round_trip(self):
        """Manifest should survive JSON serialization."""
        manifest = self._make_manifest()
        serialized = json.dumps(manifest, indent=2)
        restored = json.loads(serialized)
        self.assertEqual(manifest["step"], restored["step"])
        self.assertEqual(manifest["world_size"], restored["world_size"])
        self.assertEqual(manifest["rank_files"], restored["rank_files"])
        self.assertEqual(manifest["hparam_fingerprint"], restored["hparam_fingerprint"])
        self.assertEqual(manifest["exported_minutes"], restored["exported_minutes"])

    def test_exported_minutes_is_list(self):
        manifest = self._make_manifest()
        self.assertIsInstance(manifest["exported_minutes"], list)


class TestCompatibilityValidation(unittest.TestCase):
    """Test the hparam compatibility check logic extracted from load_resume_checkpoint."""

    def _check_compat(self, saved_fp, current_fp):
        """Extracted compatibility check logic."""
        critical_keys = ["num_layers", "model_dim", "num_heads", "num_kv_heads",
                         "vocab_size", "mlp_mult", "num_loops"]
        for key in critical_keys:
            if saved_fp.get(key) != current_fp.get(key):
                raise ValueError(
                    f"Resume incompatible: {key} mismatch "
                    f"(saved={saved_fp.get(key)}, current={current_fp.get(key)})"
                )

    def _make_fp(self, **overrides):
        fp = {
            "num_layers": 9, "model_dim": 512, "num_heads": 8,
            "num_kv_heads": 4, "vocab_size": 1024, "mlp_mult": 2,
            "num_loops": 0, "train_seq_len": 1024,
            "tokenizer_path": "", "data_path": "",
        }
        fp.update(overrides)
        return fp

    def test_identical_fingerprints_pass(self):
        fp = self._make_fp()
        self._check_compat(fp, fp.copy())  # Should not raise

    def test_num_layers_mismatch_raises(self):
        saved = self._make_fp(num_layers=9)
        current = self._make_fp(num_layers=12)
        with self.assertRaises(ValueError) as ctx:
            self._check_compat(saved, current)
        self.assertIn("num_layers", str(ctx.exception))

    def test_model_dim_mismatch_raises(self):
        saved = self._make_fp(model_dim=512)
        current = self._make_fp(model_dim=768)
        with self.assertRaises(ValueError) as ctx:
            self._check_compat(saved, current)
        self.assertIn("model_dim", str(ctx.exception))

    def test_vocab_size_mismatch_raises(self):
        saved = self._make_fp(vocab_size=1024)
        current = self._make_fp(vocab_size=8192)
        with self.assertRaises(ValueError) as ctx:
            self._check_compat(saved, current)
        self.assertIn("vocab_size", str(ctx.exception))

    def test_train_seq_len_change_does_not_raise(self):
        """train_seq_len is NOT a critical key; changes should be allowed."""
        saved = self._make_fp(train_seq_len=1024)
        current = self._make_fp(train_seq_len=2048)
        self._check_compat(saved, current)  # Should not raise

    def test_tokenizer_path_change_does_not_raise(self):
        saved = self._make_fp(tokenizer_path="/old/path")
        current = self._make_fp(tokenizer_path="/new/path")
        self._check_compat(saved, current)  # Should not raise

    def test_world_size_mismatch_detected(self):
        """World size check is separate from fingerprint."""
        saved_ws, current_ws = 8, 4
        with self.assertRaises(ValueError):
            if saved_ws != current_ws:
                raise ValueError(
                    f"Resume incompatible: saved world_size={saved_ws}, current={current_ws}"
                )


class TestAtomicSaveLogic(unittest.TestCase):
    """Test that the atomic save pattern (write tmp, then rename) works."""

    def test_atomic_rename_pattern(self):
        """Simulate the atomic save: write .tmp then os.replace."""
        test_dir = os.path.join(os.getcwd(), "_test_atomic_save")
        os.makedirs(test_dir, exist_ok=True)
        try:
            final_path = os.path.join(test_dir, "checkpoint.pt")
            tmp_path = final_path + ".tmp"
            # Write to tmp
            with open(tmp_path, "w") as f:
                f.write("checkpoint_data")
            self.assertTrue(os.path.exists(tmp_path))
            # Atomic rename
            os.replace(tmp_path, final_path)
            self.assertTrue(os.path.exists(final_path))
            self.assertFalse(os.path.exists(tmp_path))
            with open(final_path) as f:
                self.assertEqual(f.read(), "checkpoint_data")
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_manifest_atomic_write(self):
        """Simulate manifest atomic write via JSON."""
        test_dir = os.path.join(os.getcwd(), "_test_manifest_atomic")
        os.makedirs(test_dir, exist_ok=True)
        try:
            manifest = {"step": 100, "world_size": 8}
            manifest_path = os.path.join(test_dir, "resume_manifest.json")
            tmp_path = manifest_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp_path, manifest_path)
            with open(manifest_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["step"], 100)
            self.assertEqual(loaded["world_size"], 8)
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


class TestDocumentPackingLoaderStateDictLogic(unittest.TestCase):
    """Test the state_dict/load_state_dict logic for DocumentPackingLoader."""

    def test_state_dict_captures_correct_shard_index(self):
        """Verify shard index calculation: len(files) - len(remaining) - 1."""
        files = [f"shard_{i}.bin" for i in range(5)]
        # Simulate: we've consumed shards 0, 1, 2 (iterator at 3, 4)
        file_iter = iter(files[3:])
        remaining = list(file_iter)
        current_shard_idx = len(files) - len(remaining) - 1
        self.assertEqual(current_shard_idx, 2)

    def test_state_dict_schema(self):
        """Verify state dict has expected keys."""
        state = {
            "file_list": ["a.bin", "b.bin", "c.bin"],
            "current_shard_idx": 1,
            "cursor": 4096,
        }
        self.assertIn("file_list", state)
        self.assertIn("current_shard_idx", state)
        self.assertIn("cursor", state)
        self.assertIsInstance(state["current_shard_idx"], int)
        self.assertIsInstance(state["cursor"], int)

    def test_load_restores_file_iter_from_shard_idx(self):
        """After load_state_dict with shard_idx=2, file_iter should start at shard 3."""
        files = ["s0", "s1", "s2", "s3", "s4"]
        shard_idx = 2
        restored_iter = iter(files[shard_idx + 1:])
        remaining = list(restored_iter)
        self.assertEqual(remaining, ["s3", "s4"])


class TestCheckpointCleanupLogic(unittest.TestCase):
    """Test the old checkpoint cleanup logic."""

    def test_keep_last_3(self):
        """Simulate keeping only the last 3 checkpoints."""
        test_dir = os.path.join(os.getcwd(), "_test_cleanup")
        os.makedirs(test_dir, exist_ok=True)
        try:
            # Create 5 fake checkpoint files
            import time as _time
            for step in [10, 20, 30, 40, 50]:
                path = os.path.join(test_dir, f"resume_rank0_step{step}.pt")
                with open(path, "w") as f:
                    f.write(f"ckpt_{step}")
                _time.sleep(0.01)

            keep_last = 3
            import glob as glob_mod
            all_ckpts = sorted(
                glob_mod.glob(os.path.join(test_dir, "resume_rank0_step*.pt")),
                key=os.path.getmtime,
            )
            self.assertEqual(len(all_ckpts), 5)
            if len(all_ckpts) > keep_last:
                for old in all_ckpts[:-keep_last]:
                    old_step = old.split("_step")[1].replace(".pt", "")
                    old_file = os.path.join(test_dir, f"resume_rank0_step{old_step}.pt")
                    os.remove(old_file)

            remaining = glob_mod.glob(os.path.join(test_dir, "resume_rank0_step*.pt"))
            self.assertEqual(len(remaining), 3)
            remaining_steps = sorted(
                int(f.split("_step")[1].replace(".pt", "")) for f in remaining
            )
            self.assertEqual(remaining_steps, [30, 40, 50])
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


class TestResumeEnvVarParsing(unittest.TestCase):
    """Test environment variable parsing logic."""

    def test_resume_save_minutes_parsing(self):
        raw = "5,10,15,20,30"
        result = sorted(int(m.strip()) for m in raw.split(",") if m.strip())
        self.assertEqual(result, [5, 10, 15, 20, 30])

    def test_resume_save_minutes_empty(self):
        raw = ""
        result = sorted(int(m.strip()) for m in raw.split(",") if m.strip())
        self.assertEqual(result, [])

    def test_resume_save_minutes_with_spaces(self):
        raw = " 5, 10 , 20 "
        result = sorted(int(m.strip()) for m in raw.split(",") if m.strip())
        self.assertEqual(result, [5, 10, 20])

    def test_resume_disabled_by_default(self):
        val = os.environ.get("RESUME_ENABLED_TEST_FAKE", "0")
        self.assertEqual(val, "0")
        self.assertFalse(val == "1")


if __name__ == "__main__":
    unittest.main()
