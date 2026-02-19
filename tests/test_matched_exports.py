import glob
import hashlib
import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Callable

import numpy as np
import sentencepiece as spm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from pure_byte_tokenizer import PureByteTokenizer, default_pure_byte_tokenizer


def _default_manifest_path() -> Path:
    manifest_env = os.environ.get("MATCHED_EXPORT_MANIFEST")
    if manifest_env:
        return Path(manifest_env)
    root_env = os.environ.get("MATCHED_EXPORT_ROOT")
    if root_env:
        return Path(root_env) / "manifest.json"
    return REPO_ROOT / "data" / "matched_10B_docs2m_seed1337" / "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_data_shard(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=256)
        if header.size != 256:
            raise AssertionError(f"{path}: truncated header")
        if int(header[0]) != 20240520:
            raise AssertionError(f"{path}: magic mismatch")
        if int(header[1]) != 1:
            raise AssertionError(f"{path}: version mismatch ({int(header[1])})")
        num_tokens = int(header[2])
        toks = np.fromfile(f, dtype=np.uint16, count=num_tokens)
        if toks.size != num_tokens:
            raise AssertionError(f"{path}: token count mismatch ({toks.size} != {num_tokens})")
        return toks


class TestPureByteTokenizer(unittest.TestCase):
    def test_roundtrip_utf8(self):
        tok = default_pure_byte_tokenizer()
        text = "hello Ω🙂"
        ids = [tok.bos_id] + tok.encode(text)
        self.assertTrue(ids and ids[0] == tok.bos_id)
        self.assertTrue(all(0 <= i < tok.vocab_size for i in ids))
        self.assertEqual(tok.decode(ids), text)

    def test_json_artifact_load(self):
        tmp = Path("/tmp/pure_byte_test_tokenizer.json")
        tok = default_pure_byte_tokenizer()
        tok.save_json(tmp)
        loaded = PureByteTokenizer.from_json(tmp)
        self.assertEqual(tok.vocab_size, loaded.vocab_size)
        self.assertEqual(tok.bos_id, loaded.bos_id)
        self.assertEqual(tok.encode("abc"), loaded.encode("abc"))


class TestMatchedExportManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = _default_manifest_path()
        if not cls.manifest_path.is_absolute():
            cls.manifest_path = (REPO_ROOT / cls.manifest_path).resolve()
        if not cls.manifest_path.is_file():
            raise unittest.SkipTest(
                f"Matched export manifest not found yet: {cls.manifest_path}. "
                "Run export first or set MATCHED_EXPORT_MANIFEST."
            )
        with cls.manifest_path.open("r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        cls.docs_path = cls._resolve_path(cls.manifest["docs_jsonl"])
        if not cls.docs_path.is_file():
            raise AssertionError(f"Missing docs cache referenced in manifest: {cls.docs_path}")

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (REPO_ROOT / p).resolve()

    @staticmethod
    def _resolve_glob(pattern: str) -> str:
        p = Path(pattern)
        if p.is_absolute():
            return pattern
        return str(REPO_ROOT / pattern)

    def _iter_docs(self, limit: int):
        with self.docs_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    return
                row = json.loads(line)
                yield row["text"]

    def _load_tokenizer_for_dataset(self, dataset: dict) -> tuple[Callable[[str], list[int]], int, int, int]:
        name = dataset["name"]
        tokenizers = self.manifest["tokenizers"]
        if "byte" in name:
            entry = next(t for t in tokenizers if t.get("kind") == "byte")
            tok = PureByteTokenizer.from_json(self._resolve_path(entry["path"]))
            return tok.encode, int(tok.bos_id), int(tok.eos_id), int(tok.vocab_size)

        m = re.search(r"_sp(\d+)$", name)
        if not m:
            raise AssertionError(f"Cannot infer tokenizer for dataset {name}")
        requested = int(m.group(1))
        entry_name = f"sp_bpe_{requested}"
        entry = next(t for t in tokenizers if t.get("name") == entry_name)
        model_path = self._resolve_path(entry["model_path"])
        sp_tok = spm.SentencePieceProcessor(model_file=str(model_path))
        return (
            lambda text, tok=sp_tok: tok.encode(text, out_type=int),
            int(sp_tok.bos_id()),
            int(sp_tok.eos_id()),
            int(sp_tok.vocab_size()),
        )

    def _read_dataset_prefix_tokens(self, dataset: dict, n_tokens: int) -> np.ndarray:
        out = np.empty(n_tokens, dtype=np.uint16)
        filled = 0
        for pattern in (dataset["val_glob"], dataset["train_glob"]):
            for shard in sorted(glob.glob(self._resolve_glob(pattern))):
                toks = _load_data_shard(Path(shard))
                take = min(n_tokens - filled, toks.size)
                if take > 0:
                    out[filled : filled + take] = toks[:take]
                    filled += take
                if filled >= n_tokens:
                    return out
        raise AssertionError(
            f"Not enough tokens in dataset {dataset['name']}: needed {n_tokens}, got {filled}"
        )

    def test_docs_cache_hash_matches_manifest(self):
        docs_meta = self.manifest.get("docs_meta", {})
        expected = docs_meta.get("docs_sha256")
        self.assertIsNotNone(expected, "Manifest missing docs_meta.docs_sha256")
        self.assertEqual(_sha256(self.docs_path), expected)

    def test_dataset_doc_stats_are_matched(self):
        num_docs = int(self.manifest["num_docs"])
        num_val_docs = int(self.manifest["num_val_docs"])
        num_train_docs = num_docs - num_val_docs
        self.assertGreater(num_train_docs, 0)

        for ds in self.manifest["datasets"]:
            stats = ds["stats"]
            self.assertEqual(stats["docs_total"], num_docs, ds["name"])
            self.assertEqual(stats["docs_val"], num_val_docs, ds["name"])
            self.assertEqual(stats["docs_train"], num_train_docs, ds["name"])

    def test_tokenizer_prefix_alignment(self):
        docs_to_check = int(os.environ.get("MATCHED_PREFIX_DOCS", "64"))
        docs_to_check = max(1, min(docs_to_check, int(self.manifest["num_docs"])))
        append_eos = bool(self.manifest.get("append_eos", False))
        docs = list(self._iter_docs(docs_to_check))
        self.assertEqual(len(docs), docs_to_check)

        for ds in self.manifest["datasets"]:
            encode_fn, bos_id, eos_id, vocab_size = self._load_tokenizer_for_dataset(ds)
            expected: list[int] = []
            for text in docs:
                expected.append(bos_id)
                expected.extend(encode_fn(text))
                if append_eos:
                    expected.append(eos_id)
            expected_arr = np.asarray(expected, dtype=np.uint16)
            actual_arr = self._read_dataset_prefix_tokens(ds, len(expected_arr))
            np.testing.assert_array_equal(
                actual_arr,
                expected_arr,
                err_msg=f"Token prefix mismatch for dataset {ds['name']}",
            )
            self.assertLess(int(actual_arr.max()), int(vocab_size), ds["name"])


if __name__ == "__main__":
    unittest.main()

