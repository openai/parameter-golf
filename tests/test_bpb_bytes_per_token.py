import importlib.util
import os
import random
import unittest
from pathlib import Path

import sentencepiece as spm
import tiktoken
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_gpt_module():
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    spec = importlib.util.spec_from_file_location("train_gpt_for_bpb_tests", REPO_ROOT / "train_gpt.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _shape_pair(stream: list[int], width: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    x = torch.tensor(stream[:-1], dtype=torch.long)
    y = torch.tensor(stream[1:], dtype=torch.long)
    usable = (x.numel() // width) * width
    if usable <= 0:
        raise AssertionError("token stream too short for test reshape")
    return x[:usable].reshape(-1, width), y[:usable].reshape(-1, width), usable


def _text_corpus() -> list[str]:
    texts = [
        "",
        "hello",
        " hello world",
        "tabs\tand\tspaces",
        "newlines\nline2\n line3",
        "punctuation !?.,;:\"'()[]{}",
        "Accents: café naïve résumé",
        "Emoji: 🙂🚀🔥",
        "CJK: 你好 世界 日本語",
        "ZWJ emoji: 👨\u200d👩\u200d👧\u200d👦",
        "trailing space ",
    ]
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789     \t\n.,;:!?/\\-_+*=()[]{}\"'") + [
        "é",
        "ß",
        "ø",
        "ç",
        "ñ",
        "ü",
        "Ω",
        "λ",
        "中",
        "文",
        "你",
        "好",
        "🙂",
        "🚀",
        "🔥",
        "£",
        "€",
        "©",
        "®",
        "👨",
        "👩",
        "👧",
        "👦",
        "\u200d",
    ]
    rng = random.Random(1234)
    for _ in range(24):
        n = rng.randint(0, 64)
        texts.append("".join(rng.choice(pool) for _ in range(n)))
    return texts


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


class TestBitsPerByteTokenAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_gpt = _load_train_gpt_module()
        cls.texts = _text_corpus()

    def setUp(self):
        self.train_gpt._BYTES_LUT_CACHE.clear()

    def test_gpt2_bytes_per_token_exact(self):
        tg = self.train_gpt
        tg.TOKENIZER_KIND = "gpt2"

        enc = tiktoken.get_encoding("gpt2")
        eot = int(enc.eot_token)
        stream = [eot]
        for i, text in enumerate(self.texts):
            stream.extend(enc.encode_ordinary(text))
            if i % 3 == 0:
                stream.append(eot)

        x, y, usable = _shape_pair(stream, width=31)
        got = tg.bytes_per_token(x, y).reshape(-1).tolist()
        exp = []
        for tok in stream[1 : 1 + usable]:
            tok = int(tok)
            exp.append(0 if tok == eot else len(enc.decode_single_token_bytes(tok)))
        self.assertEqual(got, exp)

    def test_sentencepiece_bytes_per_token_exact(self):
        tg = self.train_gpt
        sp_path = _first_existing(
            [
                REPO_ROOT / "data" / "tokenizers" / "fineweb_1k_bpe.model",
                REPO_ROOT / "data" / "matched_10B_docs2m_seed1337" / "tokenizers" / "fineweb_1024_bpe.model",
            ]
        )
        if sp_path is None:
            raise unittest.SkipTest("No local SentencePiece model available for BPB tests")

        sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        bos_id = int(sp.bos_id())
        eos_id = int(sp.eos_id())

        tg.TOKENIZER_KIND = "sentencepiece"
        tg.FIXED_TOKENIZER_PATH = str(sp_path)

        # Per-doc exact byte totals: sum(bytes_per_token) must equal decoded UTF-8 bytes.
        for idx, text in enumerate(self.texts):
            doc = [bos_id]
            doc.extend(map(int, sp.encode(text, out_type=int)))
            if idx % 2 == 0 and eos_id >= 0:
                doc.append(eos_id)
            if len(doc) < 2:
                continue
            x_doc = torch.tensor(doc[:-1], dtype=torch.long).reshape(1, -1)
            y_doc = torch.tensor(doc[1:], dtype=torch.long).reshape(1, -1)
            got_sum = int(tg.bytes_per_token(x_doc, y_doc).sum().item())
            exp_sum = len(sp.decode(doc).encode("utf-8"))
            self.assertEqual(got_sum, exp_sum, msg=f"doc={idx} text={text!r}")

        # Streamed per-token checks across BOS/EOS boundaries and row reshapes.
        stream = [bos_id]
        for i, text in enumerate(self.texts):
            stream.extend(map(int, sp.encode(text, out_type=int)))
            if i % 4 == 0 and eos_id >= 0:
                stream.append(eos_id)
            stream.append(bos_id)
        if eos_id >= 0:
            stream.extend([eos_id, bos_id])

        x, y, usable = _shape_pair(stream, width=29)
        got = tg.bytes_per_token(x, y).reshape(-1).tolist()

        # For byte-fallback tokens, each token is exactly one raw byte; prefix decode deltas
        # are not reliable on partial UTF-8 sequences. For all other tokens, prefix deltas are exact.
        prefix_lens = [0] * (usable + 2)
        for i in range(usable + 2):
            prefix_lens[i] = len(sp.decode(stream[:i]).encode("utf-8"))

        byte_token_hits = 0
        for i, g in enumerate(got, start=1):
            tok = int(stream[i])
            if sp.is_byte(tok):
                byte_token_hits += 1
                exp_i = 1
            else:
                exp_i = prefix_lens[i + 1] - prefix_lens[i]
            self.assertEqual(
                g,
                exp_i,
                msg=f"pos={i-1} prev={stream[i-1]} tok={tok} prev_piece={sp.id_to_piece(int(stream[i-1]))!r} tok_piece={sp.id_to_piece(tok)!r}",
            )
        self.assertGreater(byte_token_hits, 0, "Expected to cover at least one byte-fallback token")

    def test_sentencepiece_cache_key_tracks_model_path(self):
        tg = self.train_gpt
        model_a = _first_existing(
            [
                REPO_ROOT / "data" / "tokenizers" / "fineweb_1k_bpe.model",
                REPO_ROOT / "data" / "matched_10B_docs2m_seed1337" / "tokenizers" / "fineweb_1024_bpe.model",
            ]
        )
        model_b = _first_existing(
            [
                REPO_ROOT / "data" / "tokenizers" / "fineweb_4k_bpe.model",
                REPO_ROOT / "data" / "matched_10B_docs2m_seed1337" / "tokenizers" / "fineweb_4096_bpe.model",
            ]
        )
        if model_a is None or model_b is None:
            raise unittest.SkipTest("Need two SentencePiece models for cache-key regression test")

        text = "Hello world 🙂 multiple words and unicode 你好 with spaces and punctuation!!!"
        sp_a = spm.SentencePieceProcessor(model_file=str(model_a))
        sp_b = spm.SentencePieceProcessor(model_file=str(model_b))

        tg.TOKENIZER_KIND = "sentencepiece"
        tg.FIXED_TOKENIZER_PATH = str(model_a)
        seq_a = [int(sp_a.bos_id())] + list(map(int, sp_a.encode(text, out_type=int)))
        x_a = torch.tensor(seq_a[:-1], dtype=torch.long).reshape(1, -1)
        y_a = torch.tensor(seq_a[1:], dtype=torch.long).reshape(1, -1)
        _ = tg.bytes_per_token(x_a, y_a)  # prime cache with model A

        tg.FIXED_TOKENIZER_PATH = str(model_b)
        seq_b = [int(sp_b.bos_id())] + list(map(int, sp_b.encode(text, out_type=int)))
        x_b = torch.tensor(seq_b[:-1], dtype=torch.long).reshape(1, -1)
        y_b = torch.tensor(seq_b[1:], dtype=torch.long).reshape(1, -1)

        got_without_clear = tg.bytes_per_token(x_b, y_b).clone()
        tg._BYTES_LUT_CACHE.clear()
        got_with_clear = tg.bytes_per_token(x_b, y_b).clone()
        self.assertTrue(torch.equal(got_without_clear, got_with_clear))


if __name__ == "__main__":
    unittest.main()
