from __future__ import annotations

from pathlib import Path


def load_tokenmonster_vocab(vocab_ref: str):
    try:
        import tokenmonster
    except ImportError as exc:
        raise RuntimeError("tokenmonster is required") from exc

    path = Path(vocab_ref).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"} and path.is_file():
        return tokenmonster.new(path.read_bytes())
    return tokenmonster.load(vocab_ref)


def tokenmonster_charset_name(vocab) -> str:
    try:
        return str(vocab.charset())
    except Exception:
        return "utf-8"


def tokenmonster_byte_encoding(vocab) -> str:
    return "latin-1" if tokenmonster_charset_name(vocab) == "None" else "utf-8"


def tokenmonster_decoded_text_to_bytes(text: str, vocab) -> bytes:
    return text.encode(tokenmonster_byte_encoding(vocab))
