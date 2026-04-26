# save as make_dummy_tokenizer.py, run: python make_dummy_tokenizer.py
from pathlib import Path
import sentencepiece as spm

out_dir = Path("data/tokenizers")
out_dir.mkdir(parents=True, exist_ok=True)

corpus = out_dir / "dummy_corpus.txt"
corpus.write_text(
    "\n".join([
        "hello world",
        "this is a tiny local dry run corpus",
        "parameter golf baseline test",
        "quick cpu fallback validation",
    ] * 500),
    encoding="utf-8",
)

model_prefix = out_dir / "fineweb_1024_bpe"
spm.SentencePieceTrainer.train(
    input=str(corpus),
    model_prefix=str(model_prefix),
    vocab_size=1024,
    model_type="bpe",
    hard_vocab_limit=False, 
    pad_id=0,                
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print(f"wrote: {model_prefix}.model")