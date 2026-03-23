# Non-Record Submission: PipeOwl-1.8-jp-parameter-golf

This is a non-record submission.

PipeOwl explores an alternative to transformer-based language models.

- O(n) over vocabulary
- No attention
- No transformer weights
- CPU-friendly (<16MB model)

---

## Idea

Instead of relying on pairwise token interactions (O(n²)),
PipeOwl uses a field-based representation with linear-time scoring.

This work focuses on:
- embedding compression
- alternative token interaction
- simple and deterministic inference

---

## How to run

1. Load model:


git clone https://huggingface.co/WangKaiLin/PipeOwl-1.8-jp-parameter-golf

cd PipeOwl-1.8-jp-parameter-golf

pip install numpy safetensors

python quickstart.py


2. Inference (example):


Please enter words： 東京

Top-K Tokens:
1.000 | 東京
0.739 | 東京都
0.679 | 大阪
0.666 | ロンドン
0.646 | 名古屋

Please enter words： 大阪

Top-K Tokens:
1.000 | 大阪
0.756 | 関西
0.728 | 難波
0.717 | 京都
0.712 | 守口


---

## Architecture

- Static embedding table (V × D)
- Aligned vocabulary index
- Δfield (scalar bias field)
- Linear scoring function
- Pluggable decoding stage

Designed for:
- CPU environments
- low-latency systems (e.g. IME)
- deterministic behavior

---

## Configuration


VOCAB_SIZE: 26155
EMBEDDING_DIM: 256
DTYPE: FP16
FORMAT: safetensors
LANGUAGE: Japanese

Startup time: <1s
Query latency: 1.0 ~ 1.8 ms (CPU, full vocabulary scan)

Artifact size: 13,889,536 bytes


---

## Notes

- This is not a transformer model
- No attention mechanism is used
- Uses field-based representation (base + Δfield)
- Not optimized for benchmark score

---

## Limitations

- Training pipeline is minimal and not included
- Focus is on inference structure and model compression

---

## Included Files

- `submission.json` — leaderboard metadata