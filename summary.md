# OpenAI Model Craft Challenge: Parameter Golf

## Overview
This repository hosts the code, data processing scripts, and official submissions for the **OpenAI Model Craft Challenge: Parameter Golf**. The objective of this competition is to train the most performant language model under extreme constraints:
- The model weights must fit within a **16MB artifact**.
- Training must complete in **under 10 minutes** on an 8xH100 GPU node.
- Models are evaluated based on their compression performance (bits per byte) on the FineWeb validation set.

Inspired by NanoGPT Speedrunning, this challenge serves as an optimization problem for neural scaling laws with constrained parameter counts. It pushes participants to explore highly efficient architectures (depth recurrence, aggressive parameter tying), compression techniques (Quantization-Aware Training, low precision), and other creative paradigms (test-time training, novel tokenizers).

## Repository Structure

- **`train_gpt.py`**: The primary PyTorch training script containing the base GPT architecture, training loop, evaluation logic, and optimizers.
- **`train_gpt_mlx.py`**: An MLX-based training script tailored for users running experiments on Apple Silicon.
- **`data/`**: Contains scripts and utilities for downloading and preprocessing the training and evaluation data. 
  - `cached_challenge_fineweb.py`: Caches and tokenizes the FineWeb dataset used for the challenge.
  - `download_hf_docs_and_tokenize.py`: Utility for downloading and tokenizing HuggingFace documentation.
- **`records/`**: Stores past and current leaderboard submissions. This directory contains detailed training logs, model binaries, and scripts for the different runs.
  - `track_10min_16mb/`: Official submissions that strictly adhere to the 10-minute compute limit and 16MB file size constraint.
  - `track_non_record_16mb/`: Experimental submissions that respect the parameter/size limits but do not strictly follow the 10-minute compute constraint, exploring the upper bounds of 16MB performance.
- **`requirements.txt`**: Python dependencies required to run the training and evaluation scripts.

## Getting Started
The main entry point for participating is exploring and modifying `train_gpt.py` to test new architectural tweaks, quantization methods, or optimization techniques. You can refer to the leaderboard and submission details in the main `README.md` to see the techniques used by top submissions and find instructions on how to submit your own records.
