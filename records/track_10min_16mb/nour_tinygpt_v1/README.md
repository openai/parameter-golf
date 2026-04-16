# Nour TinyGPT v1 - Parameter Golf Submission

## Overview
This project implements a lightweight GPT-style language model trained on the FineWeb sp1024 dataset.  
The goal is to optimize for **speed and efficiency**, as required by the Parameter Golf challenge.

## Features
- Small transformer model (3 layers, 128 embedding size)
- Reduced context length (128)
- Cosine learning rate schedule with warmup
- Gradient clipping for stability
- Optimized for fast training on GPU

## Setup

Install dependencies:

```bash
pip install -r requirements.txt