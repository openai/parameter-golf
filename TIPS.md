# Tips for Parameter Golf Newcomers

Here are some collected tips to help you get started with the Parameter Golf challenge:

## 1. Understand the 16MB Limit
The limit is on the **compressed** artifact. This means you can use more than 16M parameters if they compress well (e.g., through quantization or weight tying).
- Baseline uses ~22M parameters but compresses to <16MB using int8 + zlib.
- Aggressive quantization (int6, int5) allows for even more parameters (3x MLP expansion).

## 2. Tokenizer Choices
- You can bring your own tokenizer.
- Smaller vocab sizes (like the default 1024) save a lot of parameters in the embedding and head layers.
- Check `data/cached_challenge_fineweb.py` for how to use different vocab sizes.

## 3. Training Optimizations
- **Muon Optimizer:** Works very well for matrix parameters. It orthogonalizes updates, which is particularly effective in constrained parameter settings.
- **Learning Rate Warmdown:** The challenge has a strict time limit. Efficient warmdown (linear or based on remaining time) is crucial.
- **Precision:** Use `bfloat16` for training to save memory and potentially speed up training on H100s.

## 4. Architectural Ideas
- **Weight Tying:** Not just embedding/head tying, but also consider tying weights across layers (Depth Recurrence).
- **GQA (Grouped Query Attention):** Saves parameters in the KV projections.
- **Softcapping Logits:** Helps stabilize training with high learning rates.

## 5. Evaluation
- Use **Sliding Window Evaluation** for a significant boost in BPB without changing the model.
- Evaluation sequence length can be different from training sequence length.

## 6. Helpful Community Tools
- Check the issues and PRs for community-made leaderboards and monitoring tools.
- Join the OpenAI Discord #parameter-golf channels.
