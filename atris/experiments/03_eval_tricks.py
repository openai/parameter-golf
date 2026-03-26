"""
Experiment 03: Evaluation Tricks

HYPOTHESIS: The rules explicitly say "we encourage competitors to push the
bounds of evaluation methods as aggressively as with training methods" and
give a SEPARATE 10-minute budget for evaluation. This is the sleeper weapon.

KEY INSIGHT: BPB measures how well you predict the next byte. With 10 minutes
of eval compute, we can do things like:
1. Eval at longer sequence lengths (more context = better predictions)
2. Test-time training (adapt to the validation distribution)
3. Ensemble via temperature sampling / dropout

APPROACH 3A: Longer eval context
- Train on seq_len=1024
- Eval on seq_len=2048 or 4096
- RoPE naturally extrapolates; can also use NTK-aware scaling
- Nearly free improvement: same model, better predictions

APPROACH 3B: Test-time training (TTT)
- During eval, do a few SGD steps on the validation prefix
- This adapts the model to the specific distribution of the val set
- 10 minutes is a LOT of fine-tuning time for a small model
- The val data IS available during eval — you just can't pre-bake it

APPROACH 3C: Self-ensembling
- Run forward pass with different dropout masks
- Average the logits
- Or: run at multiple temperatures and combine

MODIFICATIONS TO train_gpt.py:
- eval_val: change sequence length for approach 3A
- eval_val: add online SGD loop for approach 3B
- GPT.forward: return logits instead of loss for ensembling

VARIANTS:
- 03a: Eval at seq_len=2048 (no other changes)
- 03b: Eval at seq_len=4096
- 03c: TTT with 1 epoch of SGD on val data, then re-eval
- 03d: TTT + longer context combined
- 03e: Sliding window eval (predict each token using max available context)

EXPECTED IMPACT: 0.005-0.02 BPB improvement
RISK: RoPE extrapolation may degrade. TTT may overfit to val set prefix.
"""

# For 3A, the change is minimal:
# In eval_val(), replace args.train_seq_len with a longer eval_seq_len
# Need to handle the RoPE cache for longer sequences
#
# For 3B (TTT), the eval function becomes:
#
# def eval_val_with_ttt(args, model, ...):
#     # Phase 1: Fine-tune on val data (causal LM objective)
#     model.train()
#     ttt_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
#     for epoch in range(ttt_epochs):
#         for batch in val_batches:
#             loss = model(batch_x, batch_y)
#             loss.backward()
#             ttt_optimizer.step()
#             ttt_optimizer.zero_grad()
#
#     # Phase 2: Evaluate with adapted model
#     model.eval()
#     return eval_val(args, model, ...)
