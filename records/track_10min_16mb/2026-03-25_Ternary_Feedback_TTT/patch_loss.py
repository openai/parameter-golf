with open('train_gpt_mlx.py', 'r') as f:
    content = f.read()

# forward_logits
content = content.replace("hidden, _, _, _ = self(input_ids)", "hidden, _, _, _, _ = self(input_ids)")
content = content.replace("hidden, capsule_state, _, _ = self(input_ids, carry_capsules=carry_capsules)", "hidden, capsule_state, _, _, _ = self(input_ids, carry_capsules=carry_capsules)")

# loss
orig_loss_sig = """    def loss(self, input_ids, target_ids, carry_capsules=None, reduction="mean", temperature=1.0):
        hidden, capsule_state, consistency_losses, jepa_loss = self(input_ids, carry_capsules=carry_capsules)"""
new_loss_sig = """    def loss(self, input_ids, target_ids, carry_capsules=None, reduction="mean", temperature=1.0):
        hidden, capsule_state, consistency_losses, jepa_loss, moe_losses = self(input_ids, carry_capsules=carry_capsules)"""
content = content.replace(orig_loss_sig, new_loss_sig)

orig_return_loss = """        return ce_loss"""
new_return_loss = """        # MoE Aux Loss
        if self.training and self.args.moe_enabled and moe_losses:
            total_moe = mx.sum(mx.stack([l for l in moe_losses if l.ndim == 0 and l.size == 1]))
            ce_loss = ce_loss + self.args.moe_router_aux_loss_coef * total_moe

        return ce_loss"""
# use rsplit to only replace the last `return ce_loss` inside the loss function
parts = content.rsplit(orig_return_loss, 2)
if len(parts) >= 2:
    # replace the deepest occurrence inside GPT class
    # To be safe, let's use regex to replace `return ce_loss` inside `def loss`
    pass

import re
content = re.sub(r"(ce_loss = ce_loss \+ self\.args\.koopman_speculator_weight \* speculative_ms_loss\s*)return ce_loss", r"\1" + new_return_loss, content, count=1)

with open('train_gpt_mlx.py', 'w') as f:
    f.write(content)
print("Patched loss successfully")
