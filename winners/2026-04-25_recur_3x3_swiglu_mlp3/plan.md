# Experiment 0057_swiglu_recur_3x3

Parent: 0056_depth_recur_3x3

## Question
Compounds: does **SwiGLU(mlp=3)** + **K=3 L=3 depth recurrence** beat 0051? 0044 showed SwiGLU(3) at 9L is +0.011 over 0051's parent but cap-violating (16.46 MB). 0056 showed K=3 L=3 recurrence is -0.013 vs 0051 with 5.4 MB artifact. The compound puts both in cap and asks: does the SwiGLU gating advantage compound with looped reuse?

## Hypothesis [CONJECTURE]
Optimistic: SwiGLU's gating gives slightly better per-block representations; in recurrence each block runs 3× per step → gating advantage compounds → predicted Δ ≥ 0 vs 0051.

Pessimistic: SwiGLU's 9L gain came partly from per-block param count; at K=3 with fewer total blocks, the gain shrinks. Plus recurrent setup may use gating less efficiently → Δ ≤ -0.020.

Predicted Δ vs 0051: -0.020 to +0.010.

## Change
Code-only, env-var gated. Modified `MLP` class to support `MLP_TYPE=swiglu`:

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self._mlp_type = os.environ.get("MLP_TYPE", "relu2")
        if self._mlp_type == "swiglu":
            self.w_gate = CastedLinear(dim, hidden, bias=False)
            self.w_up   = CastedLinear(dim, hidden, bias=False)
            self.w_down = CastedLinear(hidden, dim, bias=False)
            self.w_down._zero_init = True
        else:
            self.fc = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
            self.proj._zero_init = True

    def forward(self, x):
        if self._mlp_type == "swiglu":
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        x = torch.relu(self.fc(x))
        return self.proj(x.square())
```

Env (vs 0056): `MLP_TYPE=swiglu MLP_MULT=3` (down from 4). Depth-recurrence vars unchanged: `NUM_UNIQUE_LAYERS=3 NUM_LOOPS=3`.

## Disconfirming
- **Δ ≤ -0.020 vs 0051** → SwiGLU + recurrence worse than recurrence alone — gating doesn't help under repeated invocation.
- **Δ in [-0.013, +0.005]** → tie. SwiGLU helps recurrence by ~recurrence cost. Cap headroom remains; pivot to wider MLP / more loops.
- **Δ ≥ +0.010** → real win, promote.
- **Step-1 train_loss anomaly worse than 0044's 20.67**: numerical issue specific to SwiGLU under recurrence.

## Notes from execution
