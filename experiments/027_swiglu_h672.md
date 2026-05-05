# Experiment 027: SwiGLU Activation (hidden=672) — THE BIG BET

## Status: RUNNING (wandb: parameter-golf / exp027_swiglu_h672)

## Hypothesis
Replace relu² MLP with SwiGLU(hidden=672). Same FLOPs, -1.5% params, fits 16MB.
Expected 0.006-0.018 BPB improvement based on LLaMA/Mistral/Gemma literature.
Zero speed penalty. This is the "free lunch" we should have tried first.

## Configuration
- **Architecture**: 9L d512, SwiGLU MLP (up/gate/down at hidden=672)
- **Params**: ~16.88M (-1.5% vs baseline 17.06M)
- **Artifact**: ~15.7MB (325KB headroom)
- **Speed**: Should match baseline (~210ms on 1GPU, ~43ms on 8xH100)
- **Other**: softcap=15, eps=1e-10

## Key Code Change
```python
class SwiGLUMLP(nn.Module):
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```
