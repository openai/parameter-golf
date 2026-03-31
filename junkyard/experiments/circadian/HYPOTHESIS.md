# Circadian Rhythm: Phase-Offset Layer Contribution Gates

## Biological inspiration
Synaptic efficacy cycles on a ~24h clock. Different neural pathways are dominant at
different phases. The IRRATIONAL period prevents synchronization lock-in.
This is literally why sunflowers use φ for seed packing — most efficient non-repeating
coverage, no two seeds ever perfectly aligned.

## Architecture
Each layer i gets a learned phase offset θ_i, but base spacing between layer phases
is φ (irrational — no two layers ever fully align, prevents redundant roles):

    gate_i = sigmoid(A * cos(2π * φ * i / N + θ_learned_i))

Where:
- φ = 1.618... (golden ratio — irrational, prevents lock-in)
- i = layer index (0 to N-1)
- θ_learned_i = learned per-layer phase offset (initialized to 0, trained by Adam)
- A = learned amplitude (scalar, initialized to 0.5)
- N = num_layers = 11

During training, layers phase in and out of dominance in a smooth, non-repeating wave.
The model learns which layers should dominate at which point in the sequence.

φ bonus: The mathematical reason sunflowers use φ for seed packing.
         Golden angle = 2π(1 - 1/φ) ≈ 137.5°. Maximally irrational phyllotaxis.

## Key hyperparameters
- CIRCADIAN_ENABLED = 1
- CIRCADIAN_AMPLITUDE_INIT = 0.5   (A initial value)
- CIRCADIAN_LR = 0.025             (phase/amplitude learning rate)

## Implementation
In GPT.__init__:
```python
PHI = (1 + 5**0.5) / 2
self.circadian_phases = nn.Parameter(torch.zeros(num_layers))  # θ_learned_i
self.circadian_amplitude = nn.Parameter(torch.tensor(0.5))     # A

# In forward, per-layer gate:
import math
gate_i = torch.sigmoid(
    self.circadian_amplitude * torch.cos(
        2 * math.pi * PHI * i / self.num_layers + self.circadian_phases[i]
    )
)
x = x_residual + gate_i * x_layer_output  # replaces existing residual
```

## Buildability: ★★★★☆ — ~10 lines in forward loop
Risk: gating might suppress layers entirely early in training (amplitude near 0).
Mitigation: initialize gate to 1.0 (no gating effect) by choosing A_init carefully.
