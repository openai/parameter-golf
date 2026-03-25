# Cubric Visualization — Model Internals as Art

## Concept

Turn the computational process of language modeling into visual/sonic art
by capturing the per-token tension between neural prediction and statistical
memory. Every token scored is a data point with rich metadata that maps
naturally to artistic dimensions.

## Data Streams Available

### Training Phase (6,800 steps × 786K tokens/step)
- Per-step loss trajectory (the learning curve as a waveform)
- Per-layer gradient magnitudes (11 voices, each learning at different rates)
- Attention pattern snapshots (88 heads × sequence length — what the model "looks at")
- U-Net skip connection flow (information passing from encoder to decoder)
- Weight distribution evolution (26.9M parameters shifting over time)
- Muon optimizer momentum (the "inertia" of learning)

### N-gram Eval Phase (121,000 windows)
- Per-token: model probability vs n-gram probability vs mixed
- Per-token: did the counter help or hurt? (the tension moment)
- Cache fill rate over time (hash table growing from empty to rich)
- Heatmap: which regions of text the n-gram dominates vs the model
- Confidence spectrum: certain tokens (model wins) vs uncertain (counter wins)
- The 80/20 boundary — visualize what changes at different alpha values

### Artistic Mappings

| Data | Visual | Sonic |
|------|--------|-------|
| Model confidence | Brightness/opacity | Volume |
| N-gram agreement | Color hue (blue=agree, red=disagree) | Harmony/dissonance |
| Cache growth | Particle density | Reverb depth |
| Loss landscape | Terrain height | Pitch |
| Attention heads | Connecting lines/arcs | Polyphonic voices |
| The mix moment | Blend/interference pattern | Two signals merging |

### The Core Image

A document rendered as a stream of tokens. Each token colored by who predicted
it better — the neural network (trained on millions of texts) or a simple counter
(watching this specific document unfold). The moments where the counter wins are
moments where local structure beats general knowledge. Those moments cluster
around repeated phrases, structured data, boilerplate — the "texture" of text.

## Implementation Notes

- Add `--dump-token-log` flag to eval that writes per-token JSON
- Fields: position, token_id, token_text, model_p, ngram_p, mixed_p, ngram_helped, cache_size
- Separate visualization tool reads the log and renders
- Could be real-time with websocket streaming during eval

## Status
IDEA — explore when time allows. No code yet.
