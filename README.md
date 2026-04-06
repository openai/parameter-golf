<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in dom linear maps

# Ishika Pareek — Parameter Golf Research
Forked from openai/parameter-golf.

## My Approach
Product designer and ML researcher. 
Building from first principles with low LLM training background at this scale. Aim is to find something novel while I use the anchors provided by leaderboard and challenge curators.

## My Changes
- LeakyReLU(0.5)² on MLP activations
- TrigramHash(4096) — richer token context
- Mid-layer JEPA (in progress)

## Results
| Run | Hardware | Steps | val_bpb |
|-----|----------|-------|---------|
| A100 4500 steps | 1×A100 | 4500 | 1.2819 |
| 8×H100 5274 steps | 8×H100 | 5274 | 1.2631 |

## Research Log
See RESEARCH_LOG.md for daily progress and thinking.

## Next
Implementing mid-layer hybrid JEPA architecture. (Listed as a requested implementation by curators)
