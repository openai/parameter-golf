# Research Log — IshiPareek

## Day 0 
- Understood the Baseline architecture
- Played with Hyperparameters on my local machine
- Implemented layer averaging
- Realised it is faster to work on top of the leaderboards architecture 

## Day 1 — March 25
- Understood PR #414 architecture
- Added LeakyReLU(0.5)^2 — keeps neurons alive during training
- Added TrigramHash(8192) — richer token context before attention
- Score: 1.3762 bpb (1xA100, 7000 steps)
- Submitted non-record PR

## Day 2 — March 26
Ideas explored:
1. Anchor MLP — initial self-reflection layer before any attention (The most valuable to be tested) 
2. Alternating attention-heavy / MLP-heavy layer pairs
3. Scheduled sampling — close the training/inference gap (teacher forcing problem)


Key insights: 
Where is the model not communicating when it should be? That gap is always an opportunity.
On a larger scale, when is complicated too complicated? 
Which idea should be picked? How does one work within these resource constraints? 

Next: implement anchor MLP, smoke test locally and if it beats my score of 2.3, we run it on the GPU. 
