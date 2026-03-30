# Research Log — IshiPareek

## Day 0 
- Understood the Baseline architecture
- Played with Hyperparameters on my local machine
- Implemented layer averaging
- Realised it is faster to work on top of the leaderboards architecture 

## Day 1
- Understood PR #414 architecture
- Added LeakyReLU(0.5)^2 — keeps neurons alive during training
- Added TrigramHash(8192) — richer token context before attention
- Score: 1.3762 bpb (1xA100, 7000 steps)
- Submitted non-record PR

## Day 2 
Ideas explored:
1. Anchor MLP — initial self-reflection layer before any attention (The most valuable to be tested) 
2. Alternating attention-heavy / MLP-heavy layer pairs
3. Scheduled sampling — close the training/inference gap (teacher forcing problem)

## Day 3 
Focus : Scheduled Sampling 
- How did this idea come to be? It is to essentially match inference with learning. Also, add reflection, one of the proven ways to enhance intelligence. The idea came to be when I wanted to add a step that helps the model recover from it's answer. The focus started with working on output softmax but shifted to something like the following. 
- How does this fit? While the traditional scheduled sampling works as a downward curve, I tried a U-Shap approach wherein we try to balance treacher forcing and sampling together. We start with a 100% teacher forcing, go down to a 50% teacher forcing midpoint and then work upwards to 100% till we reach the final step. 
- How does this fit with LeakyRelu and Trigram Hash? I was worried the mid step with high % of prediction tokens will start a chain of activation of wrong tokens but I realised that before we have reached the next step, the error would be corrected. 
<img width="692" height="190" alt="Screenshot 2026-03-27 at 2 24 04 PM" src="https://github.com/user-attachments/assets/6ed8cb8e-d22a-4ce1-be2a-35030ccb7d2f" />

- Result of running a smoke test on my machine with a batch of 500 : 
   Result of running something new like this on my machine: Steps: 500
  step 1: 6.94 ← random guessing
  step 2: 18.83 ← relatively higher spike! 
  step 3: 17.11 ↓ recovering
  .
  .
  .
  step 200: 4.77 ↓ improving
 
 val_bpb: 2.367 (worse than my initial run), but apparently inconclusive because the step size might be too small. 
 Fix: We can try starting scheduled sampling after a good amount of learning. Maybe after the first 100 steps. Let's see how that runs.

 ## Day 4 
 - Ran the Relu and Hashing on Runpod (1XA100) with 4500 steps, surprisingly a very good run. A val-bpb of :  1.2819. This gives me good confidence to run these 2 ideas on a 8XH100. Although the landscape fully changes when we have 8 GPUs interacting with eachother. I am encountering errors and ended up wasting alot of my credits that way. 
Now, to not waste more credits I stopped the run. If my credits get approved I am going to implement a good run soon.

Runs today:
- A100 4500 steps: 1.2819 bpb (LeakyReLU + TrigramHash + scheduled sampling placeholder)
- 8×H100 5274 steps: 1.2631 bpb (first real leaderboard-comparable run)

 Issues found:
- Model size was 16.1MB — 0.1MB over 16MB limit
- Only got 5274 steps in 10 min (TrigramHash overhead slowing steps)

Fixes pushed:
- TrigramHash buckets 8192 → 4096 (saves 0.42MB, speeds up steps)
- fp16 embedding weights (saves 0.5MB, faster lookup)
- Expected model size: ~15.2MB ✓

Next: clean 8×H100 run with fixes → target beat 1.2244 baseline


## Key insights
- Where is the model not communicating when it should be? That gap is always an opportunity.
- On a larger scale, when is complicated too complicated? 
- Which idea should be picked? How does one work within these resource constraints? 
- It is important to have a near identical environment in your local machine. 

Next: implement anchor MLP, smoke test locally and if it beats my score of 2.3, we run it on the GPU. 
