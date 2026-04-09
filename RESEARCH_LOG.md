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

## Day 5 
Today felt like a completely start over in terms of strategy. I woke up thinking, what am I doing with the "8" GPUs and the "H100" I have.
Have I thought of optimising my architecture to the hardware? 
This is where I started exploring strategies which distribute computation among the GPUs in an optimised manner. 

Intuition : Increase Step time 
- Use H100s native 8-bit floating point hardware by using a float point of 8, and replace the standard layers with a transformer engine

Intuition : Make GPUs interact with eachother more
- Replace the DDP style interaction to a FSDP (Fully Sharded Data Parallel : this includes weights and gradients) but sharding would be selective, only maybe in the attention layers. But it is worth it to go deeper into understand this. 

I would remove Scheduled Sampling, because we not sure if the model needs Error Recovery if there remains no inference at the end. The challenge is about getting the val_bnb to the lowest, that happens with precision, not 'confusion'. But again, this still is to be tested. Following this train of thought, I will focus on making our trigram hashing more precision. 

Awaiting Runpod Credits :) 

## Day 6-7-8-9 
Everyday can be a new start with a challenge of such a kind. This makes me undersstand how important it is to have a value system that we are training towards or past success which can help anchor all our strategies to, but the best part is the novelty here. 

After going into a spiral of training for infrastructure vs training for more precision. I re-assessed. This re-assession proved to be a start of a completely new direction. I realised that the baseline would already be a optimised for the GPUs, while it is important to optimise for the hardware, how much of a complex change would you want to implement which takes you away from a already well defined GPU-optimised architecture from the baseline. This is where I had to look for a new strategy completely. 
I studied all the "request for implementation" architechture choices presented. 

The one that caught my attention was JEPA, reason one being that it is still not implemented in the non-record submissions and secondly that it's promise to provide more abstraction is truly novel. 

Now, I study and make an architecture optmised for JEPA. 

## Architecture for JEPA  
JEPA : Joint Embedding Predictive Architecture -> https://arxiv.org/pdf/2509.14252
This architecture allows prediction of embedding rather than tokens, helping the LLM look for the underlying meaning rather than the token itself. 

Standard loss : 
- model predicts next token ID
- loss = cross_entropy(predicted_token, real_token)
- penalizes every wrong word equally, even valid synonyms

JEPA loss :
- context encoder processes input tokens → context embedding
- target encoder processes target tokens → target embedding  
- predictor (small MLP) maps context embedding → predicted embedding
- loss = MSE(predicted_embedding, target_embedding)

Things to consider :
1. JEPA adds two new hyperparameters, that would affect our 16MB.
2. JEPA doesn't store Gradients, how does that affect us? What instead? Understand EMA
3. EMA : Exponential Moving Average is the main game, it gets us away from token prediction by become the slow update to the credits and learning, while it keeps the absolute truth but also updates to new knowledge.
4. From reading the paper, the point is to learn and the embedding space and then point to tokens from there 

Strategies : 
1. Build a only JEPA Model : This is to see how it behaves on the baseline, after computing loss we can proceed to optmise it as needed.
2. Build a hybrid model : We start with standard attention layers, use JEPA for the middle layers and then finish with standard attention layers. We operate with two types of losses. 

Let's see how not learning from the absolute truth fairs.

## Implemented JEPA & pushed it train_gpt_jepa.py
1. JEPAPredictor class — 2-layer MLP that maps 
   context embedding → predicted target embedding

2. GPT __init__ — added target encoder (EMA copy of 
   all 11 layers, no gradients) + predictor + 
   jepa_lambda=0.1, ema_decay=0.996

3. update_ema() method — pulls target encoder weights 
   0.4% toward context encoder every step

4. GPT forward — two parallel paths:
   context encoder (gradients) + target encoder (EMA)
   loss = CE + 0.1 × MSE(predicted_emb, target_emb)

5. Training loop — update_ema() called after every step

Validated locally
- forward pass: loss 23.8 (expected for random model)
- EMA update: clean
- import: clean

Next :
- H100 run 
- compare val_bpb vs baseline
- if works → tune jepa_lambda and ema_decay

## Observations from JEPA Run 
8XH100 in 600 seconds 

- It is taking alot of time to start the warmup steps are slow
- step_avg: 64ms (slightly slower than baseline due to two encoders)
- The step avg is increasing as we move forward
- The step avg has also reached 69ms in some steps around the 
- Loss is going down consistently
  
# Val_bpb 
- 1000/7000 :  1.3831
- 2000/7000 : 1.3248
- 3000/7000 :  1.3010
- 4000/7000 : 1.2855
- 5000/7000 : 1.2776 
- 6000/7000 : 1.2702
- 7000/7000 : 1.2646

Final roundtrip : 1.2699
OpenAI baseline: 1.2244
0.045 over baseline which is not bad. 

Model size : large (135MB) : My mistake, I should have verified the layers and model dimensions 

## Next step 
- Reduce model size
- Increase the step_avg above baseline
- Reduce the val_bpb 

## Modified JEPA Run 
- Reduced the model size by deleting the target encoder before serialisation + tried to different approaches with changed hyper-params, reduced model_dim and reduced num_layers 

# Results of reduced model_dim along with a deleted targer enoder : 
Our half run estimated a val_bpb of : 1.28 - 1.30 

# Results of reduced num_layers along with a deleted targer enoder : 
# Val_bpb 
Step 0 : 4.1081 
Step 1000 : 1.4147
Step 2000 : 1.3619 
Step 3000 : 1.3411
Step 4000 : 1.3258
Step 5000 : 1.3179
Step 6000 : 1.3114
Step 7000 : 1.3114
Higher than the reduced model_dim run wich proves that layers are important.

# Model size
Reduced to a 91MB from 135MB 

# Step Avg with model_dim = 386 and num_layers = 9
Average of 55ms 

# Step Avg with model_dim = 512 and num_layers = 6
Average of 40ms (way below baseline) which is obvious but we shouldn't go way below baseline. 

## To think about next 
- How might we reducing the model size?
- Need to utilise each step better 
- Model_dim and num_layers should stay 512 and 9 respectively. 

---
## Key insights
- Where is the model not communicating when it should be? That gap is always an opportunity.
- On a larger scale, when is complicated too complicated? 
- Which idea should be picked? How does one work within these resource constraints? 
- It is important to have a near identical environment in your local machine. This is a good business idea.
- We need depth for LLMs, very important
