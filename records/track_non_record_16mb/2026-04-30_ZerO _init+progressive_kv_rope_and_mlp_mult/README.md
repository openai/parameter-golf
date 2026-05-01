# Constrained by Time
*(Not in the traditional sense)*

Between schoolwork and other commitments, this was definitely an on-and-off type of project.
- Prior to this, I have never trained any language models of any kind, let alone in a speedrunning context.
- I was planning on full-scale testing but had to wait until 4/30/2026 for a sufficient compute grant for full H100 testing. With only a day to really experiment, there wasn't much time for me to validate anything or find very good ideas.
	- Smaller-scale testing was done on Google Colab and the RTX 5090s provided by RunPod, but these were slow and it was hard to really know what worked when moving so slowly.

There were many (frankly) strange implementations that I tried experimenting with with the help of AI.
- Extreme depth/low dim
- Low depth/high dim
- Rolling data inside each batch (give the model one more chance at "harder" data)
- Dynamic loss scaling
- You'll find more by going through [my repo's](https://github.com/AlstonTang/parameter-golf) commit history.

Although this probably isn't going to shatter any records (at all), I do hope that this at least shines an interesting light at some potential ideas that may be integrated into future GPT training/speedruns.
- Even if this ultimately doesn't go very far alone, it would be nice if some of the ideas in this implementation were explored. Perhaps there's too much going on in this implementation, and that together they're conflicting each other. Or perhaps's it's merely a hyperparameter configuration away from getting solid results.

My focus ended up not being on Test-Time Training (TTT) or any significant implementation-specifc optimizations (e.g. fp8 training). Rather, my focus was on the underlying architecture itself, and really (trying towards) pushing the limits of what a conventional transformer can do.
- I may continue experiments even after this competition, since research isn't just one and done! I may take some ideas from my implementation and iteratively add it to new language models I may train in the future as I both learn more about LLMs and advanced deep learning in general.
- Some commits may show attempts implementation-specific optimizations (e.g. my attempt at fp8 training), but these usually either failed or led to training instability during experimentation.

## ZerO initalization
I was intersted in this paper: https://arxiv.org/pdf/2110.12661
- Zhao et. al. describes how performance of deep networks can be both better and more reproducable through a more deterministic initialization method involving Hadaramard/Identity-like matrices

The actual usage of this initialization does involve some level of non-determinism, but it's less pronounced than fully random initalization.

## Progression of Various Model Hyperparameters
Throughout each layer, I utilized a progression of KV head count, rope proportion, and MLP multiple, all of which increase as the layer's depth w.r.t. the model increases. The rationale is as follows:
1. Earlier layers most likely focus on nearby context and shouldn't worry about long-range dependencies.
2. As tokens go further into the model, more information is going to be needed.

Whether these should all be scaled linearly, geometrically, or something else is a question for another day.

## So, could it work?
Well, maybe if there was more time (to experiment + training time) and a better implementation.
- Of course, hyperparameter choice also plays a role. However, I did not have too much time to really test anything.

## What would I have done if I had more time?
1. Tuning hyperparameters.
2. More experimentation with other strange hypotheses.
3. Optimizing the implementation.

If I wasn't constrained by the constraints, I would also test it on larger-scale models (e.g. 100 million+ parameters).
- Perhaps the model was too small to really realize the potential gains of my proposed implementation.
- Going beyong 16 MB would be nice to see if my ideas could potentially fly!

## Why just one Result?
Time...
- If I had more time, I would submit more results (and probably more refined ones).
- The result should (hopefully) be reproducable across seeds due to the more deterministic [ZerO intialization](#zero-initalization) being used.
