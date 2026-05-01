# Redoing ZerO and A Follow Up To PR 2104
The original [writeup](https://github.com/AlstonTang/parameter-golf/blob/fef7edc8c96ce169f31754f8deae1334a76e0fba/records/track_non_record_16mb/2026-04-30_ZerO%20_init%2Bprogressive_kv_rope_and_mlp_mult/README.md) in [PR 2104](https://github.com/openai/parameter-golf/pull/2104) was written semi-hastily in order to at least submit something in time within the deadline. However, after a bit more experimentation, there were some surprising results!
- The ZerO implementation from generative AI used in PR 2104 was flawed.
- If you just want to see results, either view logs or see [here](#the-surprise) for a high-level summary.

In this submission I dump various training logs, plus the final train_gpt.py used in the final run.
- Keep in mind that each training log has a copy of train_gpt.py used during the run.

I do not explore whether progression of various hyperparameters within each layer will have worked.
- for more details + implementation, see the [pull request](https://github.com/openai/parameter-golf/pull/2104).

## Correcting ZerO
Previously, I had generative AI try to implement the [ZerO](https://arxiv.org/pdf/2110.12661) initialization function. However, due to issues in prompting and visibility of the document (a URL was sent instead of the underlying PDF document), the model most likely hallucinated the implementation.
- It did take inspiration from the usage of Hadamard matrices, though this could also have been a hallucination or perhaps was not fully memorized from the training corpus that the LLM I prompted went through.
- However, due to various other ideas being tested concurrently, validation of this implementation was largely overlooked during the challenge.
	- Admittedly, it isn't a really good idea to test too many things at once.

However, post-challenge, I wanted to see what ZerO could really do given a fresh start. I had previously suspected that the implementation was perhaps somewhat off due to non-determinisism being used, but I hadn't really thought of generating a fresh implementation until now.

When feeding generative AI the actual PDF, it was able to more accurately generate a valid implementation of ZerO.
- The implementation used in this submission does include usage beyond transformers (e.g. Convolutions).
- The implementation also takes into account the existing transformer implementation in the original train_gpt.py.

Note that, although I tried to set $W_k$ and $W_v$ as zeroed matrices as described in the paper, since the projection right after attention was also zeroed out due to the existing implementation, the model trained very poorly.
- Hence, there is still a slight deviation from the canonical ZerO implementation.

## Reimplementation
The implementation is based on the original train_gpt.py, not the submitted one in PR 2104.
- This means that we can more accurately see whether or not ZerO works and is doing its thing instead of it being potentially masked by the other hypotheses concurrently tested within the submitted train_gpt.py in PR 2104.

## So was ZerO the Bottleneck?
It would seem so, at least for GPT speedrunning. Although more testing is needed, when reading through section 4.1, you will see that the standard initialization actually beats out ZerO with a small number of layers.
- Perhaps ZerO may perform better with more layers and/or more training time.

Interesting things to note are that:
1. Experimentation of ZerO within the paper was primarily focused on Convolutional Neural Networks (specifically ResNet).
2. Much of the paper focuses on the math (and proofs) instead of emperical testing. This could explain why the initialization makes the model slightly worse in practice despite sounding better on paper.

I do have a [few plans](#future-work) to see if ZerO really is a bottleneck in a more realistic setting beyond speedrunning.

## Experimentation
Without the constant knowledge of having to get something submitted, this went surprisingly smoothly
- Less focus on tuning hyperparameters to get something in one day, more focus on getting things right.

With a much more accurate and deterministic intialization, it could be likely that one training run is all that is needed when testing hypotheses moving forward.

Initially, there was an invalid implementation due to the naming conventions used in the train_gpt.py file. This was resolved in later training runs, but for convenience, the following list shows the names of log(s) containing invalid implementations:
- e31e596e-e21a-4c48-829c-78233c992cc8.txt
- fe927335-4827-415a-b543-8b5d2706de4c.txt

These are still included so that you can kind of see the experiemntation process. All other logs contain the correct implementation of ZerO.

## The surprise
As it turns out, due to the fundamental nature of ZerO, we actually get a much better compression result.

Within run_logs/3c25e790-2f8a-4c36-881c-67066cf1e465.txt, although the model contained 17,059,912 parameters, the serialized model only reaches a size of 13,695,416 bytes. Although this could be partly attributed to a higher bpb (suggesting that the model learned less within the 10 minutes of training time), I strongly suspect that this is due to the low-rank learning trajectory that the model goes through within ZerO initialization. Because of this, the model is naturally more compressible.

Adding one more layer (so that the model has 10 layers, 8 query heads, 4 kv heads, and a dim of 512 with mlp_mult of 2), and running it for the full 20,000 steps (see run_logs/24d3334a-0ddf-45ca-8be8-9dbd470f8866.txt), we get a final bpb of ~1.2494 with the submission size still only taking up 15,221,665 despite a parameter count of 18,897,488.

I initially thought that ZerO would at least yield better loss. Although it didn't, the improvements in compression were very surprising, and I'm interesting in further increasing parameter efficiency with ZerO.

## Future Work
My plans for this consist of the following:
1. Testing ZerO with more step counts
2. Investigating how to best alter hyperparameters using ZerO initialization
3. Using ZerO with much larger (and deeper!) models.
