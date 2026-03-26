# OpenAI Parameter Golf Challenge Day 6: The Pod Lottery

*Article 4 of an ongoing series.*

In Article 3, I squeezed 157KB out of a model by switching one compression library, added two techniques I didn't invent, ran three seeds at 1am, and submitted PR #657 at 1.1229 bpb — four ten-thousandths better than the merged SOTA. Then I went to bed.

I woke up to a different competition.

---

## The Leaderboard Moved Without Me

While I slept, someone submitted 0.9674 bpb. Not 1.09. Not 1.05. Zero point nine six seven four. That's 0.16 bpb better than my submission. In a competition where people fight over 0.002.

The technique: n-gram caching. Build frequency tables from tokens you've already scored during evaluation, then mix those statistics with the neural model's predictions. It's backward-looking — you only use tokens you've already graded — so it doesn't violate the rules. Probably. The organizers haven't ruled yet.

Six PRs appeared overnight using variations of the same idea. Multi-order backoff from 7-grams down to unigrams. Entropy-adaptive mixing weights. Zero artifact cost — the tables are built on the fly during eval and thrown away after. The neural model doesn't change. You just augment its predictions with local token statistics.

My 1.1229 went from "matching SOTA" to "6th tier" in twelve hours.

I stared at the leaderboard for a while. Ran the numbers. From a 1.12 neural base, n-gram caching should push you to roughly 0.96-1.03. The gain scales with the quality of your base model. My base is actually stronger than most of the n-gram submissions — they're at 1.127 pre-cache, I'm at 1.122. If I added the same cache, I should beat them.

But should I? The legality question is real. The organizers had already disqualified 25+ PRs in two enforcement sweeps. Full GPTQ with calibration data: illegal. Multi-epoch TTT: illegal. Oracle token selection: illegal. N-gram caching built from scored tokens: ...silence. Six open PRs. Zero organizer comments. Days passing.

That silence is either "we haven't gotten to it yet" or "it's fine." I genuinely don't know which.

---

## TTT: The Final Attempt

Before pivoting to n-gram caching, I had one more thing to try. Test-time training had failed on my architecture three times: SGD at lr=0.002 diverged catastrophically. SGD at lr=0.001 was even worse. The model council diagnosed it as "VRL gate desync" — my Value Residual Learning creates dependencies between layers that break when you modify weights mid-inference.

But then my research agents pulled up PR #688. Their TTT worked. And the recipe was completely different from what I'd been trying:

| Setting | My Failed TTT | PR #688's Working TTT |
|---------|--------------|----------------------|
| Optimizer | SGD(lr=0.002) | AdamW(lr=0.0001) |
| Frozen blocks | 0 (all unfrozen) | 9 of 11 (only last 2) |
| Weight averaging | None | Polyak (decay=0.998) |

Twenty times lower learning rate. Nine blocks frozen instead of zero. And Polyak averaging — you score with smoothed weights, train with live weights. I'd been trying to adapt the entire model. They barely touched it.

I implemented it. Launched it on an 88ms/step India pod. Training finished, sliding window eval came back at 1.1228 — our best ever pre-TTT score. Then the TTT eval started.

Chunk 1/1893: running bpb = 1.193. Higher than pre-TTT.

That's expected — the first chunks have no adaptation history.

Chunk 101: 1.145. Coming down.

Chunk 201: 1.162. Going back up.

I watched it oscillate for 700 chunks. The running bpb never dropped below the pre-TTT baseline. Not once. AdamW was more stable than SGD — it didn't explode — but it still couldn't help. The model was slowly degrading with each chunk of adaptation.

TTT is dead on my architecture. Three optimizers. Four learning rates. Multiple freezing strategies. Polyak averaging. None of it works. The VRL gates were calibrated during training to expect specific weight distributions, and any modification — no matter how gentle — disrupts them.

I killed the run. Stopped the pod. Accepted it.

---

## The Pod Lottery

Here's something nobody talks about in ML competitions: not all GPUs are created equal, even when they have the same name.

I ran the same code on the same "8xH100 SXM" pod template across five different sessions this week. The step times:

| Pod Location | Step Time | Steps in 10 min |
|-------------|-----------|-----------------|
| India (pod A) | 87ms | 6,889 |
| India (pod B) | 91ms | 6,593 |
| India (pod C) | 106ms | 5,660 |
| Japan | 268ms | 2,238 |
| Canada | 272ms | 2,205 |

Same GPU. Same code. Same container image. Three-fold speed difference. The Japan and Canada pods ran at walking pace while the India pods sprinted. The step time directly determines how much data you see in 10 minutes, which directly determines your bpb.

The competition leaderboard is partly a hardware lottery. The top submissions report 83-88ms/step. If you land on a pod that runs at 260ms, you physically cannot produce a competitive result. Not because your model is worse, but because your model saw one-third the data.

I don't know why the speeds differ so much. NVLink topology? Thermal throttling? Different H100 batches? CPU bottlenecks? I just know that every time I spin up a pod, the first thing I do is run a 20-step benchmark. If it's over 120ms, I kill it and try again. At $21.52/hour for 8xH100, each bad pod costs about $2 before I catch it. Each good pod saves about $15 in wasted training time.

---

## Something Changed

Then something happened that I didn't expect.

I was checking the live commentary thread — a community-maintained analysis of every PR in the competition — and I found my name. Not my PR number. My name.

PR #745, a submission at 1.0222 bpb (the best non-n-gram score at the time), listed their six techniques. One of them was "Value Residual Learning (PR #657)." My PR. Credited.

Then I found a commit in someone else's fork. ChideraIbe123, a competitor I'd never talked to, had copied my VRL implementation verbatim into their codebase. 28 lines of code. The commit message cited my PR and the ResFormer paper.

I didn't invent VRL. I implemented it from a paper and proved it worked in competition conditions. And now other people were building on it. The technique I'd added at midnight — 20 lines of code, 10 scalar parameters — was becoming part of the competition's shared vocabulary.

This is the thing about open competitions that I keep forgetting. The goal isn't just to win. It's to contribute something that moves the field. My VRL implementation isn't going to win me the competition. But it might win someone else a few hundredths of a bpb, and they'll stack something on top of it that I'll then learn from. The whole thing is a giant collaborative gradient descent on the problem of "how good can a 16MB language model be?"

I went back to my research system and pulled up the live commentary thread again. This time I wasn't looking at the leaderboard. I was looking at the "Untried Combinations" section — a community-curated list of techniques nobody had tested yet.

There were ideas I'd never heard of. Context Tree Weighting. Logistic-domain mixing. Fixed-Share Hedge with non-stationary expert tracking. Some of them had names that sounded made up. Some of them had arXiv links that I spent an hour reading.

The competition isn't about having the best idea. It's about having the best information. And right now, the information is telling me that n-gram caching is the play — if it survives the legality review.

---

## The Strategic Play

Here's where I am at the end of Day 6.

**What I have:** PR #175 at 1.1229 bpb, three valid seeds, March 19 timestamp (the earliest of any competitive PR because I reopened an old submission). A clean architecture that other people are building on. VRL spreading through the competition.

**What I don't have:** TTT. N-gram caching. Anything that breaks below 1.12.

**What I'm building:** An n-gram cache implementation on a separate branch, isolated from my clean submission. If the organizers rule it legal, I deploy it. If they don't, I still have 1.1229 on PR #175 with the oldest timestamp in the game.

**What I've spent:** Over $1,000 in GPU compute across the week. Four failed FA3 builds. Three failed TTT implementations. Six slow pods killed on sight. Twenty-something full training runs across five days. Two closed PRs. One article I wrote at 3am that more people read than I expected. And the discovery that `pip install flash_attn_3 --find-links .../cu128_torch291` installs in 30 seconds what took me 60 minutes and $100 to build from source. Someone shared that link in the competition thread on Day 4. I found it on Day 6.

**What I've learned:** The hard problems aren't architectural. They're operational. SSH connections that die mid-training. Pods that lose their GPU allocation at step 4500. Container disks that fill up at 99.7% through a CUDA kernel build. Compression libraries that aren't installed on the official template. Pod speeds that vary 3x for the same hardware. Every one of these burned hours and dollars. None of them improved my bpb by a single millinat.

## The Evening: Everything Falls Into Place

Around 8pm, while debugging why the n-gram cache was making things worse (spoiler: I was mixing 30% n-gram noise into good neural predictions, which is like adding static to a clear signal), the research system surfaced a pattern I'd been missing.

Every time I tried to improve my model during evaluation — SGD, AdamW, LoRA, you name it — it broke because the modifications destabilized the VRL gates. The model's internal state was calibrated during training, and any weight change at eval time, no matter how gentle, disrupted that calibration.

But the Hedge Mixer doesn't change weights. At all. It takes my frozen model's predictions and mixes them with simple n-gram statistics using an online learning algorithm. The transformer produces logits. The n-gram tables produce probability estimates. The Hedge algorithm learns, token by token, how much to trust each source. The mixing weights update via multiplication — `w *= exp(-eta * loss)` — not via backpropagation. No gradients flow through the model. No compiled graphs get invalidated. No VRL gates get desynced.

PR #745, the submission that cited my VRL work, uses exactly this approach. Their pre-TTT base model scores 1.1348. After Hedge mixing: 1.0222. A gain of 0.11 bpb from an algorithm that never modifies the model.

My base model scores 1.1229. That's 0.012 better than theirs. If the Hedge algorithm gives even close to the same gain...

I spent the rest of the evening implementing it. Then I stopped. Not because I was stuck, but because I'd spent $30 today on runs that taught me things but didn't move the number. I have $30 left. That's two shots. I need them to count.

The plan for tomorrow is simple. Implement the Hedge Mixer offline (zero GPU cost). Test it once on a fast pod. If it works, run three seeds and update PR #175.

The competition runs until April 30. Five more weeks. The frontier is at 0.96 and dropping. My 1.1229 is irrelevant in the current landscape — unless I can stack the Hedge Mixer on top of it.

I think I can. The research system spent all evening analyzing how the top submissions implemented their mixers, what alpha values they use, how they handle the cold-start problem. By morning there will be a complete implementation plan waiting for me.

The hard problems aren't architectural anymore. They aren't even operational. The hard problem now is: can I execute a clean implementation of a well-understood algorithm, validate it in two runs, and submit before someone else does it better?

Tomorrow I find out.
