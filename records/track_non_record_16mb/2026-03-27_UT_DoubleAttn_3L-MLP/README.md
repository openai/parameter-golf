This run uses a shared UT-style recurrent block with two attention layers before the MLP, so the model can form circuits like induction heads before passing through the MLP. I also changed the feedforward to a 3-layer MLP (added a fully connected layer between the up and down proj), which felt like a cleaner use of parameters than pushing a standard MLP all the way to `16x` just to match KV-pair count. (KV-pair view of MLPs as described in https://arxiv.org/pdf/2505.19488v1).

The norms stay independent across depth, and I add a bias to pre-norms. The bias is important to get this to work, since it acts like a depth embedding, adding a different vector at each depth while still sharing the main weights. For quantization, I reused the noisy QAT idea from the other non-record DepthRecurrence submission. I am not sure how optimal it is here, but it helped a bit on quantized BPB.

A big part of making this competitive is the layer/depth schedule. Training at a lower depth early on is something enabled by UT and can save a lot of time. There could be ways to speed it up even further with early exiting strategies. 

All scheduled depths are compiled up front in the warmup/priming stage (an idea I got from modded-nanogpt speedrun), so we don't hit recompiles when switching. This run uses `NUM_LAYER_SCHEDULE=0:2,2000:6`; the schedule itself can probably be tuned a lot more since with limited compute, I could only guess what would transfer to full `8xH100` scale, and it doesn't seem optimal. 

I removed the UNet style extra skip connections for simplicity, as I'm not sure it's a good fit for shared weights. Another direction to explore could be re-adding this by having 2 sets of weights, one for encoder and one for decoder layers then repeating both.

It also does not include any of the leaderboard improvements made since the baseline. If I can get more compute i will continue experimenting with it, and I'm confident it could be a good path for others to use as a starting point later on.

This run trained at `NUM_LAYER_SCHEDULE=0:2,2000:6` under the `600s` wallclock cap and stopped at step `6011`. Final numbers from [train.log.txt](/Users/serdargulbahar/GitHub/parameter-golf/records/track_non_record_16mb/2026-03-27_UT_DoubleAttn_3L-MLP/train.log.txt): pre-quant `val_bpb=1.2542`, final int8+zlib roundtrip `val_bpb=1.25595494`, total size `15,982,324` bytes.
