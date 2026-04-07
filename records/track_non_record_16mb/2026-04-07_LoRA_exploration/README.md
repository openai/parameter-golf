# LoRA Exploration: Rank Scaling, Technical LoRA Tweaks, and Selective LoRA

Non-record exploration of Learned Adapters on Random Linear Maps. This archive starts from PR #1113, removes ideas not directly related to learned adapters, and compares several LoRA variants and selective-placement settings.

## Training Hardware Setup

The experiments were trained mainly on 4xA6000, with some follow-up runs on 1xH100. For 4xA6000, I first tested with the Naive Baseline setup without LA on an extended wall clock. My assumption is that in the naive baseline, the hardware difference mainly appears as different step average time, while the training steps required to achieve a certain `val_bpb` remain roughly the same. Based on this idea, 2535 s training time on 4xA6000 is treated as equivalent to about 154 s on 8xH100s. This is half of my initial target, as I was actually targeting an equivalent 360 s on 8xH100s. This is also why I switched to 1xH100 later, with training time set to 1440 s, hoping to reach an equivalent number of steps to the 4xA6000 runs.

## Experiment 1: Naive Baseline with LoRA on Different Rank and Layer Number

Baseline: equivalent to the Naive Baseline record run (9 layers), but retried locally.

Experiment command:

Run from repo root:

```bash
RUN_ID=baseline USE_LEARNED_ADAPTERS=0 MAX_WALLCLOCK_SECONDS=2535 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_LA.sh

RUN_ID=baseline_LA_64 USE_LEARNED_ADAPTERS=1 LEARNED_ADAPTER_RANK=64 MAX_WALLCLOCK_SECONDS=2535 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_LA.sh

RUN_ID=baseline_LA_128 USE_LEARNED_ADAPTERS=1 LEARNED_ADAPTER_RANK=128 MAX_WALLCLOCK_SECONDS=2535 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_LA.sh

NUM_LAYERS=11 RUN_ID=baseline_LA_64_11layers USE_LEARNED_ADAPTERS=1 LEARNED_ADAPTER_RANK=64 MAX_WALLCLOCK_SECONDS=2535 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_LA.sh
```

Results:

| Setup | steps | step_avg | pre-quant val_bpb | post val_bpb | artifact size |
|---|---:|---:|---:|---:|---:|
| Baseline | 3576 | 709.11 | 1.2640 | 1.2672 | 15,778,072 B |
| LA 64 rank | 3668 | 691.29 | 1.4249 | 1.4301 | 6,498,206 B |
| LA 128 rank | 3337 | 759.71 | 1.3442 | 1.3478 | 10,940,942 B |
| LA 64 rank 11 layer | 2942 | 861.70 | 1.4104 | 1.4137 | 7,855,642 B |

Increasing rank still provides a better result compared with increasing depth. But this rank increase is also limited, because at some point it no longer provides artifact size reduction; above roughly rank 200 the artifact size would already be higher. At that point the value of using LA becomes questionable.

## Experiment 2: Technical LoRA Tweaks

I used LA 64 rank with 9 layers, with these ideas:

- randLora: Switch to RandLoRA to test the strongest pure-random-basis version of the idea, where learned combinations of frozen random bases remove the usual low-rank bottleneck.
- Rslora: Use high-rank rsLoRA with non-uniform ranks to test whether simply spending the unused budget on much larger adapters is enough to close most of the baseline gap.
- Adalora: Replace fixed ranks with AdaLoRA-style adaptive rank allocation so the model can automatically spend adapter budget on the most important matrices instead of forcing one manual rank schedule everywhere.
- lora_ga: Add LoRA-GA initialization to test whether the main problem is slow or poor convergence rather than insufficient capacity.

Experiment command:

Run from repo root:

```bash
RUN_ID=learned_adapters_ga MAX_WALLCLOCK_SECONDS=2535 USE_LEARNED_ADAPTERS=1 RANDOM_ADAPTER_IMPL=lora RANK_QK_EARLY=64 RANK_QK_LATE=64 RANK_VO_EARLY=64 RANK_VO_LATE=64 RANK_MLP_EARLY=64 RANK_MLP_LATE=64 ADAPTER_ALPHA=64 ADAPTER_INIT=lora_ga LORA_GA_MICROBATCHES=1 MAX_STEPS=12000 WARMDOWN_STEPS=720 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 records/track_non_record_16mb/2026-04-07_LoRA_exploration/train_gpt_lora_ga.py

RUN_ID=rslora MAX_WALLCLOCK_SECONDS=2535 USE_LEARNED_ADAPTERS=1 RANDOM_ADAPTER_IMPL=rslora RANK_QK_EARLY=48 RANK_QK_LATE=64 RANK_VO_EARLY=64 RANK_VO_LATE=96 RANK_MLP_EARLY=96 RANK_MLP_LATE=128 ADAPTER_ALPHA=16 ADAPTER_INIT=standard MAX_STEPS=12000 WARMDOWN_STEPS=4000 SEED=${SEED:-42} OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 records/track_non_record_16mb/2026-04-07_LoRA_exploration/train_gpt_rslora.py

RUN_ID=adalora MAX_WALLCLOCK_SECONDS=2535 USE_LEARNED_ADAPTERS=1 RANDOM_ADAPTER_IMPL=adalora ADALORA_INIT_RANK=128 ADALORA_TARGET_AVG_RANK=80 ADALORA_TSTART=1000 ADALORA_TEND=7000 ADALORA_DELTA_T=200 ADALORA_BETA1=0.85 ADALORA_BETA2=0.85 ADAPTER_INIT=standard MAX_STEPS=9000 WARMDOWN_STEPS=2500 SKIP_GPTQ_DURING_SWEEP=0 SEED=42 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 records/track_non_record_16mb/2026-04-07_LoRA_exploration/train_gpt_adalora.py

RUN_ID=randlora MAX_WALLCLOCK_SECONDS=2535 USE_LEARNED_ADAPTERS=1 RANDOM_ADAPTER_IMPL=randlora RAND_BASIS_RANK=16 RAND_NUM_BASES_QO=32 RAND_NUM_BASES_KV=16 RAND_NUM_BASES_MLP=32 RAND_LATE_LAYER_MULT=2 ADAPTER_INIT=standard MAX_STEPS=10000 WARMDOWN_STEPS=3000 SKIP_GPTQ_DURING_SWEEP=0 SEED=42 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 records/track_non_record_16mb/2026-04-07_LoRA_exploration/train_gpt_randlora.py
```

Results:

| Setup | steps | step_avg | pre-quant val_bpb | post val_bpb | artifact size |
|---|---:|---:|---:|---:|---:|
| Rslora | 2148 | 1180.60 | 1.4437 | 1.4454 | 6,832,491 B |
| lora_ga | 2187 | 1159.60 | 1.4737 | 1.4765 | 6,568,722 B |
| Adalora | 1996 | 1270.57 | 1.4168 | 1.4180 | 10,650,382 B |
| Randlora | 1301 | 1948.61 | 2.1259 | 2.1267 | 585,709 B |

While all of these ideas look promising in the literature, all of them add a significant step average time. As such they are not well suited to this challenge, where we are limited by training time rather than total training steps.

## Experiment 3: Selective LoRA

Lastly, I tested selective LoRA placement. The main motivation comes from these papers:

- Layer-wise LoRA fine-tuning: a similarity metric approach (https://arxiv.org/abs/2602.05988)
- PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models (https://arxiv.org/abs/2506.20629)

Their main gist is that applying LoRA to all layers is inefficient, and that we can target specific layers by depth or type and achieve a result comparable with full fine tuning, with much more reduced parameters. Of course our case is not fine tuning, but the idea could still be applicable, especially if we want to reduce parameter count and artifact size.

For this one, I used 1xH100. For comparison, I use the Naive Baseline leaderboard log at a similar number of steps, around 3800-4000. Here I only used 1440 s training time, which is equivalent to about 2.8 minutes on 8xH100s. I use the pre-quant `val_bpb` result here, because I need a comparison but I have not rerun the Naive Baseline on my exact setup. I also report total size after quantization. All LoRA runs here use rank 64.

Experiment command:

Run from repo root:

```bash
RUN_ID=1H100_LA4-6 MAX_WALLCLOCK_SECONDS=1440 LEARNED_ADAPTER_LAYERS=4-6 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_targeted_LA.sh

RUN_ID=1H100_LA-MLP MAX_WALLCLOCK_SECONDS=1440 LEARNED_ADAPTER_TARGETS=mlp bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_targeted_LA.sh

RUN_ID=1H100_LA-MLP4-6 MAX_WALLCLOCK_SECONDS=1440 LEARNED_ADAPTER_TARGETS=mlp LEARNED_ADAPTER_LAYERS=4-6 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_targeted_LA.sh

RUN_ID=1H100_LA-ATTN4-6 MAX_WALLCLOCK_SECONDS=1440 LEARNED_ADAPTER_TARGETS=attn LEARNED_ADAPTER_LAYERS=4-6 bash records/track_non_record_16mb/2026-04-07_LoRA_exploration/run_targeted_LA.sh
```

Results:

| Setup | Result |
|---|---|
| Naive Baseline | step:3800/20000 val_loss:2.1767 val_bpb:1.2858 |
| Naive Baseline | step:4000/20000 val_loss:2.1718 val_bpb:1.2829 |
| LoRA on layer 4-6 | step:3780/12000 val_loss:2.1794 val_bpb:1.2908 train_time:1440145ms step_avg:380.99ms Total submission size int8+zlib: 12704423 bytes |
| LoRA on MLP only | step:3732/12000 val_loss:2.2807 val_bpb:1.3508 train_time:1440206ms step_avg:385.91ms Total submission size int8+zlib: 9908490 bytes |
| LoRA on layer 4-6 MLP only | step:3991/12000 val_loss:2.1572 val_bpb:1.2776 train_time:1440030ms step_avg:360.82ms Total submission size int8+zlib: 13855019 bytes |
| LoRA on layer 4-6 ATTN only | step:3941/12000 val_loss:2.1341 val_bpb:1.2639 train_time:1440268ms step_avg:365.46ms Total submission size int8+zlib: 14724469 bytes |

A good result here is one that beats the Naive Baseline `val_bpb` around 3800-4000 steps, which is around 1.2829-1.2858. A selective layer depth or type alone does not provide a better result here. But when they are combined, they can provide a better `val_bpb`, namely 1.2776 and 1.2639. This result is interesting, because LoRA fine-tuning typically takes more steps to converge than full fine tuning. We could assume the same behavior should also apply to training from scratch. Therefore, a better `val_bpb` at this step budget, equivalent to 2.8 minutes of the full 10 minute budget on 8xH100s, should indicate a much better result when trained fully.

Additionally, this also comes with extra size reduction, around 87% and 93%. If this reduction remains true in the full 8xH100 run, then the extra size can be used for other improvements, for example increasing bigram size.

