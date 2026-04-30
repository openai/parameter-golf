<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...). 

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest time given constrained loss) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest loss given constrained dataset size), can be thought of as equivalent challenges in this family.

Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting *leaderboard submissions* to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.

We also know compute is expensive, so **OpenAI is sponsoring $1,000,000 in compute credits** to help people get started training their models. To request a compute grant, use this form: [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form).
When requesting compute, please make sure you choose the appropriate level, write sufficient justification, and **submit with an email tied to a OpenAI / ChatGPT account**.

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th. 

Happy training!

## Leaderboard

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| BOS-Fixed SmearGate + LQER + SparseAttnGate + 9-Hparam Stack | 1.0611 | codemath3000 | On PR #1855: BOS-fixed #1797-derived stack with LQER, PR #1787 SparseAttnGate/PolarNS/FusedCE base, per-group lrzip compression, and 9 greedy hyperparameter overrides; submitted 3-seed mean 1.06108 with broader reproduction support (p=0.188 vs PR #1868 latest rerun) | 2026-04-27 | [info](https://github.com/openai/parameter-golf/pull/1855), [repro](https://github.com/openai/parameter-golf/pull/1855#issuecomment-4336629746) |
| BOS-Fixed SmearGate + LQER Asymmetric + PR1787 SparseAttn + Phased TTT | 1.0614 | aquariouseworkman | On PR #1851 with 3-seed compliance-rerun support from PR #1868: BOS-boundary fix from PR #1851 applied to dexhunter's PR #1797 SmearGate + LQER stack, using the PR #1787 SparseAttnGate/PolarNS/FusedCE base plus CaseOps and phased score-first TTT | 2026-04-27 | [info](https://github.com/openai/parameter-golf/pull/1851), [3-seed](https://github.com/openai/parameter-golf/pull/1868) |
| PR1736 + PolarNS + MIN_LR + SparseAttnGate + FusedCE + Warm-A TTT | 1.0634 | nprime06 | On PR #1787: PR #1736 CaseOps stack plus Polar Express Newton-Schulz coefficients, MIN_LR=0.1, SparseAttnGate, fused softcapped CE, and PR #1767-style warm-start-A TTT | 2026-04-23 | [info](https://github.com/openai/parameter-golf/pull/1787) |
| CaseOps + MLPClip12 + SmearGate/LoRA-TTT | 1.0645 | dexhunter | On PR #1769: CaseOps stack with SmearGate/LoRA-TTT refinements and MLPClip12; 5-seed mean improves the accepted CaseOps frontier (p=0.063 vs #1736) | 2026-04-22 | [info](https://github.com/openai/parameter-golf/pull/1769) |
| SP8192 + CaseOps + GatedAttn + QuantGate + Loop45 + Phased TTT | 1.0655 | dexhunter | On PR #1736: adopts romeerp's lossless CaseOps transform from PR #1729 with byte-sidecar BPB accounting, then adds gated attention and quant-gate scaling on the PR #1530 SP8192 phased-TTT stack | 2026-04-19 | [info](https://github.com/openai/parameter-golf/pull/1736) |
| CaseOps Tokenizer + Tapered WD + Phased TTT | 1.0678 | romeerp | On PR #1729: lossless CaseOps bijective case transform with validation byte sidecars, plus mild late Muon weight-decay taper on the PR #1626 legal phased-TTT stack | 2026-04-19 | [info](https://github.com/openai/parameter-golf/pull/1729) |
| SmearGate + Attention Output Gate + Legal TTT | 1.0714 | MarioPaerle | On PR #1667: SmearGate, attention output gate, depth recurrence, parallel residuals, QK-Gain 5.25, quantization, and score-first TTT | 2026-04-16 | [info](https://github.com/openai/parameter-golf/pull/1667) |
| VarLen Attention + Fused MLP + Multi-Phase Global SGD TTT | 1.0719 | dexhunter | On PR #1626: VarLen attention, fused MLP, multi-phase global SGD TTT, trimmed GPTQ, MLR 0.026, int7 embeddings, and adaptive clip | 2026-04-14 | [info](https://github.com/openai/parameter-golf/pull/1626) |
| VarLenAttn + PhasingTTT | 1.0728 | romeerp | On PR #1610: #1530-style VarLen/fused stack plus phased TTT over already-scored validation chunks | 2026-04-13 | [info](https://github.com/openai/parameter-golf/pull/1610) |
| VarLen Attention + Fused MLP + Doc-Independent Legal TTT | 1.0734 | samacqua | On PR #1530: variable-length FA3 attention, fused Triton MLP, grouped small-parameter all-reduces, and doc-independent score-first LoRA TTT | 2026-04-11 | [info](https://github.com/openai/parameter-golf/pull/1530) |
| Improved Parallel Residuals + CUTLASS EVT + Legal TTT | 1.0758 | msisovic | On PR #1529: PR #1523 SP8192 baseline with fuller two-lane parallel residual routing, PARALLEL_RESIDUAL_START=8, inline CUTLASS EVT/Triton fused kernels, and legal score-first TTT; corrected 3-seed mean after GPTQ reserve/seed fix (p=0.001 vs #1514) | 2026-04-11 | [info](https://github.com/openai/parameter-golf/pull/1529) |
| SP8192 + Muon 0.97 + Legal Score-First TTT | 1.0798 | dexhunter | On PR #1514: SP8192 with Muon 0.97 and legal score-first TTT; 3-seed sweep beats #1493 (p=0.020) | 2026-04-09 | [info](https://github.com/openai/parameter-golf/pull/1514) |
| SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT | 1.0810 | bigbag | On PR #1493: 3-layer recurrence, parallel residuals, QK-Gain 5.25, and legal score-first TTT on the PR #1394 stack | 2026-04-09 | [info](records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md) |
| SP8192 + Parallel Residuals + Score-First TTT | 1.0822 | aryanbhosale | On PR #1477: parallel residuals on the PR #1413 SP8192 + legal score-first TTT stack | 2026-04-08 | [info](records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/README.md) |
| SP8192 + QK-Gain 5 + Legal Score-First TTT | 1.0828 | dexhunter | On PR #1413: QK-Gain 5.0 + legal score-first TTT on the PR #1394 SP8192 stack | 2026-04-06 | [info](records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/README.md) |
| SP8192 + Parallel Residuals + Hessian-Aware SDClip | 1.0835 | Robby Sneiderman | On PR #1412: parallel residuals, Hessian-aware SDClip, and progressive recurrence on the PR #1394 stack | 2026-04-06 | [info](records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/README.md) |
| SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip | 1.0856 | Kevin Clark | On PR #1394: SP8192, GPTQ embeddings, looped layers 4-5, MuonEq-R, and std-based GPTQ clipping | 2026-04-05 | [info](records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/README.md) |
| SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R | 1.0897 | aryanbhosale | On PR #1334: SP4096 + depth recurrence + parallel residuals + MuonEq-R + QK-Gain 5.0 | 2026-04-04 | [info](records/track_10min_16mb/2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR/README.md) |
| MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ | 1.0912 | dexhunter | On PR #1285: MuonEq-R + layers 4-5 recurrence + higher weight decay + all-int6 GPTQ | 2026-04-03 | [info](records/track_10min_16mb/2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/README.md) |
| 4096-Vocab + Larger Model + High WD + Simplifications | 1.0979 | Kevin Clark | On PR #1218: SP4096 + 4x MLP + high weight decay, with TTT, hash embeddings, SmearGate, and value residuals removed | 2026-04-01 | [info](records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/README.md) |
| Parallel Residuals + Mini Depth Recurrence | 1.1063 | Marko Sisovic | On PR #1204: mini recurrence on layers 4-5 + parallel attention/MLP residual lanes + AR self-generated GPTQ calibration | 2026-03-31 | [info](records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/README.md) |
| Rascal | 1.1099 | newjordan | On PR #1120: XSA-all + Parallel Muon + coprime loader + Bigram2048/RoPE16 + SWA/late QAT without GPTQ | 2026-03-30 | [info](https://github.com/openai/parameter-golf/pull/1120) |
| Coprime-Stride Loader + Full GPTQ + XSA-all | 1.1122 | dexhunter | On PR #1060: coprime multi-shard loader + Full Hessian GPTQ + XSA on all layers + BigramHash(2816x112) | 2026-03-29 | [info](records/track_10min_16mb/2026-03-29_Loader_FullGPTQ_XSA11_BigramHash2816/README.md) |
| 11L AR Self-Gen GPTQ + XSA | 1.1147 | abaybektursun | On PR #1019: Self-Generated GPTQ Calibration Data + all-layer XSA on the PR #549 stack | 2026-03-25 | [info](records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md) |
| LeakyReLU² + Legal Score-First TTT + Parallel Muon | 1.1194 | abaybektursun | On PR #549: LeakyReLU(0.5)^2 + TTT + Parallel Muon on the PR #414 stack | 2026-03-23 | [info](records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md) |
| 11L EMA + GPTQ-lite + warmdown3500 | 1.1228 | signalrush | On PR #374: GPTQ-lite clip search + EMA, plus warmdown3500 and QAT@0.15 | 2026-03-22 | [info](records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md) |
| 11L Partial RoPE + LN Scale + EMA + XSA4 | 1.1248 | jfprincz | On PR #287: Partial RoPE (16/64) + layerwise LN scale | 2026-03-21 | [info](records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md) |
| 11L XSA4 + EMA + Int6 MLP3x | 1.1271 | jfprincz | On PR #198: XSA on the last 4 layers + EMA replacing SWA | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md) |
| 11L Efficient Partial XSA | 1.1307 | unnir | On PR #198: Efficient Partial XSA on the deepest 3 layers | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md) |
| 10L Int5-MLP + BigramHash(10240) | 1.1428 | thwu1 | 10 layers, mixed int5/int6 quantization, BigramHash(10240), SWA(0.4), WD=0.04 | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) |
| Int6 MLP3x + SmearGate + BigramHash | 1.1458 | Raahil Shah | 3x MLP + SmearGate + BigramHash + OrthoInit + Muon WD + SWA | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md) |
| 11L MLP3x + Int6 QAT | 1.1502 | aruniyer | 11 layers, 3x MLP, int6 QAT, zstd-22, WD=0.04, sliding eval | 2026-03-20 | [info](records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md) |
| SmearGate + OrthoInit + Muon WD | 1.1556 | aquariouseworkman | SmearGate + BigramHash + 3x MLP + int6 STE QAT + sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md) |
| Ternary Quantization | 1.1570 | Ciprian-Florin Ifrim | 73.7M params quantized to 1 0 -1 + misc arch changes | 2026-03-24 | [info](records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/README.md) |
| 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | yahya010 | 10 layers, int6 QAT + zstd-22, MLP 1344, Muon 0.99, sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/README.md) |
| Mixed Quant + Sliding Window Eval | 1.1630 | aquariouseworkman | Int6 block weights + int8 embeddings + 3x MLP + sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md) |
| Muon WD + 10 layer | 1.1748 | notapplica | Includes prev. wins + Spectral embed init + resid mix | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) |
| Sliding Window Eval | 1.1925 | Matthew Li | Sliding window evaluation at stride=64, increasing context for eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md) |
| Lora TTT | 1.1928 | samacqua | Test-time training with LORAs | 2026-03-19 | [info](records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md) |
| 4k seq length| 1.2014 | Spokane Way | 4k seq length + better hypers | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md) |
| 2048 seq length | 1.206 | Spokane Way | 2048 seq length (train + val) | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md) |
| int6 mixed precision | 1.2147 | Nan Liu | 10 layers, mixed int8/int6 | 2026-03-18 | [info](records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md) |
| fp16 Embed | 1.2197 | Renier Velazco | FP16 Tied Embedding + LR/Warmdown Tuning | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md) |
| Naive Baseline | 1.2244 | Baseline | 9layer 512dim 1024vocab TiedEmbeddings 4 KV heads | 2026-03-18 | [info](records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) |

#### Unlimited Compute Leaderboard & Non-record Submissions

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| 1 Bit Quantization | 1.1239 | Ciprian-Florin Ifrim | 106M params quantized to 1 bit + misc arch changes + 2hr training | 2026-03-24 | [info](records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/README.md) |
| 4-Hour Baseline | 1.2074 | Will DePue | Testing unlimited compute, 4 hours on 8xH100 | 2026-03-18 | [info](records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |

#### Requests for PRs

Breakthrough ideas are rarely immediately state-of-the-art, instead, they're developed slowly, first demonstrating signs-of-life, iterated on, then only ultimately optimized on the systems side. Don't get discouraged if a new algorithm doesn't instantly beat the best leaderboard run or even the naive baseline. If you have an idea you believe in, consider ignoring step times early on: once you prove you can beat the baseline in the same # of steps you can then start focusing on how to also make it fast.

We'd love to see weird & creative ideas in the challenge, since you never know what may work in the end. Most likely, these will be a good fit in our unlimited compute leaderboard as non-record submissions. We have some requests for what we'd love to see people implement:

- [x] 1-bit quantization - [implementation](records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/README.md)
- [x] Ternary quantization - [implementation](records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/README.md)
- [ ] JEPA
- [ ] Text diffusion
- [ ] H-net tokenization
- [ ] Universal transformer - [We have lots of depth recurrence submissions, but I'd love to see one 4 hour
- [ ] Megakernels
- [ ] State-space models, E2E TTT, super long context for evaluation or training 
- [ ] Learning adapters on random linear maps

## Getting Started

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with Runpod.

First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download our cached version of FineWeb with the 1024-token vocabulary:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default this downloads the full validation split plus 80 training shards (8B tokens). For a smaller local smoke subset, pass `--train-shards 1`, for example `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Validation always runs on the full `fineweb_val_*` split, which is the fixed first-50k-document set. The smoke command above skips periodic validation and just prints the final `val_loss` and `val_bpb` once at the end.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.  

#### Running on Modal

If you prefer Modal for iteration, install and authenticate the client first:

```bash
pip install modal
python3 -m modal setup
```

Then launch the public baseline-style single-node 8xH100 worker with the helper script in this repo:

```bash
python3 -m modal run modal_train.py \
  --run-id modal_baseline_sp1024 \
  --script train_gpt.py
```

This launcher uses the public PyTorch `2.9.1 + cu128` stack, mounts only the trainer and dataset helper code, and stores datasets, tokenizers, Hugging Face cache, and run outputs in Modal Volumes. By default it prepares the `sp1024` dataset with `80` training shards. To reuse an existing cached dataset volume, add `--skip-dataset`.

At the time of writing, the Modal H100 offering used with this launcher is the SXM variant.

To tune the launcher or trainer, pass overrides with `--extra-env`, for example `--extra-env 'MAX_WALLCLOCK_SECONDS=600,VAL_LOSS_EVERY=200'`.

#### Launching a 1xH100 Pod

1. First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s (specifically the SXM variant), but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up. You should land in `/workspace/`.

On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This defaults to the full validation split plus 80 training shards (8B tokens). If you only want a smaller subset while iterating, pass `--train-shards N`, for example `--train-shards 1`.

Launch your first training run. Note that we're passing `nproc_per_node=1` because we're running on a single H100 GPU in this case.

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

By default, `train_gpt.py` keeps its ~10 minute wallclock cap. If you want a longer run, override it explicitly, for example `MAX_WALLCLOCK_SECONDS=0`.

By default, this command prints `train_loss` step logs during training and prints `val_loss`, `val_bpb`, and compressed model size in the final `final_int8_zlib_roundtrip` lines at the end. If you want periodic validation logs during the run, set `VAL_LOSS_EVERY`, for example `VAL_LOSS_EVERY=200`. For the baseline config, the final `val_bpb` should land around ~1.2 with a compressed model size under 16MB.

For dataset export, tokenizer export, and docs-cache rebuild instructions, see [data/README.md](data/README.md).

Evaluation will be in the RunPod environment with all packages installed. `requirements.txt` is provided as a reference if you want to self-setup.

## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR. If you find an issue with a record on the leaderboard or find a record isn't reproducible, please let us know and add an Github Issue describing your findings.

**What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?**

There's no perfectly clear answer here and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it. Use your best judgment and there's no penalty for asking questions.

**What are the restrictions on evaluation?**

We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!), but otherwise you're free to evaluate however. As with modded-nanogpt, we allow evaluation at any sequence length. And, obviously, you aren't allowed to access any training data during evaluation, unless you pay for those bits in the <16MB limit. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods. You CANNOT access validation data during training, e.g. by compressing it into your 16mb with "paid prefix".

If it isn't abundantly obvious: You can't cheat on your test loss. You can't cheat by training on the validation set before you evaluate on the validation set. The validation language around test-time training has been confusing people: you are only allowed to test-time train on validation set tokens _you've already evaluated your model on_, since those tokens have already been graded!

**What is the process for accepting new submissions?**

Since all submissions are public, we're accepting record submissions chronologically depending on their PR creation time. The leaderboard may take time to update due to verification and review of submissions, so pay consideration to what the current SOTA PR is when submitting. As explained below, submissions should exceed the SOTA record with sufficient statistical significance in order to be accepted for the leaderboard. Otherwise, submissions may be accepted as 'non-record submissions' given they are sufficiently unique or interesting.

**Can I import XYZ package or library?**

Yes, you're free to import any package or library you want, so long as it does not unjustly violate the rules on evaluation, compute, training time, code size or otherwise. Just include a requirements.txt in your records folder and mention setup instructions in your README.md. Since you don't pay for bits imported in Python libraries, limitations clearly apply: You can't sneak in extra compute, capabilities, or massively increase effective code size with custom libraries, but importing FlashAttention, etc. is completely fine.


## Submission Process

New SOTA records must fulfill the following criteria:

1. They must beat the existing SOTA by at least 0.005 nats. As in modded-nanogpt, because of inter-run variance all submissions must provide enough run logs to show at `p < 0.01` that they achieved the required 0.005-nat improvement. For submissions that improve speed through systems optimization without changing the ML, this requirement is waived.

2. If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be made as a pull request that only adds a new folder to the appropriate `/records` subfolder and includes the following files. Submissions without the full set of requirements will not be accepted.

1. A README.md file that explains the submission in reasonable detail.

2. A `submission.json` file (see the example runs) that includes your name, GitHub ID, `val_bpb`, and related metadata.

3. A train log, automatically produced by your script. Please demonstrate a statistically significant win. Most often, submitting an average over 3 training runs is sufficient.

4. A `train_gpt.py` script and any other dependencies. Note: this must successfully compile and run within the records folder. Broken scripts will not be accepted.

### Non-record Submissions

Submissions are also open to unique and interesting approaches that might not beat the existing SOTA, but still satisfy the 16MB artifact limit. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions, so long as they run successfully, or even interesting negative results. We're excited to see what you come up with. We'll still maintain a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting.

We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff. Just note as such in your README file.

Non-record submissions should be made in the same fashion as SOTA records, as described above.

#### PRs on Core Code

The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but the best models should stay in the `/records` folder.

## Support


Join the [OpenAI Discord server](https://discord.com/invite/openai) and visit the Parameter Golf channels (#parameter-golf-discussions, #parameter-golf-announcements) and ask questions.

This repository adapts code from `modded-nanogpt`, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
