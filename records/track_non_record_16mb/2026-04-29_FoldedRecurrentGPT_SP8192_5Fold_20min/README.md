# Paper-Folded Recurrent GPT

**final int8+zlib roundtrip val_bpb: 1.19084839**  
**Conservative packaged artifact size: 13,701,318 bytes**  
**Parameters: 15,078,904**  
**Run type: non-record, 8xH100, 20.2 minute architecture demonstration**

This is a non-record submission for a folded recurrent transformer. It is not claiming the 10-minute leaderboard track: the logged cloud run used `MAX_WALLCLOCK_SECONDS=0`, trained for `1,211.13s`, and completed all `10,000` steps. The purpose of the submission is to document an architecture that trades stored parameters for repeated computation under the 16MB artifact cap.

## The Paper Analogy

The idea started from a simple paper analogy.

A conventional GPT is like writing a solution across a stack of separate sheets. Each transformer layer is another sheet of paper: more layers give more room to work, but every added sheet increases the amount of material you must store.

This model instead asks: what if we fold the same sheet and keep revisiting it? Folding does not create more paper, but it creates more surfaces, alignments, and stages of work. The same learned machinery can be applied again and again, while small fold-specific controls tell each pass how to behave.

That is the architectural bet here: under a hard file-size budget, depth should not have to mean owning a completely new set of weights at every stage. The model can own a compact block set, then use recurrence and fold-specific modulation to turn that small block set into a longer computation.

## How It Works

The model has three conceptual regions:

1. A visible GPT stream that embeds tokens and builds the first sequence representation.
2. A folded working space where the hidden state is projected into a smaller recurrent state.
3. Exit blocks that polish the final visible representation before the tied output head.

The folded working space is the main contribution. It uses:

- `5` recurrent folds
- `2` shared transformer blocks inside the fold stage
- `496`-dimensional folded state
- fold-specific update, refresh, preserve, attention-scale, and MLP-scale parameters
- fold step embeddings
- a curriculum that trains with fewer active folds early, then ramps to all folds

The important distinction from plain weight sharing is that each fold is allowed to behave differently. The attention and MLP weights are shared, but the fold has its own learned controls. Early folds can act like rough drafting passes; later folds can act like correction and polishing passes.

In block-application terms, the model executes approximately:

```text
1 visible stem block
+ 5 folds * 2 shared fold blocks
+ 4 exit blocks
= 15 transformer block applications
```

But it does not store 15 independent blocks. The central 10 block applications reuse only 2 shared fold blocks, with lightweight fold-specific modulation. That is how the model gets deeper behavior while keeping the compressed artifact at 13.7MB.

## Compared With A Conventional GPT

The public baseline GPT uses a normal decoder stack:

- `9` unique transformer layers
- `512` model dimension
- `1024` vocab
- tied embeddings
- no recurrence
- each layer owns its own attention and MLP weights

This folded recurrent model uses:

- `1` visible stem layer
- `5 x 2` recurrent folded block applications using only `2` shared fold blocks
- `4` exit blocks
- `576` visible dimension
- `8192` vocab
- tied embeddings
- explicit recurrent refinement before the output head

The conventional GPT path scales quality by adding distinct layers. This path scales quality by applying a smaller set of learned transformations multiple times, with per-fold controls to avoid every pass collapsing into the same operation.

The parameter-efficiency story is the strongest part of the submission. Against the public 9-layer GPT baseline, this run uses fewer parameters and a smaller artifact while scoring lower BPB. Against an unshared GPT-style version of the same 15 block-application computation, the gap is much larger: a conservative unshared equivalent would be about `34.18M` parameters, so this folded model stores only `44.1%` as many parameters, a `2.27x` reduction. A plain 15-layer, 576-wide GPT would be about `39.59M` parameters, making this model only `38.1%` as large.

| Model | Run Type | Params | Artifact Bytes | Train Time | Final Roundtrip BPB |
|---|---:|---:|---:|---:|---:|
| Naive GPT baseline | 10-minute record | 16,765,000 | 15,863,489 | 600.04s | 1.22436570 |
| 4-hour GPT baseline | non-record | not logged here | 15,810,161 | 14,400.04s | 1.20737944 |
| Estimated unshared same-compute GPT | reference estimate | ~34,177,864 | not run | not run | not run |
| Estimated plain 15-layer 576-wide GPT | reference estimate | ~39,589,752 | not run | not run | not run |
| Paper-Folded Recurrent GPT | non-record | 15,078,904 | 13,701,318 | 1,211.13s | **1.19084839** |

This is not a perfectly controlled ablation: the folded model also uses SP8192 rather than the SP1024 baseline tokenizer. The comparison is still useful because `val_bpb` is the challenge metric across tokenizers, and the result shows that a folded recurrent architecture can beat the original normal-GPT baseline family while using a smaller final artifact.

The tradeoff is compute. The naive GPT baseline ran at about `43.54 ms/step`; this model ran at `121.11 ms/step`. The recurrent folds spend roughly 2.8x more time per optimizer step, but the final roundtrip score is lower by `0.03351731 BPB` versus the public naive GPT baseline while using `10.1%` fewer parameters and a `13.6%` smaller packaged artifact. It is also lower by `0.01653105 BPB` versus the public 4-hour GPT baseline.

## Result

The cloud run was seed `1337`, 8xH100, no TTT:

```text
model_params:15078904
ttt_enabled:False
train_batch_tokens:524288
train_seq_len:1024
iterations:10000
max_wallclock_seconds:0.000
step:10000/10000 val_loss:3.0646 val_bpb:1.1864 train_time:1211130ms step_avg:121.11ms
Serialized model int8+zlib: 13636208 bytes
Total submission size int8+zlib: 13705120 bytes
final_int8_zlib_roundtrip_exact val_loss:3.07608506 val_bpb:1.19084839
```

The logged run reported `13,705,120` bytes using the then-current `train_gpt.py` code size. The submitted copy has comments trimmed below the 1,500-line limit and removes unused TTT helper code because this run does not use evaluation-time adaptation. Conservatively counting the submitted `train_gpt.py` gives `13,701,318` bytes, still well below the 16MB cap. The no-TTT point matters: this result is an architecture result, not an evaluation-time adaptation result.

## Why This Is Interesting

The model is slower per step than a normal GPT because the recurrent fold stage performs more block applications. That is the expected tradeoff. The interesting part is what the trade buys:

- More effective depth than the stored block count.
- A 15-block-application computation stored in 15.08M parameters instead of an estimated 34M-40M for unshared GPT-style equivalents.
- A smaller final artifact than the public normal GPT baselines.
- A reusable compute pattern that could be scaled by changing fold count, fold width, or exit depth.
- A clean separation between stored capacity and compute depth.
- Intermediate supervision through fold/mid/exit auxiliary losses, so the recurrent states learn to become useful before the final output.

This is a different answer to the Parameter Golf constraint. Instead of asking only "how many layers can fit under 16MB?", it asks "how many useful refinement passes can a compact set of weights perform?"

## Reproduction

The included run script reproduces the non-record cloud envelope:

```bash
chmod +x run_8xh100_nonrecord.sh
./run_8xh100_nonrecord.sh
```

It expects the SP8192 dataset and tokenizer at the standard repository paths:

```text
data/datasets/fineweb10B_sp8192
data/tokenizers/fineweb_8192_bpe.model
```

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed1337.log`
- `run_8xh100_nonrecord.sh`
