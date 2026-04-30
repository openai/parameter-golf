# Train-Time Overparameterization

## Overview

Language models are typically trained at the same capacity they will ultimately use at inference. In this work, we explore a different question:

Should a small model be trained small from the start?

We find that the answer appears to be no.

This work introduces Train-Time Overparameterization (TTO): a simple strategy in which the model is given extra MLP capacity during training, and that excess capacity is later consolidated back down to the original inference budget. The final model remains the same size at deployment, but training proceeds through a larger feature space.

The core idea is straightforward:

- start from a target small model  
- expand the MLP during training  
- learn which neurons are most useful via stochastic gating  
- consolidate back to the original width before export  

Empirically, this produces a consistent improvements over training the same final-capacity model directly.

### Training Schedule

TTO is applied over the course of training using a simple staged schedule:

- **0–5%**: train with the fully expanded MLP (no gating)  
- **5–25%**: introduce stochastic gating with a learned, budget-aware objective  
- **25–100%**: prune to the target width and continue training as a standard model  

After 25% of training, the model is functionally identical in size and structure to the baseline, and the remaining training proceeds normally.

## Results

All runs are trained for 10k steps under identical settings across three seeds.

To keep the final model within the same parameter budget, the baseline uses a 2× MLP expansion, while TTO uses a 4× temporary expansion (8× effective width during training), followed by consolidation back to the original size before export.

| Model | Seed | Pre-quant BPB ↓ | Post-quant BPB ↓ | Size (bytes) |
|-------|------|----------------:|-----------------:|-------------:|
| Baseline | 1337 | 1.2262 | 1.2328 | 15861272 |
| TTO (4× expand) | 1337 | **1.2197** | **1.2264** | 15891679 |
| Baseline | 42 | 1.2276 | 1.2343 | 15856563 |
| TTO (4× expand) | 42 | **1.2197** | **1.2270** | 15886567 |
| Baseline | 2025 | 1.2253 | 1.2321 | 15853892 |
| TTO (4× expand) | 2025 | **1.2204** | **1.2273** | 15883534 |
| **Average (Baseline)** | — | 1.2264 | 1.2331 | 15857242 |
| **Average (TTO)** | — | **1.2199** | **1.2269** | 15887260 |

Train-Time Overparameterization consistently improves over the baseline across all three seeds. 

---

## How Train-Time Overparameterization Works

### Why temporary expansion can help

The fundamental idea behind TTO is simple: a model may be fully capable of representing a good function, but still be bad at discovering it through gradient descent—training with a larger transient model both expands the space of available features and makes useful solutions easier to reach, before being compressed back down.

### Why the MLP is a natural place to apply it

The MLP is a natural place to apply TTO: it contains most of the model’s parameters and is relatively self-contained, making expansion and later pruning straightforward.

The value projection is a plausible alternative, but is much smaller, so we prioritized the MLP as the highest-leverage target (though value-TTO could still be effective).

Applying TTO to query/key projections is likely more challenging due to their tighter coupling and more global role, and may introduce more disruptive changes, though we have not explored this.

One important consideration is that TTO introduces distributional shift during training. As the active neuron budget is reduced, earlier layers change their outputs, perturbing the inputs seen by later layers. When applied more broadly, this effect may compound, suggesting that more structured (e.g. layer- or module-aware) scheduling could help isolate and control these shifts.

### The core challenge: selecting which neurons survive

Expanding the MLP is straightforward. The real challenge is deciding which neurons should remain when we return to the target model.

At a high level, TTO is built around a simple idea: enforce a **capacity budget** during training, and structure it so that the most useful neurons naturally emerge. Crucially, this budget must be applied *gradually*. Abruptly removing capacity would be highly disruptive, so instead we slowly reduce the number of active neurons over time, allowing the network to adapt as it is progressively “soft pruned” toward its final size.

A natural first approach is to assign each neuron a learned continuous gate, and use an auxiliary objective to constrain the total activation mass to match a target budget. This allows the model to softly reallocate capacity and gradually shift importance across neurons.

However, this approach runs into a fundamental issue: under a budget constraint alone, the model tends to distribute mass evenly across neurons, resulting in a “mushy” allocation where many neurons remain partially active. This is poorly aligned with the final top-k pruning step, where we need a clear separation between neurons that are used and those that are not.

To address this, we introduce a simple **separation objective** that encourages gates to move toward the extremes of 0 or 1. This pushes the network to make more decisive allocations, and in practice is critical for obtaining a clean ranking of neurons prior to pruning.

Even with this, however, the continuous formulation has a deeper problem: the network can *cheat*. Because the gates only scale activations, surrounding weights can simply increase in magnitude to compensate for reduced values. This allows the model to continue using its full effective capacity despite the imposed budget. As a result, when pruning is eventually applied, performance degrades sharply—the model was still relying on the neurons that are removed.

While it may be possible to address this with additional constraints (e.g. coupling gates to weight norms), this quickly becomes complex and difficult to tune. Instead, we take a different approach: we make the selection process explicitly **discrete**.

Each neuron is assigned a learned logit, which defines a Bernoulli probability via a sigmoid. During training, neurons are stochastically gated on or off using a hard binary mask. This eliminates the possibility of compensation through rescaling—when a neuron is off, it is truly unavailable.

This discrete formulation has two key advantages. First, it aligns training with the final model: neurons are either present or absent, just as they will be after consolidation. Second, it produces a meaningful importance signal: the learned probability directly reflects how valuable it is for a neuron to remain active.

To train these logits, we use a surrogate (straight-through) gradient estimator. Intuitively, this treats the hard gating operation as if it were a smooth scaling during backpropagation, allowing gradients to flow to the logits while preserving discrete behavior in the forward pass. This provides a stable signal for which neurons should be kept, while still enabling gradual adaptation as the budget is reduced.

By the time consolidation occurs, the network has already adapted to operate under the target budget, with most neurons effectively either fully on or fully off. As a result, the explicit pruning via top-k selection itself becomes minimally disruptive.

### Consolidation and training schedule

TTO operates in three phases: expansion, controlled sparsification, and consolidation.

We begin training with the fully expanded MLP, with all neurons active. This allows the model to take advantage of the larger feature space during the early stages of optimization.

At 5% of training, stochastic gating is enabled. From this point onward, neurons are sampled according to their learned probabilities, and the auxiliary objectives begin to take effect.

Between 5% and 15% of training, we **anneal the target budget** from the expanded width down to the final width. This gradually reduces the expected number of active neurons, allowing the model to adapt smoothly as capacity is removed.

At 25% of training, we **consolidate** the model by selecting the top-k neurons (by learned probability) and permanently pruning the rest. The MLP is then replaced with a standard dense layer of the target size, making the model architecturally identical to the baseline.

From this point onward, training proceeds normally with no additional overhead.

A key observation is that consolidation can occur relatively early. Although the soft pruning is initially disruptive, the model quickly recovers—typically within a small fraction of the remaining training steps—and maintains a consistent performance advantage thereafter.

This makes TTO practical: most of the training run is performed with the final, smaller model, while still benefiting from the improved optimization enabled by early overparameterization.

## Selection vs. optimization

TTO raises a central question: *where does the improvement actually come from?*

One possibility is **selection**. The expanded model exposes a much larger pool of neurons, and training identifies a particularly effective subset. From this perspective, overparameterization primarily acts as a search process, and the final performance is driven by the specific neurons that are retained after consolidation.

An alternative (and not mutually exclusive) explanation is **optimization**. A larger model may be easier to train: it can represent intermediate solutions more flexibly and make it easier for gradient descent to discover useful representations. In this view, even temporary overparameterization can guide the model into regions of parameter space that a smaller model would be unlikely to reach on its own, with some of these improvements persisting after consolidation.

To probe this, we ran a simple transfer experiment. We first trained a model with TTO and recorded the neurons selected during consolidation. We then trained a new model at the target size, initializing it with the same subset of neurons (i.e. same indices), but without using TTO during training.

This did **not** recover the performance of the full TTO model.

This suggests that the benefit is not solely explained by identifying a “lucky” subset of neurons at initialization, in contrast to interpretations similar to the lottery ticket hypothesis. While it is still possible that certain neurons become particularly important over the course of training, their usefulness does not appear to be fixed from the start in a way that can be trivially transferred.

Instead, this points toward a more optimization-driven explanation: the transient overparameterized phase may enable the model to form representations that are simply not accessible when training at the final capacity from the outset.

More broadly, it is unclear how localized this effect is. Although TTO is applied only to the MLP, changes in intermediate representations affect the entire network, and it is possible that attention layers also benefit indirectly from the improved feature space during early training.

These questions remain open, and we view TTO primarily as an empirical result that highlights the potential importance of training dynamics, rather than as a fully understood mechanism.

## Closing note

While it is easy to overstate the similarities between biological and artificial neural networks, TTO bears a rather striking structural resemblance to early brain development, where synaptic overgrowth is followed by large-scale pruning.

Perhaps this reflects something more fundamental: neural nets may simply learn better if you let them have a childhood.
