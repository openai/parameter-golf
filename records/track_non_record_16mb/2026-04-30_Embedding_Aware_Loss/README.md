# Question
Cross-entropy loss explicitly rewards the single ground-truth token per training sample. Can we improve training by also rewarding similar tokens (with respect to the embedding space)?

# Answer
**No**, at least not in our experiments. 

Nonetheless, we were able to show that a purely embedding-based loss correlates well with standard cross-entropy loss.

Below, we explain our experiments and methodology.

## Introduction

Standard language models are framed as next-token classifiers trained with cross-entropy, which does not explicitly reward “near misses”. A prediction like *large* instead of *big* may be penalized as harshly as nonsense. To address this potential issue, we bias the network towards a structure such that similar tokens in embedding space receive similar gradient updates via backpropagation. Mechanically, we add a term to the loss that measures the similarity between the ground-truth token’s embedding and the softmax-weighted predicted embedding.

### Method - Adding an "embedding loss" term

Let $z \in \mathbb{R}^{|V|}$ denote the model logits for a single next-token prediction, and let

$$
p = \mathrm{softmax}(z)
$$

be the predicted distribution over the vocabulary $V$.

Let $E \in \mathbb{R}^{|V| \times d}$ be the token embedding matrix, where each row corresponds to a token embedding in $\mathbb{R}^d$. If the ground-truth token is represented by a one-hot vector $y \in \mathbb{R}^{|V|}$, then its embedding is

$$
e_{\mathrm{gt}} = E^\top y.
$$

We define the model’s predicted embedding as the softmax-weighted average of token embeddings:

$$
\hat e = E^\top p.
$$

Our auxiliary embedding loss is then defined using cosine similarity:

$$
L_{\mathrm{embed}} = 1 - \frac{\hat e^\top e_{\mathrm{gt}}}{|\hat e| \cdot |e_{\mathrm{gt}}|}.
$$

The full training objective is

$$
L = L_{\mathrm{CE}} + \lambda L_{\mathrm{embed}},
$$

where $L_{\mathrm{CE}}$ is the usual next-token cross-entropy loss and $\lambda \ge 0$ controls the strength of the auxiliary term.

Intuitively, this loss encourages the model to place probability mass on tokens whose embeddings are close to the ground-truth token embedding, so that semantically similar predictions are penalized less harshly than unrelated ones.

Note that most of our experiments were run with untied embeddings, which in our experiments gave slightly better performance.

### Results

We hypothesized that this extra term might help improve the efficiency of training and lead to faster convergence. This was not the case:

<img width="487" height="612" alt="Screenshot 2026-04-30 at 2 15 39 PM" src="https://github.com/user-attachments/assets/6f8e7df0-f1ce-463e-8ad1-73f3435cec46" />

*Figure 1. Our extra loss term did not lead to quicker convergence.*

<img width="1335" height="581" alt="valb_chart_no_ws" src="https://github.com/user-attachments/assets/64c5a5c6-4e11-478c-b35a-816708a94835" />

*Figure 2. Our extra loss term leads to lower val_bpb after 10 minutes on 1X H100 SFX.*


### Next Experiment

To better understand the results, we wanted to see if our $L_{\mathrm{embed}}$ correlates at all with $L_{\mathrm{CE}}$. Intuitively, it should, since _big_ and _large_ tend to be close in embedding space and in their next-token prediction probabilities.

So, we ran an experiment where we **completely replace cross-entropy loss with our embedding loss**.

However, we see immediate collapse in this case; likely the model learned to assign the same embedding to every term in order to game the loss function.


<img width="1789" height="596" alt="collapse_plot" src="https://github.com/user-attachments/assets/0994a0ba-2900-4849-90b5-92d876892068" />


*Figure 3. Does replacing cross-entropy loss with our embedding-loss result still improve the cross-entropy loss? A little, but not much; and our embedding loss collapses to 0, giving little signal after the first few steps!*

Since the embedding loss collapses, we try to find a way to preserve its signal.

### Next Experiment

Now, we add **another loss term**: $L_{\mathrm{uniform}}$; to avoid collapse of the embedding-space, we pressure the model to keep its embeddings uniformly distributed. 

Formally, adapted from Wang & Isola (2020):

$$\mathcal{L}{\text{uniform}} = \log \frac{1}{V(V-1)} \sum{i \neq j} \exp!\left(-2 \left|\hat{e}_i - \hat{e}_j\right|^2\right)$$

And it helps! We don't get immediate collapse of the embedding space! The embedding-loss signal is preserved, and as a result, we see that **without a CE loss** we are still able to **decrease the CE loss**. Essentially, we've found a (weak) surrogate for the CE loss, at least in early training. However, it's only a _weak_ surrogate for CE loss, and as we can see below, directly training with CE loss still yields quicker convergence. 

<img width="2629" height="596" alt="eu_sweep_plot_goo" src="https://github.com/user-attachments/assets/5179009c-10f9-48c6-aa38-4f984d417cae" />

*Figure 4. We're able to significantly lower CE loss without using CE loss in our objective; instead our loss encourages predicted tokens to be **close to ground truth tokens in embedding space**, and we pressure the embedding space to be uniformly distributed to **avoid collapse**.*

Now, is it possible to mix CE loss back in to get a better result, maybe a better loss geometry?

### Next Experiment

Adding cross entropy loss back in, and running on 8 H100-SXM GPUs now, we see that our new loss achieves slightly better performance per steps, but is about 2x slower. Note that we are not using tied embeddings. 

<img width="1798" height="691" alt="8gpu_comparison" src="https://github.com/user-attachments/assets/9c986a06-601f-486d-bc53-82bad8f6c53f" />


*Figure 5. Our loss gives slightly better performance, compared to the baseline, comparing the same number of steps. However, it's 2x slower and performs worse after 10 minutes.*
 
However, the performance of the two models is very close, and it's difficult to say if these results are significant. To improve our confidence, we do three runs of each on 1 H100-SFX GPU.

<img width="2078" height="691" alt="1gpu_sweep_plot" src="https://github.com/user-attachments/assets/8e13f0f4-8bac-4455-9b74-3c549241c39f" />

*Figure 6. Despite avoiding collapse of the embedding-signal, the embedding-based loss term does not seem to improve training performance.*

 Indeed, it seems our results were not significant!

## Conclusions
We weren't able to beat the baseline by changing the loss. However, we were able to preserve the signal of our embedding-loss by biasing tokens to be uniformly distributed throughout the embedding space. And, we were able to demonstrate that a pure embedding-based loss correlates well with cross-entropy loss for next-token prediction.
