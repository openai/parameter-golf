# State-space models for parameter golf: a first-principles primer

**Bottom line up front.** State-space models (SSMs) are a beautiful, theoretically deep family of sequence models — and they are almost certainly the *wrong* primary bet for OpenAI's Parameter Golf record track at 16 MB / 10 min on FineWeb. The leaderboard is dominated by transformers with INT5/INT6 GPTQ + QAT, depth recurrence, and legal score-first TTT (current SOTA 1.0810 BPB, PR #1493). The two on-leaderboard SSM-family attempts — a Hymba-style parallel attn+SSM (1.1828) and an S4D-Lin hybrid (1.1682) — both sit roughly 0.06–0.10 BPB *behind* the contemporaneous transformer SOTA. Three structural facts cause this gap: (1) Zoology shows associative recall accounts for ~82% of the SSM↔attention perplexity gap, and FineWeb is recall-heavy; (2) the SSM recurrence amplifies quantization error multiplicatively, with no mature W4 recipe (Quamba2's W4A8 takes Hadamard rotations + per-state-group calibration that you cannot replicate in a 10-min budget); (3) Mamba's training has razor-sharp LR cliffs incompatible with no-sweep regimes. The README explicitly invites SSM submissions to the **non-record / wishlist track** — which is the right place to submit a Hymba-lite as a visibility play. The rest of this document is the rigorous theory you asked for, layered foundation→application, with the math you need to derive everything yourself.

------

## Layer 1 — Foundational theory

### 1.1 The continuous-time linear SSM

A linear time-invariant (LTI) state-space model is the coupled pair

$$\dot{x}(t) = A,x(t) + B,u(t), \qquad y(t) = C,x(t) + D,u(t),$$

with state $x(t)\in\mathbb{R}^N$, scalar input $u(t)\in\mathbb{R}$ (in deep SSMs), scalar output $y(t)\in\mathbb{R}$, state matrix $A\in\mathbb{R}^{N\times N}$, input map $B\in\mathbb{R}^{N\times 1}$, output map $C\in\mathbb{R}^{1\times N}$, feedthrough $D$ (almost always set to $0$ in deep SSMs and absorbed into a residual). This is the canonical object of Kalman-era control theory.

**Solving the ODE.** Variation of constants gives, with $x(0)=x_0$,

$$x(t) = e^{tA} x_0 + \int_0^t e^{(t-s)A} B, u(s),ds,$$

so with $D=0$, $x_0=0$, causal input,

$$y(t) = (K * u)(t),\qquad K(t) = C, e^{tA}, B.$$

The continuous SSM is **fully characterized by its impulse response** $K(t)$ — a one-dimensional convolution kernel.

**Frequency-domain view.** Laplace-transforming yields the transfer function

$$H(s) = C(sI - A)^{-1} B + D,$$

a $1\times 1$ rational function whose poles are the eigenvalues of $A$. Stability of the LTI system is exactly $\operatorname{Re}\lambda(A) < 0$.

**Why this matters for deep learning.** S4 and descendants treat each channel independently as a SISO ($p=q=1$) SSM with a learnable $(A,B,C,\Delta)$. The same parameters expose **two equivalent computational views** — a recurrence and a convolution — and that duality is the entire game.

### 1.2 Discretization (zero-order hold, bilinear, generalized α)

Real data are sampled with step $\Delta$; we need a discrete recurrence

$$x_k = \overline{A}, x_{k-1} + \overline{B}, u_k, \qquad y_k = C,x_k.$$

**Zero-order hold (ZOH).** Assume $u(t)\equiv u_k$ on $[k\Delta,(k+1)\Delta)$. Then exactly

$$x((k+1)\Delta) = e^{\Delta A}x(k\Delta) + \Bigl(\int_0^\Delta e^{\tau A}d\tau\Bigr),B,u_k,$$

and $\int_0^\Delta e^{\tau A}d\tau = A^{-1}(e^{\Delta A}-I)$. Multiplying inside/outside by $\Delta$ to match the S4D convention,

$$\boxed{;\overline{A} = \exp(\Delta A),\quad \overline{B} = (\Delta A)^{-1}\bigl(\exp(\Delta A) - I\bigr),\Delta B.;}$$

ZOH maps the open LHP exactly to the open unit disk, so it preserves stability.

**Bilinear / Tustin.** Trapezoidal-rule integration of the ODE (or the substitution $s\leftrightarrow \tfrac{2}{\Delta}\tfrac{1-z^{-1}}{1+z^{-1}}$ in $H(s)$) gives

$$\boxed{;\overline{A} = \bigl(I-\tfrac{\Delta}{2}A\bigr)^{-1}\bigl(I+\tfrac{\Delta}{2}A\bigr),\quad \overline{B} = \bigl(I-\tfrac{\Delta}{2}A\bigr)^{-1}\Delta B.;}$$

Bilinear is the conformal Möbius map from LHP to unit disk and is A-stable (with mild frequency warping near Nyquist). Importantly, $\overline{A}$ is a *rational* function of $A$, so structure (DPLR; §1.5) is preserved — this is why S4 chose bilinear.

**Generalized bilinear (α-discretization).** For $\alpha\in[0,1]$,

$$\overline{A}*\alpha = (I-\alpha\Delta A)^{-1}\bigl(I+(1-\alpha)\Delta A\bigr),\qquad \overline{B}*\alpha = (I-\alpha\Delta A)^{-1}\Delta B.$$

$\alpha=0$ is forward Euler, $\alpha=\tfrac12$ bilinear, $\alpha=1$ backward Euler. S4D uses ZOH; S4 uses bilinear; in practice the difference is small once $\Delta$ is learned.

### 1.3 Recurrence ↔ convolution duality (rigorous derivation)

Start from $x_{-1}=0$ and unroll:

$$x_k = \sum_{j=0}^{k} \overline{A}^{,j}\overline{B},u_{k-j};\Longrightarrow;y_k = \sum_{j=0}^{k}\bigl(C\overline{A}^{,j}\overline{B}\bigr)u_{k-j}.$$

Define the **SSM convolution kernel**

$$\boxed{;\overline{K}=(C\overline{B},,C\overline{A},\overline{B},,C\overline{A}^2\overline{B},,\ldots,,C\overline{A}^{L-1}\overline{B})\in\mathbb{R}^{L}.;}$$

Then the full output sequence is the causal 1-D convolution $y = \overline{K}*u$. With identical learned parameters $(A,B,C,\Delta)$ we get two computational regimes for free:

| Mode                                                     | Time             | Memory                   | Use                      |
| -------------------------------------------------------- | ---------------- | ------------------------ | ------------------------ |
| Recurrent: $x_k = \overline{A}x_{k-1} + \overline{B}u_k$ | $O(NL)$          | $O(N)$ — constant in $L$ | Autoregressive inference |
| Convolutional: $y = \overline{K}*u$ via FFT              | $O((N+L)\log L)$ | $O(N+L)$                 | Parallel training        |

The FFT step is $y = \mathcal{F}^{-1}(\mathcal{F}(\overline{K})\odot \mathcal{F}(u))$ in $O(L\log L)$ once $\overline{K}$ is materialized. Constructing $\overline{K}$ naively requires $L$ matrix powers $\overline{A}^k$, costing $O(N^2 L)$ — this is the LSSL bottleneck that S4 was designed to solve.

**Why the duality is the whole point.** SSMs **train like a CNN** (parallel convolution) and **infer like an RNN** (constant-state recurrence). Transformers can only do the former; classical RNNs only the latter; SSMs do both, while inheriting HiPPO-style theoretical guarantees about long-range memory.

### 1.4 HiPPO: optimal polynomial projections (arXiv:2008.07669)

HiPPO formalizes the question: at every $t$, store an $N$-dimensional vector $c(t)$ that is the best polynomial reconstruction of the *entire* past $u_{\le t}$, with online (ODE) update.

Pick a measure $\omega(t,\cdot)$ on $(-\infty,t]$ specifying which past times matter, and an orthogonal polynomial basis ${g_n(t,\cdot)}_{n=0}^{N-1}$ orthonormal w.r.t. $\omega$. The optimal coefficients

$$c_n(t) = \int_{-\infty}^t u(s),g_n(t,s),\omega(t,s),ds$$

satisfy a closed-form linear ODE (derivable from the three-term recurrences of orthogonal polynomials)

$$\frac{d}{dt}c(t) = A(t),c(t) + B(t),u(t).$$

Two key measures:

- **LegT (translated Legendre):** $\omega(t,s) = \tfrac{1}{\theta}\mathbf{1}_{[t-\theta,t]}(s)$ — sliding window of length $\theta$. Recovers Voelker's LMU.
- **LegS (scaled Legendre):** $\omega(t,s) = \tfrac{1}{t}\mathbf{1}_{[0,t]}(s)$ — uniform weight on entire growing past, **timescale-invariant**.

For LegS with the orthonormal scaled Legendre basis $g_n(t,s) = (2n+1)^{1/2} P_n(2s/t-1)$, applying Bonnet's recurrences for $P_n$ yields **the** HiPPO-LegS matrix used by S4:

$$\boxed{;A_{nk} = -\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2}, & n>k,\ n+1, & n=k,\ 0, & n<k,\end{cases}\qquad B_n = (2n+1)^{1/2}.;}$$

Lower-triangular, closed form. **Why HiPPO-LegS works:** it stores the optimal $N$-th order polynomial reconstruction of $u_{\le t}$ under a uniform-importance measure, so no past is forgotten exponentially fast; the gradient $\partial c(t)/\partial u(s)$ decays polynomially as $O(1/t)$, not exponentially, so vanishing-gradient is averted; the construction is timescale-equivariant by design. Empirically S4's ablation: HiPPO-init A → 87% on sCIFAR; random Gaussian A → ~60% with severe instability; permuted-MNIST jumps from 60% → 98.3% just from this initialization. Each orthogonal-polynomial family yields its own HiPPO matrix (LegT, LegS, LagT, FouT), giving a full taxonomy.

### 1.5 S4: structured state spaces (arXiv:2111.00396)

The naive bottleneck: materializing $\overline{K}$ takes $O(N^2 L)$, infeasible at $N=64,L=16{,}384$ (Path-X).

**Why naive diagonalization fails.** If $A=V\Lambda V^{-1}$, the kernel becomes a Vandermonde sum, computable fast. But **the HiPPO-LegS eigenvector matrix $V$ has entries $\sim 4^N$** (Lemma 3.2 of S4), so the diagonalizing change of basis is catastrophically ill-conditioned for $N\gtrsim 50$.

**The DPLR fix.** S4 proves (Theorem 1) that all HiPPO matrices admit a **Normal-Plus-Low-Rank (NPLR)** decomposition $A = V\Lambda V^* - PQ^\top$ with $V$ unitary, $\Lambda$ diagonal, rank $r=1$ or $2$. Conjugating by $V$ gives a **Diagonal-Plus-Low-Rank (DPLR)** form over $\mathbb{C}$:

$$A = \Lambda - p,q^*.$$

The unitary conjugation is well-conditioned, and DPLR is preserved under bilinear discretization.

**Computing the kernel via generating function + Cauchy + Woodbury.** Never form $\overline{K}$ in time; compute its truncated z-transform at the $L$-th roots of unity then inverse-FFT:

$$\hat K(z) = \sum_{k=0}^{L-1} C\overline{A}^k\overline{B},z^k = C,(I-\overline{A}^L z^L),(I-\overline{A}z)^{-1}\overline{B}.$$

The matrix power $\overline{A}^L$ is computed once; the only $z$-dependent piece is the inverse $(I-\overline{A}z)^{-1}$. For DPLR $\overline{A}$ this inverse is *also* DPLR (Woodbury identity preserves the structure), and after substituting through, the evaluation reduces to four **Cauchy-kernel** matrix–vector products

$$k_{z,\Lambda}(a,b) = \sum_{n=0}^{N-1}\frac{a_n b_n}{g(z) - \lambda_n}.$$

Each Cauchy product is $\widetilde O(N+L)$ via fast multipole methods (Pan). Result (S4 Theorem 3): **$\overline{K}$ in $\widetilde O(N+L)$ time and $O(N+L)$ memory.**

**LRA results.** S4 was the first model to crack Path-X (length 16,384) — every prior method was at chance. Average LRA: 80.5 vs ~53–55 for efficient transformers, a 25-point swing.

### 1.6 DSS / S4D: diagonal is enough (arXiv:2206.11893)

Gupta's empirical surprise (DSS, arXiv:2203.14343): drop the low-rank correction. Take just the diagonal $\Lambda$ from the NPLR decomposition of HiPPO-LegS — the resulting diagonal SSM matches S4 on LRA. S4D (Gu, Gupta, Goel, Ré) cleaned this up: combine S4's *computation* with DSS's *initialization* in the simplest possible way.

The S4D parameterization is $A=\operatorname{diag}(\lambda_1,\ldots,\lambda_N)\in\mathbb{C}^N$, so $\overline{A} = \operatorname{diag}(e^{\Delta\lambda_n})$ (ZOH) and the kernel is a Vandermonde sum

$$K_\ell = \sum_{n=1}^N C_n \overline{A}_n^{,\ell}\overline{B}_n,$$

literally two lines of code:

```python
log_K = (Delta * A).unsqueeze(-1) * torch.arange(L)
K = torch.einsum('n, nl -> l', C * B, torch.exp(log_K))
```

**Why diagonal works.** Algebraically, *almost every* matrix in $\mathbb{C}^{N\times N}$ is diagonalizable — non-diagonalizable matrices form a measure-zero algebraic variety. If $A=V\Lambda V^{-1}$, substituting $\tilde h = V^{-1}h$ absorbs $V$ into $B' = V^{-1}\overline{B}$ and $C' = CV$ at no expressivity cost. Need complex $\Lambda$ for oscillatory modes (real $A$ may have complex-conjugate eigenvalues); pair conjugates to keep outputs real, with $K_\ell = \sum_n 2\operatorname{Re}(c_n\lambda_n^\ell b_n)$. The catch is *initialization*: random $\Lambda$ → bad performance, hence S4D-Inv, S4D-Lin (e.g. $\lambda_n = -\tfrac12 + i\pi n$), all of which match HiPPO-LegS asymptotically as $N\to\infty$ (S4D Theorem 1). LRA averages: S4-LegS 86.0; S4D-LegS 85.6; S4D-Lin 85.5 — gap closed. **As of 2023+, diagonal SSMs are the standard.**

### Single best resource per topic in Layer 1

| Topic                         | Read this                                                    |
| ----------------------------- | ------------------------------------------------------------ |
| Continuous LTI SSM motivation | Hazy Research, "Simplifying S4" — https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4 |
| Discretization (ZOH/bilinear) | §3.1 of S4D, arXiv:2206.11893                                |
| Recurrence ↔ convolution      | The Annotated S4, https://srush.github.io/annotated-s4/      |
| HiPPO                         | Hazy Research blog post https://hazyresearch.stanford.edu/blog/2020-12-05-hippo, then §2 + Appendix D of arXiv:2008.07669 |
| S4 (DPLR/Cauchy/Woodbury)     | The Annotated S4 Part 2                                      |
| S4D / diagonal                | arXiv:2206.11893 §1–4                                        |
| End-to-end                    | Albert Gu's PhD thesis, https://purl.stanford.edu/mb976vf9362 |

Direct arXiv links: HiPPO https://arxiv.org/abs/2008.07669 · LSSL https://arxiv.org/abs/2110.13985 · S4 https://arxiv.org/abs/2111.00396 · DSS https://arxiv.org/abs/2203.14343 · SaShiMi https://arxiv.org/abs/2202.09729 · S4D https://arxiv.org/abs/2206.11893 · HTTYH https://arxiv.org/abs/2206.12037.

------

## Layer 2 — Modern selective and gated SSMs

### 2.1 Mamba (arXiv:2312.00752)

**The LTI bottleneck.** S4/S4D/H3/GSS all use time-invariant $(A,B,C,\Delta)$, so the I/O map is a fixed convolution $y = K*u$ with $K_\ell = C\overline{A}^\ell\overline{B}$. A fixed-kernel filter cannot make content-based decisions: at every position $t$ the *same* operator decides whether $u_t$ enters the state. Two synthetic tasks expose this — selective copying (variable-spacing copy) and induction heads (associative recall) — both requiring the model to compare current and past tokens.

**The selection mechanism (S6).** Make $(B,C,\Delta)$ functions of the current input:

$$\boxed{;\Delta_t = \operatorname{softplus}(\operatorname{Linear}*d(u_t)+\tau*\Delta),\quad B_t = \operatorname{Linear}_N(u_t),\quad C_t = \operatorname{Linear}_N(u_t).;}$$

$A$ stays a learned diagonal *parameter* (not input-dependent), entering only through the discretization $\overline{A}*t = \exp(\Delta_t A)$. softplus on $\Delta$ ensures positivity. With $A=-1$ this collapses to a classical gated update $h_t = (1-\sigma_t)h*{t-1} + \sigma_t \overline{B}_t u_t$ with $\sigma_t = 1-e^{-\Delta_t}$ — Mamba's selection is a smooth analogue of an RNN forget-gate.

**Why this breaks the convolutional view.** Now

$$y_t = \sum_{s\le t} C_t\Bigl(\prod_{r=s+1}^t \overline{A}_r\Bigr)\overline{B}_s,u_s,$$

so the local kernel at position $t$ depends on the *entire input* through cumulative products. There is no fixed $K$ to FFT-convolve; the model is time-varying, and the LTI/FFT algorithm is gone.

**Hardware-aware parallel scan.** Mamba recovers efficiency by scanning the recurrence on GPU with three ingredients: kernel fusion (everything from $\Delta,A,B,C$ through the discretization to the scan in one CUDA kernel; expanded $(\overline{A}_t,\overline{B}_t)$ never touch HBM, only SRAM); SRAM materialization of the state; and FlashAttention-style **recomputation** in the backward pass (don't store intermediate states; recompute from $u,\Delta,A,B$). Result: $O(LN)$ FLOPs, $O(L)$ memory, throughput SRAM-bandwidth-bound.

**The Mamba block** is a single homogeneous unit (no H3+MLP alternation): expand-projection branches into two paths; one runs Conv1d → SiLU → selective SSM, the other runs SiLU as a gate; they multiply, then a reduce-projection plus residual. ~$3 \cdot \text{expand}\cdot d^2$ params per block.

**Empirical headline.** 5× higher inference throughput than a same-size transformer; **Mamba-3B matches Transformers twice its size** on Pile pretraining loss and zero-shot evals; scales to 1M tokens in audio/genomics.

### 2.2 Mamba-2 / SSD (arXiv:2405.21060)

**The SSD restriction.** Mamba-2 restricts $A_t$ further to *scalar-times-identity*: $A_t = a_t I_N$ with $a_t\in\mathbb{R}$. The recurrence becomes

$$h_t = a_t h_{t-1} + B_t x_t,\qquad y_t = C_t^\top h_t.$$

Multi-head with head dim $P=64$–$128$ (matching modern attention); state dim $N$ jumps from 16 (Mamba-1) to 64–256.

**The duality theorem.** Define the lower-triangular causal mask $L\in\mathbb{R}^{T\times T}$ with $L_{ij} = a_i a_{i-1}\cdots a_{j+1}$ for $i\ge j$ (else 0), $L_{ii}=1$. Then $L$ is **1-semiseparable** (every below-diagonal submatrix has rank ≤ 1), and the SSD transformation can be written as

$$\boxed{;Y = M X,\qquad M = L\circ (CB^\top).;}$$

Setting $a_t\equiv 1$ recovers **causal linear attention** $Y=(L_{\text{causal}}\circ QK^\top)V$ under $(C,B,X)\mapsto(Q,K,V)$. The mask $L$ acts as an **input-dependent relative positional encoding** with discount factor $a_{j+1}\cdots a_i$. Same idea concurrently in GateLoop (arXiv:2311.01927).

**The block-decomposition algorithm.** Partition into chunks of size $Q\in[64,256]$. Diagonal blocks (intra-chunk) use the quadratic attention form (pure GEMM). Chunk-end states computed via matmul. Off-diagonal (inter-chunk) blocks are low-rank: pass chunk states sequentially over $T/Q$ chunks (the only non-parallel step). Output combination via matmul. ~30 lines of PyTorch, **2–8× faster than Mamba-1's selective scan** because most work is now matmul (tensor-core friendly: A100 BF16 matmul 312 TFLOPS vs 19 TFLOPS FP32 scalar).

**Tradeoff.** Mamba-1 has more independent dynamics per state for fixed $N$ (full diagonal $A$ vs scalar $a$); Mamba-2 enables much larger $N$ and $P$ for fixed wall-clock — and on GPUs the latter wins.

### 2.3 The broader linear-recurrence family

All members instantiate $S_t = G_t\odot S_{t-1} + k_t v_t^\top$, $o_t = q_t^\top S_t$, varying only the gate.

| Model                         | Recurrence gate                                              | Train algorithm                                    | Reference                                                    |
| ----------------------------- | ------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------ |
| S4 / S4D / GSS                | LTI diagonal $\overline{A}$                                  | FFT-conv $O(L\log L)$                              | arXiv:2111.00396, 2206.11893, 2206.13947                     |
| **H3** (Hungry Hungry Hippos) | LTI diag SSM + LTI shift SSM + multiplicative gate           | FFT-conv                                           | https://arxiv.org/abs/2212.14052                             |
| **Hyena**                     | implicit long conv (filter via small FFN of position) + data-controlled diag gating, $N$-fold | FFT-conv $O(NLd\log L)$                            | https://arxiv.org/abs/2302.10866                             |
| **RetNet**                    | scalar fixed $\gamma$ (data-independent); 1-semiseparable mask $D_{ij}=\gamma^{i-j}$ | parallel/recurrent/chunkwise — special case of SSD | https://arxiv.org/abs/2307.08621                             |
| **RWKV-4 → v6 (Eagle/Finch)** | scalar/matrix exponential decay (data-dep in v5+); time-mix + channel-mix sublayers | RNN parallel form                                  | https://arxiv.org/abs/2305.13048, https://arxiv.org/abs/2404.05892 |
| **Mamba-1**                   | data-dep diagonal $\overline{A}_t = \exp(\Delta_t A)$        | parallel scan, no matmul                           | arXiv:2312.00752                                             |
| **Mamba-2 (SSD)**             | data-dep scalar $a_t$                                        | block-decomp / matmul                              | arXiv:2405.21060                                             |
| **GLA**                       | data-dep diagonal $\alpha_t \in (0,1)^d$                     | FlashLinearAttention chunkwise                     | https://arxiv.org/abs/2312.06635                             |

H3 was the proximate motivation for Mamba (closed most of the SSM↔Transformer gap on language but still needed attention for full parity). Hyena's filters are *implicit* — produced by a small FFN $\gamma_\theta(t)$ as a function of position, so filter parameter count is constant but length is $L$. RetNet is provably a special case of SSD with constant scalar decay. RWKV's "two equivalent forms" (Transformer-like for training, RNN-like for inference) is the same parallel/recurrent pattern that motivates SSD.

### 2.4 The unifying picture

Attention and SSMs are **two algorithmic decompositions of the same family of structured-matrix transforms**. Selectivity = data-dependent gates ($\overline{A}_t$, $a_t$, $\alpha_t$). Choice between decompositions is a question of which structure best fits hardware (matmul vs scan) and data.

### Single best resource per topic in Layer 2

| Topic                           | Read this                                                    |
| ------------------------------- | ------------------------------------------------------------ |
| Mamba intuition                 | Maarten Grootendorst, "A Visual Guide to Mamba" — https://www.maartengrootendorst.com/blog/mamba/ |
| Mamba scan rigor                | James Chen, "Mamba No. 5" — https://jameschen.io/jekyll/update/2024/02/12/mamba.html |
| Mamba paper                     | arXiv:2312.00752 §3 + Appendix D                             |
| SSD theory                      | Goomba Lab Mamba-2 series Parts I–IV — https://goombalab.github.io/blog/2024/mamba2-part1-model/ |
| SSD algorithm                   | Princeton PLI, "Mamba-2 Algorithms and Systems" — https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems |
| Hyena                           | Hazy Research blog — https://hazyresearch.stanford.edu/blog/2023-03-07-hyena |
| Annotated Mamba (Triton kernel) | https://srush.github.io/annotated-mamba/hard.html            |

------

## Layer 3 — Theoretical rigor

### 3.1 The kernel formula and how to compute it

For an LTI SSM, unrolling from $h_0=0$:

$$\boxed{;K = (C\overline{B},,C\overline{A},\overline{B},,\ldots,,C\overline{A}^{L-1}\overline{B})\in\mathbb{R}^L,\qquad y = K*u.;}$$

Naive cost is $O(LN^3)$ dense or $O(LN^2)$ structured for the powers; numerically unstable when $\rho(\overline{A})\approx 1$ (which is exactly what long memory needs).

**Diagonal $A$.** $\overline{A}=\operatorname{diag}(\bar a_n)$, so $K_\ell = \sum_{n=1}^N c_n \bar a_n^{,\ell}\bar b_n$ — a Vandermonde matrix–vector product $K = V(c\odot\bar b)$ with $V_{\ell,n}=\bar a_n^\ell$. Direct cost $O(LN)$; fast Vandermonde gives $O((L+N)\log^2(L+N))$ (used by S4D).

**FFT convolution.** Once $K$ is built, $y = \mathcal{F}^{-1}(\mathcal{F}(K)\odot \mathcal{F}(u))$ in $O(L\log L)$ (zero-pad both to ≥ $2L$ to avoid circular wrap-around).

**DPLR (S4).** Use the generating-function trick $\hat K(z) = C(I-\overline{A}^L z^L)(I-\overline{A}z)^{-1}\overline{B}$; the inverse stays DPLR via Woodbury; evaluation at $L$-th roots of unity reduces to four Cauchy products at $\widetilde O(N+L)$ each; inverse-FFT to recover $K$.

**The Mamba case.** Selectivity makes $K$ depend on $u$ — no fixed kernel, no FFT path. Use a parallel scan instead.

### 3.2 Why diagonal SSMs are expressive

If $A=V\Lambda V^{-1}$ is diagonalizable, $K(t) = Ce^{tA}B = (CV)e^{t\Lambda}(V^{-1}B) = \tilde C e^{t\Lambda}\tilde B$. So **diagonal $\Lambda$ + arbitrary $\tilde B,\tilde C$ has the same expressive power as full $A$**. Diagonalizable matrices are dense in $\mathbb{C}^{N\times N}$ and have full Lebesgue measure (the non-diagonalizable set is the zero locus of the discriminant of the characteristic polynomial — measure zero). Need complex $\Lambda$ to capture oscillation; pair conjugates and write $\lambda_n = -e^{a_n}+ib_n$ to keep parameters real, outputs real, and stability built-in. Caveat: algebraic diagonalizability ≠ well-conditioning. The HiPPO-LegS $V$ has entries $\sim 4^N$, hence the obsession with HiPPO/Skew-HiPPO initializations in S4/S4D. Mamba sidesteps this via selectivity (different inductive bias) and real $A<0$ initialization.

### 3.3 The associative / parallel scan algorithm

The selective recurrence $h_t = a_t h_{t-1} + b_t$ becomes a prefix-scan under the binary operator

$$\boxed{;(a,b)\bullet(a',b') = (aa',,a'b + b').;}$$

**Associativity.** $$((a_1,b_1)\bullet(a_2,b_2))\bullet(a_3,b_3) = (a_1a_2a_3,,a_2a_3 b_1 + a_3 b_2 + b_3) = (a_1,b_1)\bullet((a_2,b_2)\bullet(a_3,b_3)).;\checkmark$$

The prefix product $(A_t,H_t) = (a_1,b_1)\bullet\cdots\bullet(a_t,b_t)$ has $H_t = h_t$. So computing the recurrence reduces to a prefix scan under $\bullet$.

**Hillis–Steele scan.** $O(L\log L)$ work, $O(\log L)$ depth — step-efficient.

**Blelloch scan (work-efficient).** Two phases, up-sweep (build a balanced reduction tree, halving active threads each level) and down-sweep (distribute prefixes back). Total $\sim 2L$ ops, depth $2\log_2 L$. **Asymptotically optimal** (reading all $L$ inputs is $\Omega(L)$).

In Mamba's `selective_scan_fwd_kernel.cuh`, a Blelloch-style scan executes per CUDA block over a chunk of the sequence; each scan element is the tuple $(a_t, b_t) = (\overline{A}_t, \overline{B}_t u_t)$, computed *inside* the kernel from $(\Delta_t, A, B_t, u_t)$ so the expanded tensors never hit HBM. Block boundaries are stitched with a second-level scan over chunk reductions; the backward recomputes states from saved $(u,\Delta,A,B,C)$ → $O(Ld)$ memory rather than $O(LdN)$. ML-RNN context: Martin & Cundy 2018 (arXiv:1709.04057) and S5 (arXiv:2208.04933).

### 3.4 Computational complexity table

Per layer, ignoring batch. $L$ = seq len, $d$ = model dim, $N$ = state dim per channel (or KV head dim), $C$ = chunk size in chunkwise schemes.

**Training (forward):**

| Model                      | Time                           | Memory                      | Tensor cores? |
| -------------------------- | ------------------------------ | --------------------------- | ------------- |
| Transformer (softmax attn) | $O(L^2 d)$                     | $O(Ld)$ with FlashAttention | ✓             |
| S4 / S4D / GSS             | $O(d(LN + L\log L))$           | $O(Ld+dN)$                  | partial       |
| H3 (FlashConv)             | $O(d L\log L)$                 | $O(Ld)$                     | partial       |
| Hyena (order $N$)          | $O(NdL\log L)$                 | $O(Ld)$                     | partial       |
| Linear attention (causal)  | $O(LdN)$                       | $O(Ld+dN)$                  | ✓             |
| RetNet parallel            | $O(L^2 d)$                     | $O(Ld)$                     | ✓             |
| RetNet chunkwise           | $O(Ld(C+N))$                   | $O(C^2+Ld)$                 | ✓             |
| GLA (chunkwise)            | $O(LdC + LdN)$                 | $O(Ld+dN)$                  | ✓             |
| Mamba-1 (selective scan)   | $O(LdN)$, SRAM-bandwidth-bound | $O(Ld)$                     | ✗             |
| Mamba-2 / SSD              | $O(LdN)$ FLOPs, mostly matmul  | $O(Ld+LN)$                  | ✓             |

**Inference (autoregressive, per generated token):**

| Model                             | Time/step           | "KV-cache" / state               |
| --------------------------------- | ------------------- | -------------------------------- |
| Transformer                       | $O(Ld)$             | $O(Ld)$ — **grows with context** |
| S4 / Mamba-1                      | $O(dN)$             | $O(dN)$ — constant in $L$        |
| Mamba-2 / SSD                     | $O(dN)$             | $O(dN)$ but $N$ is 4–16× larger  |
| Linear attn / RetNet / GLA / RWKV | $O(d^2)$ or $O(dN)$ | $O(d^2)$ or $O(dN)$              |

Headline: **SSMs have constant time and memory per generated token**. A 7B Mamba-1 with $N=16$, $d=4096$ holds 65,536 floats per layer per sample regardless of context length; a transformer at $L=10^5$ context needs ~$3\times 10^8$ floats per sample for its KV cache. Mamba-2's contribution is matmul-friendliness, not asymptotic improvement.

------

## Layer 4 — Application to Parameter Golf

### 4.1 The challenge, confirmed

Source: https://github.com/openai/parameter-golf and the live digest at Issue #140.

| Constraint | Value                                                      |
| ---------- | ---------------------------------------------------------- |
| Artifact   | ≤ 16,000,000 bytes (decimal MB), code + compressed weights |
| Training   | ≤ 10 min on 8×H100 SXM                                     |
| Eval       | ≤ 10 min                                                   |
| Metric     | Bits-per-byte on FineWeb validation (tokenizer-agnostic)   |
| Record bar | Beat SOTA by ≥ 0.005 nats at $p<0.01$ (≈3 seeds)           |
| Deadline   | 30 Apr 2026                                                |

**Current top 5 (≈9 Apr 2026):** all transformers using SP8192, parallel residuals, depth recurrence, QK-Gain, GPTQ/QAT INT5–6, legal score-first TTT. SOTA is **1.0810 BPB** (PR #1493, "bigbag"). Naive baseline 1.2244.

**SSMs in the README RFC list:** Yes — *"State-space models, E2E TTT, super long context for evaluation or training"* is explicitly listed as a wishlist item. So an SSM submission is **welcomed as a non-record contribution** — but the org has not signalled it as likely SOTA.

**SSMs actually attempted on the leaderboard:**

- **Hymba-style parallel attn+SSM:** 7L parallel attn+SSM with learned mixing — **1.1828 BPB** (3 seeds), labelled "first competitive non-transformer," ~0.10 BPB behind contemporary transformer SOTA.
- **S4D-Lin hybrid (2 SSM + 9 Transformer):** **1.1682 BPB**. Issue #140 verdict: *"attention > SSM at this scale."*
- **PR #1227 SSM hybrid:** improved CE 18% at $d=192$ but worsened BPB 2.7% at $d=512$ — Issue #140 calls this **"scale deception"** (the canonical SSM failure mode for Parameter Golf).

### 4.2 Parameter efficiency at ~10–50M params

Mamba's own scaling-law claim: Mamba-3B matches Transformer++ of 2× its size on Pile perplexity, but plots span 125M–1.4B; **<100M is not directly validated**. NVIDIA's 8B controlled study (Waleffe et al., arXiv:2406.07887) shows pure Mamba/Mamba-2 match or slightly beat Transformer++ on perplexity but **lag on copying, in-context learning, MMLU, phonebook lookup, and long-context reasoning** — gaps that hybrids close.

Two specific hazards at small scale:

1. **Sharp LR cliffs.** "Revisiting Associative Recall" (arXiv:2508.19029) shows Mamba/Hyena have very narrow optimal LR bands; transformers tolerate much wider ones. Incompatible with a 10-min, no-sweep budget.
2. **Mamba's fp16/bf16 instability.** state-spaces/mamba README explicitly warns *"SSMs are sensitive to their recurrent dynamics… use fp32 for parameters."* That's a memory and bandwidth tax in a 16-MB regime.

Where SSMs win at small scale (none of which matter on FineWeb): very long sequences (≫16k), genomics/audio, copy-free state tracking. Parameter Golf evaluates at seq 2048–8192 on natural English — exactly where transformers are strongest.

### 4.3 Fixed-state compression

Each Mamba layer holds a state of shape $(B, \text{expand}\cdot d_{\text{model}}, d_{\text{state}})$. With Parameter-Golf-typical $d=512$, $d_{\text{state}}=16$, expand=2, that's $512\cdot 2\cdot 16 = 16{,}384$ floats per layer — fixed in $L$. Useful for very long context, but at small scale the bottleneck is real: Zoology shows ~82% of the perplexity gap to attention on natural-language data is *associative recall*, and that gap appears even at <2k context. There is no free lunch better than a transformer's dynamic KV cache.

### 4.4 Quantization-friendliness — the decisive issue

Top Parameter Golf submissions burn 0.05–0.07 BPB of headroom on INT5/INT6 GPTQ + QAT.

Best sources: **Quamba** (arXiv:2410.13229) and **Quamba2** (arXiv:2503.22879); also **MambaQuant** (arXiv:2501.13484). Headline numbers on Mamba-2.8B / Vim:

| Method                                                       | W/A bits | Result                                    |
| ------------------------------------------------------------ | -------- | ----------------------------------------- |
| Naive RTN W8A8                                               | 8/8      | 21% accuracy drop on Vim-T (catastrophic) |
| SmoothQuant W8A8                                             | 8/8      | Significant drop                          |
| Quamba W8A8 (custom Hadamard + per-tensor SSM)               | 8/8      | 0.9% drop                                 |
| MambaQuant W8A8 (KLT rotation + SSLU)                        | 8/8      | <1% drop                                  |
| Quamba2 W4A8 (Hadamard + per-state-group + offline rearrangement) | 4/8      | 1.6% drop                                 |
| QuaRot W4A4                                                  | 4/4      | Catastrophic                              |

Why SSMs are quantization-hostile:

1. **Selective scan has hyper-sensitive activations**, with massive output outliers absent in attention (Quamba authors).
2. **Recurrence amplifies quantization error multiplicatively over time.** MambaQuant: *"the parallel scan further amplifies these outliers, leading to uneven and heavy-tailed data distributions."* Issue #140 PR #363 documented depth recurrence (a much milder analog) amplifying quant error ~900× over 3 cycles. For a 100-step recurrence at W4, this is fatal.
3. **Attention's softmax is bounded** $[0,1]$, so quant error stays bounded. The SSM recurrence has no such bound.
4. **W4 needs sophisticated machinery** (Hadamard rotations, per-state-group calibration, offline reordering) — none reproducible in 10 minutes plus QAT.
5. **No published Mamba QAT exists at this scale.** All published methods are PTQ. The competition's INT5/INT6 results lean on Muon-based QAT, GPTQ-lite, Late-QAT — no recipe-book equivalent for SSM blocks.
6. **fp32 master parameters for the recurrence** (Mamba README) means paying bits twice.

Verdict: an SSM that even half-recovers the leaderboard's quant gains would need Quamba2-grade infrastructure or aggressive INT8 (≈25% larger artifact, smaller model). Either path leaves you well behind.

### 4.5 Known weaknesses at small scale

| Paper                                                        | Claim                                                        | Severity for Parameter Golf                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Repeat After Me** — Jelassi et al., https://arxiv.org/abs/2402.01032 | Theorem: any fixed-state GSSM cannot copy long random strings; transformers can copy strings exponential in hidden dim. Empirically, pretrained Mamba severely underperforms transformer on copy/retrieve tasks. | Moderate. FineWeb has heavy in-context repetition (URLs, names, code, numbers); sliding-window eval at stride 64 turns this into a live disadvantage. |
| **Zoology / MQAR** — Arora et al., https://arxiv.org/abs/2312.04927 | A 70M-param attention model beats a 1.4B gated-conv on associative recall. **82% of the perplexity gap on Pile is associative recall.** | **High — the killer.** Losing 82% of the perplexity gap to recall at 30M params is exactly the BPB Parameter Golf measures. |
| **The Illusion of State** — Merrill, Petty, Sabharwal, https://arxiv.org/abs/2404.08819 | S4 and Mamba live in TC⁰; cannot solve permutation composition / true state tracking — same expressivity ceiling as transformers. | Low for BPB; but kills the theoretical motivation for picking SSM over transformer. |
| **Revisiting Associative Recall** — https://arxiv.org/abs/2508.19029 | Some of the SSM recall gap is fixable with very careful LR tuning, but tuning is *much* more sensitive than transformers (sharp LR cliffs). | High in a 10-min budget — you cannot afford 20 LR sweeps.    |

Quantitative summary: at 30M params on FineWeb-style data, expect a pure SSM to leave **~0.03–0.07 BPB on the table** vs an equally-quantized transformer of equal param count, before any recall-specific tricks (BigramHash, SmearGate). The on-leaderboard Hymba's 1.1828 vs contemporary transformer 1.1228 SOTA (≈0.06 BPB gap) is consistent with this prediction.

### 4.6 Hybrids: Jamba, Zamba, Samba, Hymba

| Hybrid                       | Citation                                     | Attn:SSM mix                                                 | Scale tested   | Key finding                                                  |
| ---------------------------- | -------------------------------------------- | ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ |
| **Jamba**                    | https://arxiv.org/abs/2403.19887             | ~1:7 attention to Mamba (≈12.5%) + MoE                       | 52B/12B active | First production-grade Mamba hybrid; strong recall           |
| **Zamba / Zamba2**           | https://arxiv.org/abs/2405.16712             | Mamba backbone + 1–2 shared global attention every 6 layers (≈12–17%) | 7B / 2.7B      | "Best non-transformer model" at 7B                           |
| **Samba**                    | https://arxiv.org/abs/2406.07522 (ICLR 2025) | Mamba + SwiGLU + sliding window attn + SwiGLU stacked (≈25% attn via SWA) | 421M – 3.8B    | Beats Phi-3-mini at 3.8B; perfect 256k retrieval from 4k-trained model |
| **Hymba**                    | https://arxiv.org/abs/2411.13676             | **Parallel** attn+SSM heads in *every* layer (1:2.12 attn:SSM by params) + 3 full-attn (rest SWA) + meta tokens + KV-cache sharing | 125M–1.5B      | **Strongest small-scale hybrid (<200M).** Beats SmolLM-135M, MobileLM, Pythia-160M on commonsense+recall |
| **NVIDIA 8B Mamba-2-Hybrid** | https://arxiv.org/abs/2406.07887             | 43% Mamba-2, 7% self-attention, 50% MLP                      | 8B             | Beats pure Transformer on all 12 standard tasks (+2.65 avg)  |

**Consensus optimal attn:SSM ratio at small/medium scale: ~10–25% attention layers.** Hymba is the most relevant data point — the only hybrid actually tested **<200M** with full ablations. Yet the on-leaderboard Hymba attempt landed at 1.1828, ~0.06 BPB behind contemporary transformer SOTA. **Hybrids in Parameter Golf are interesting but not winning.**

### 4.7 Recommended verdict and concrete strategy

**Pure SSM: NO. Pure transformer (current meta): STRONG YES. Hybrid (Hymba/Samba-lite): MAYBE as a secondary swing for the non-record track.**

The strongest expected-value move is to fork the current SOTA stack (e.g. PR #1493 / #1394: SP8192 + depth recurrence + parallel residuals + GPTQ + legal TTT + Muon) and innovate within it. SSM-first burns the entire bag of QAT/GPTQ/SmearGate/BigramHash gains the meta has accumulated and asks you to rebuild equivalent SSM-specific machinery from scratch in <2 weeks.

**If you want to also submit a Hymba-lite for visibility (the README RFC explicitly invites this):**

Architecture: 7–9 layers, $d=512$, parallel attn+SSM heads in every layer.

- Attention branch: 4 heads, head_dim=64, sliding window 512, with one full-attention layer mid-stack for global recall. Rotary partial RoPE (16/64), QK-norm, GQA.
- SSM branch: Mamba-2-style, $d_{\text{state}}=64$ (not 16 — recall bottleneck is real), expand=1.5 (not 2 — save params), $A$ and $\Delta$ in fp32, the rest bf16 → INT6.
- Mix ratio attn:SSM heads ≈ 1:2 (per Hymba ablation, saturates at 1:2.12).
- 16 prepended learnable meta tokens (~8k params; fixes attention sink).
- MLP: 3× SwiGLU (or Star-ReLU/relu² which dominate the leaderboard).
- Tied embedding fp16; BigramHash(4096) bias to recover associative recall cheaply.

Quantization plan:

- INT8 on attention input/output projections only (bounded outputs, safe).
- INT6 on MLP and embeddings (proven safe in the meta).
- **Keep SSM A, B, C, Δ in bf16** — quantizing them risks 900×-style error amplification.
- Hadamard transform on SSM input/output (Quamba's trick) before any INT8 attempt.
- Skip QAT on SSM blocks — no published recipe, no time. Use Quamba-style per-tensor static PTQ + Hadamard.

Training: Muon for matrix params, AdamW for SSM-specific params (`A_log`, `D`, `dt_bias`); fp32 master params on the recurrence; 3-point LR sweep at ${0.005, 0.01, 0.02}$ before the final run because Mamba's LR cliff is sharp; sequence length 2048; sliding-window eval stride 64; legal score-first TTT on top.

**Realistic BPB target ≈ 1.13–1.18**; you will probably *not* beat 1.0810 SOTA but you may match Hymba's 1.1828 or slightly beat it. Submit the transformer entry to the **record track** as primary; submit the Hymba-lite to the **non-record / wishlist track**. Both can be linked from the participant form for OpenAI hiring visibility.

Risks: no published Mamba QAT; Mamba kernels are slow at $d=512$ (tuned for $d\ge 1024$); scale deception; fp32-recurrence overhead; Hymba's parallel heads need custom kernels (FlexAttention + selective_scan_cuda) — implementation risk in 2 weeks is real; the leaderboard moves ~0.005 BPB/day.

**What to do tomorrow.** (1) Fork PR #1394 today as your transformer baseline; (2) reproduce + ablate over 2 days, confirm 1.085–1.09 on your hardware; (3) spend 3–4 days on a Hymba-lite branch that replaces 1 of 11 transformer layers with one SSM layer first, measure BPB delta; (4) if positive, scale to 2–3 SSM layers parallel-fused; if negative, drop SSM and pour time back into TTT/QAT/depth-recurrence; (5) always submit the transformer to the record track.

------

## Layer 5 — Implementation resources

### 5.1 The Annotated S4 (Sasha Rush & Sidd Karamcheti)

- URL: https://srush.github.io/annotated-s4/ (code: https://github.com/srush/annotated-s4; mirror https://iclr-blog-track.github.io/2022/03/25/annotated-s4/)
- **Stack:** JAX + Flax. ~3–5 hours to read; full reproduction (sCIFAR/sMNIST/QuickDraw) takes a day. There is a follow-up Annotated S4D at https://srush.github.io/annotated-s4/s4d.
- **Coverage:** classical SSM (continuous, discretization) → recurrent/conv views → HiPPO matrix → S4 (DPLR/NPLR, Cauchy kernel, Woodbury, generating function, FFT) → training scripts.
- **Strength:** the single best resource for building S4 from scratch. Math and code interleaved line-by-line with the original paper. Slides: https://srush.github.io/annotated-s4/slides.html.

### 5.2 Official `state-spaces` org

Two public repos (both Apache-2.0):

- **state-spaces/s4** — https://github.com/state-spaces/s4 — PyTorch + Hydra; S4, S4D, HiPPO, LSSL, SaShiMi, DSS, HTTYH, S4ND. Maintenance mode (last update mid-2024).
- **state-spaces/mamba** — https://github.com/state-spaces/mamba — Mamba-1, Mamba-2, Mamba-3. Active. Key files:
  - `mamba_ssm/modules/mamba_simple.py` — Mamba-1 block, cleanest public reference.
  - `mamba_ssm/modules/mamba2.py`, `mamba2_simple.py` — Mamba-2 block.
  - `mamba_ssm/modules/ssd_minimal.py` — minimal SSD module (Listing 1 of the Mamba-2 paper).
  - `mamba_ssm/ops/selective_scan_interface.py` — calls into the CUDA kernel; contains a Python `selective_scan_ref` for CPU debugging.
  - `mamba_ssm/models/mixer_seq_simple.py` — full LM head model.
  - `csrc/selective_scan/` — C++/CUDA kernels.
- **Install:** `pip install mamba-ssm --no-build-isolation`; needs Linux + NVIDIA GPU + PyTorch ≥1.12 + CUDA ≥11.6; expect environment friction.
- HF checkpoints: `state-spaces/mamba-{130m,370m,790m,1.4b,2.8b}`, `mamba2-2.7b`.

### 5.3 Minimal / educational implementations

| Repo                                 | URL                                                          | Stack                                    | What it is                                                   |
| ------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| **mamba-minimal** (johnma2006)       | https://github.com/johnma2006/mamba-minimal                  | Pure PyTorch, single ~300 LOC `model.py` | Numerically equivalent to official Mamba forward/backward, sequential `selective_scan`. Cross-references Annotated S4. **Strongest pedagogical Mamba implementation.** |
| **mamba2-minimal** (tommyip)         | https://github.com/tommyip/mamba2-minimal                    | Pure PyTorch, single file                | Mamba-2 / SSD analogue.                                      |
| **mamba.py / MambaPy** (alxndrTL)    | https://github.com/alxndrTL/mamba.py                         | PyTorch + MLX                            | Adds an associative-scan parallel implementation that trains fast on GPU/Mac. Includes Jamba and Vision Mamba. Upstreamed into 🤗 transformers. **Best if you want to train, not just read.** |
| **mamba-minimal-jax** (radarFudan)   | https://github.com/radarFudan/mamba-minimal-jax              | JAX                                      | JAX port; consistent with the Annotated S4.                  |
| **The Annotated Mamba** (Sasha Rush) | https://srush.github.io/annotated-mamba/hard.html (repo https://github.com/srush/annotated-mamba) | Triton + PyTorch                         | Re-implements the S6 selective-scan kernel in Triton line-by-line. **Best for the hardware-aware story.** |
| **mamba.c** (kroggen)                | search awesome-mamba                                         | Pure C/CUDA                              | Inference-only à la `llama2.c`.                              |
| **mamba-in-depth** (deepbiolab)      | https://github.com/deepbiolab/mamba-in-depth                 | PyTorch                                  | Self-paced course wrapping S4 → Mamba on MNIST.              |

There is no repo named "minS4"; the Annotated S4 is the de-facto minimal S4. For pure-PyTorch S4 see `mamba-in-depth` or community ports linked from https://github.com/radarFudan/Awesome-state-space-models.

### 5.4 Albert Gu's PhD thesis

- **Modeling Sequences with Structured State Spaces** (Stanford 2023). Stanford SearchWorks https://searchworks.stanford.edu/view/14784021 · PURL https://purl.stanford.edu/mb976vf9362 · PDF https://stacks.stanford.edu/file/druid:mb976vf9362/gu_dissertation-augmented.pdf.
- Advisor Christopher Ré; ~250 pages; Part I deep SSMs and LSSL; Part II structured state spaces (diagonal/NPLR/DPLR, S4, S4D, S5, Cauchy kernel); Part III HiPPO and "How to Train Your HiPPO" (arXiv:2206.12037). Mamba was concurrent and is not in the thesis, but every theoretical building block is. **Single best resource for theory.** ~15–25 hours to study carefully.

### 5.5 Talks and lectures

- **Albert Gu — Stanford MLSys Seminar #46:** https://www.youtube.com/watch?v=EvQ3ncuriCM (canonical S4 talk; clips https://www.youtube.com/watch?v=xiHmn6-xgiw, https://www.youtube.com/watch?v=ugaT1uU89TA).
- **MedAI Seminar #41 — Albert Gu** (longer-form; search YouTube).
- **"Structured State Space Models for Deep Sequence Modeling"** — https://www.youtube.com/watch?v=OpJMn8T7Z34 (May 2023 overview).
- **Cognitive Revolution / Nathan Labenz interview** — https://www.youtube.com/watch?v=1zjMalKLHiA (motivation/aesthetics).
- **The AI Epiphany w/ Albert Gu & Karan Goel** — https://www.youtube.com/watch?v=iUfUFKQLGBQ.
- **Tri Dao** — talks linked from https://tridao.me/blog/2024/mamba2-part1-model/ ; ICML 2024 SSD lecture.
- **Sasha Rush** — JAX/Annotated-S4 talk slides https://srush.github.io/annotated-s4/slides.html ; MLSys 2023 keynote "Do We Need Attention?".
- **Stanford CS25 (Transformers United)** — course https://web.stanford.edu/class/cs25/ ; recordings https://web.stanford.edu/class/cs25/recordings/ ; YouTube playlist https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM ; SSM/Mamba episodes in V4 (Spring 2024) and V5/V6.
- **MIT 6.S191** — has an RNN/Transformer/Attention lecture; **no dedicated SSM/Mamba lecture exists** as of this writing.
- **Yannic Kilcher — Mamba (Paper Explained)** https://www.youtube.com/watch?v=9dSkvxS2EB0 (~41 min, best high-level intuition video).
- **Umar Jamil — Mamba and S4 Explained** (~2 h, very thorough).
- **Gabriel Mongaras** https://www.youtube.com/watch?v=866SfiCHZ4o; **Samuel Albanie** https://www.youtube.com/watch?v=ouF-H35atOY; **AI Coffee Break w/ Letitia** (short intuitive intro).

### 5.6 Surveys (2024–2025)

- **From S4 to Mamba: A Comprehensive Survey on Structured State Space Models** (Mar 2025) — https://arxiv.org/abs/2503.18970. **Best single 2025 survey for architecture-level understanding.**
- **State Space Model for New-Generation Network Alternative to Transformers: A Survey** (Apr 2024) — https://arxiv.org/abs/2404.09516.
- **Mamba-360: Survey of State Space Models as Transformer Alternative** — https://arxiv.org/html/2404.16112v1.
- **A Survey of Mamba** (Aug 2024) — https://arxiv.org/html/2408.01129v1 (focuses specifically on Mamba and SSD).
- **Computation-Efficient Era: SSMs in Medical Image Analysis** — https://arxiv.org/abs/2406.03430 (reference Vision-Mamba survey).
- **Curated paper lists:**
  - https://github.com/radarFudan/Awesome-state-space-models
  - https://github.com/XiudingCai/Awesome-Mamba-Collection
  - https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List
  - https://github.com/gauravfs-14/awesome-mamba

### 5.7 Quality blogs and Hazy Research

- **Maarten Grootendorst — A Visual Guide to Mamba and SSMs** https://www.maartengrootendorst.com/blog/mamba/ — **single best resource for intuition** (50+ custom diagrams).
- **Sebastian Raschka — Beyond Standard LLMs** https://magazine.sebastianraschka.com/p/beyond-standard-llms (covers SSM/Transformer hybrids, including Qwen3-Next, Kimi Linear, Granite 4.0, Nemotron Nano 2). His 2024 list https://sebastianraschka.com/blog/2024/llm-research-papers-the-2024-list.html.
- **HuggingFace blog — Introduction to State Space Models** by lbourdois https://huggingface.co/blog/lbourdois/get-on-the-ssm-train.
- **Sascha Kirch — Here Comes Mamba** (Medium, Part 3 of a multi-part series).
- **Tri Dao — https://tridao.me/** — Mamba-2 blog series Parts I–IV (https://tridao.me/blog/2024/mamba2-part1-model/), Mamba-3 series (https://tridao.me/blog/2026/mamba3-part1/, Part 2 https://tridao.me/blog/2026/mamba3-part2/), FlashAttention v1/v2/v3.
- **Albert Gu / Goomba Lab — https://goombalab.github.io/blog/** — "On the Tradeoffs of SSMs and Transformers" (2025) https://goombalab.github.io/blog/2025/tradeoffs/; Mamba-2 Parts I–IV cross-posts; Mamba-3 Part 1 https://goombalab.github.io/blog/2026/mamba3-part1/; Hydra (bidirectional SSMs) https://goombalab.github.io/blog/2024/hydra-part1-matrix-mixer/ (code https://github.com/goombalab/hydra). Princeton PLI Mamba-3 cross-post https://pli.princeton.edu/blog/2026/mamba-3-improved-sequence-modeling-using-state-space-principles.
- **Hazy Research — https://hazyresearch.stanford.edu/blog** — From Deep to Long Learning (https://hazyresearch.stanford.edu/blog/2023-03-27-long-learning); H3 (https://hazyresearch.stanford.edu/blog/2023-01-20-h3); Hyena (https://hazyresearch.stanford.edu/blog/2023-03-07-hyena), Hyena Safari, HyenaDNA; Zoology Part 0 (https://hazyresearch.stanford.edu/blog/2023-12-11-zoology0-intro), Part 1 (https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis); Based (https://hazyresearch.stanford.edu/blog/2024-03-03-based); ThunderKittens kernels DSL (https://hazyresearch.stanford.edu/blog/2024-05-12-tk).

### 5.8 "Single best resource" picks per goal

| Goal                  | Pick                                                         |
| --------------------- | ------------------------------------------------------------ |
| Building from scratch | **The Annotated S4** + **mamba-minimal** as the Mamba follow-on. |
| Theory                | **Albert Gu's PhD thesis** https://purl.stanford.edu/mb976vf9362 |
| Intuition             | **Maarten Grootendorst's Visual Guide** https://www.maartengrootendorst.com/blog/mamba/ |
| 2025 survey           | **From S4 to Mamba** https://arxiv.org/abs/2503.18970        |

### 5.9 Recommended study path

1. **Intuition (1 h):** Maarten Grootendorst's visual guide + Yannic Kilcher Mamba video.
2. **S4 from scratch (1 weekend):** Work through the Annotated S4, then derive HiPPO by hand using thesis Ch. 3 + "How to Train Your HiPPO" (arXiv:2206.12037).
3. **Mamba from scratch (1 day):** Read `mamba-minimal/model.py` end-to-end, run the demo, diff against `state-spaces/mamba/.../mamba_simple.py`.
4. **Mamba-2 / SSD (1 day):** Tri Dao + Gu Mamba-2 Parts I–IV blogs, then `ssd_minimal.py` and `tommyip/mamba2-minimal`.
5. **Hardware-aware kernel (1 day, optional):** Sasha Rush's Annotated Mamba (Triton).
6. **Survey + frontier (2 h):** "From S4 to Mamba" (2025), Gu's "Tradeoffs" essay, Mamba-3 Parts 1–2 to see where the field is going.

### 5.10 Caveats

- `state-spaces/mamba` has a hard CUDA dependency for the fast path; pin CUDA + PyTorch versions exactly to a release wheel. Use `selective_scan_ref` (Python) for CPU debugging.
- No dedicated MIT 6.S191 SSM lecture exists yet.
- Mamba-3 is bleeding-edge; treat the blogs as the most authoritative writeup until the paper stabilizes.
- Multiple "Mamba in 100 lines" gists exist but lack the rigor of `mamba-minimal`; skip them.

------

## Conclusion: take the rigor, skip the gamble

Three things are simultaneously true. First, SSMs are intellectually beautiful — the unification of continuous-time linear systems, polynomial projections, structured matrix algorithms, and parallel scans is one of the cleanest arcs in modern deep learning. The recurrence-convolution duality, HiPPO's polynomial-memory derivation, S4's Cauchy/Woodbury kernel computation, and Mamba-2's structured semiseparable masks repay the time spent. Second, at the **competition's regime** (≤30M params, FineWeb English, 16 MB compressed, 10 minutes on 8×H100), the empirical evidence is unanimous: Zoology shows ~82% of the SSM↔attention perplexity gap is associative recall; SSM quantization is research-grade where attention's is industrial; Mamba's LR cliffs are incompatible with no-sweep training; and the only on-leaderboard SSM/hybrid attempts sit 0.06–0.10 BPB *behind* contemporaneous transformer SOTA. Third, OpenAI's RFC list explicitly invites SSM submissions to a **non-record track** — which is exactly the right venue for a Hymba-lite as a visibility play, separate from your record-track entry.

Tony's optimal play is therefore split: a transformer fork of PR #1394 (or #1493) on the record track, where the leaderboard meta has built a 0.14-BPB moat from naive baseline; and, if time and curiosity remain, a Hymba-lite — parallel attn+SSM heads, $d_{\text{state}}=64$, fp32 recurrence, INT6 MLP, no SSM QAT, BigramHash for cheap recall, legal score-first TTT on top — for the non-record track and OpenAI hiring visibility. Build the SSM understanding because *what you cannot create you do not understand*; build the transformer submission because *what you cannot quantize you cannot ship in 16 MB*.





# Another agent's feedback to this document

Fair pushback. Let me actually think rather than rubber-stamp the document.

## What "work" means matters

Three different thresholds:

1. **Beat naive baseline (1.2244)** — yes, plausibly, in 5 days with a clean S4D drop-in
2. **Be competitive (within 0.05 BPB of contemporary SOTA)** — possible with serious effort
3. **Set new SOTA (beat 1.0810)** — extremely unlikely in 5 days, low probability for anyone before April 30

The document collapses all three into "SSMs won't work." That's overclaiming.

## The empirical case against, examined

**The Hymba 1.1828 result.** This is the document's strongest evidence — a real SSM hybrid attempt landing 0.06 BPB behind transformer SOTA. But context matters: that submission had not absorbed the stack of techniques (depth recurrence, parallel residuals, QK-Gain, GPTQ, score-first TTT) that the contemporary transformer SOTA had. The leaderboard SOTA at 1.0810 is the product of 30+ PRs of community optimization. The Hymba attempt is one PR's worth of effort. Apples-to-apples comparison would be Hymba-1.1828 vs a transformer with comparable optimization budget — that gap is probably 0.04, not 0.10. Still real, but smaller than the headline.

**The 82% recall gap.** Real Zoology result. But measured at 1.4B params on Pile, not at 30M on FineWeb at 2048 context. FineWeb is recall-heavy but not as recall-heavy as long-context tasks. Critically: BigramHash exists on the leaderboard and is a non-parametric associative recall mechanism. Bolted onto an SSM hybrid, it directly addresses the gap. The document mentions BigramHash but doesn't credit it as a recall remedy for SSM.

**The quantization argument.** This is the most overstated piece. The document points out Mamba is quantization-hostile (true), but parameter golf doesn't require W4. INT5/INT6 is the meta. And — here's the move the document misses — *you don't have to quantize the SSM at all*. Keep the SSM block in fp16, quantize MLP and attention as the meta does. If your SSM is 2M of your 30M params, fp16 SSM costs 2 MB, which fits the budget. The Quamba complexity is only relevant if you're trying to push SSM weights to int4. You aren't.

**LR cliffs.** Real. But "incompatible with no-sweep training" is wrong — parameter golf allows offline hyperparameter tuning, you just pay in compute credits. People sweep LR all the time.

## The actual probability distribution

For Tony specifically — 5 days, Mac iteration, no Mamba background, learning while building:

- Beat current 1.0810 SOTA: **<2%**
- Land in 1.10–1.15 range: **~10%**
- Working hybrid that beats 1.18 (interesting result): **~30–40%**
- Land in 1.20–1.30 (above naive baseline): **~30%**
- Fail to ship anything: **~20–25%**

For someone with full Mamba expertise, dedicated focus, and the full deadline ahead of them: best case 5–10% chance of beating 1.0810, more like 30–40% of landing in the 1.10–1.15 range.

For *the field* — will any SSM submission beat 1.0810 by April 30? Probably not, but I'd put it at 15–25%, not 0%. That's the rate at which non-transformer architectures crack speedrun-style benchmarks once enough community attention focuses on them.

## What would have to be true for SSM to *win*

This is the test the document should have set out. For an SSM hybrid to beat transformer SOTA in parameter golf:

1. Recall gap needs to be closeable cheaply — BigramHash is one path. Plausible.
2. SSM block needs to contribute capacity per byte that exceeds an equivalent attention block — empirically unproven at this scale, but Hymba's ablations suggest yes for some configurations.
3. Training needs to be stable enough that LR sweeping doesn't eat the time budget — manageable with offline credits.
4. The hybrid composition needs to compose with the existing stack of meta tricks (TTT, GPTQ on transformer parts, depth recurrence) — uncertain. This is the real research question and there's no published evidence either way.

None of these is *impossible*. They're all *unproven*. The document treats unproven as proven negative, which is the actual error.

## Honest research verdict

The document's "almost certainly the wrong primary bet" is correct **for the goal of winning the competition**. It's wrong as a general statement about whether SSMs can work in this regime. The evidence supports:

- SSMs are currently behind in this competition by about 0.04–0.06 BPB at equivalent optimization effort
- This gap is real but not structural — it reflects accumulated meta-optimization on the transformer stack
- The gap can shrink with hybrid architectures and recall remedies, but no one has yet demonstrated full closure
- For Tony's stated goals (learn deeply + produce something on the non-record track), SSM is a fine bet

The "never gonna work" framing is the kind of overclaim that empirical ML often falls into — confusing "hasn't worked yet under these specific constraints" with "can't work." Both Mamba (vs. attention generally) and Hymba (at small scale) had to overcome similar dismissals.

What's *true* is that 5 days isn't enough time to be the person who proves SSMs can win parameter golf. What's *false* is that no one will ever prove it, or that you can't produce a meaningful SSM submission in 5 days.

You wanted me to be objective. The document is rigorous on theory and pessimistic-but-not-wrong on competitive prospects. It is wrong to elevate "won't win" into "won't work."

---

Searched the actual leaderboard rather than relying on the document. Here's what's verified.

## Every SSM submission to date

Four data points, none in the records track:

| Submission                           | What                                              | Result              |
| ------------------------------------ | ------------------------------------------------- | ------------------- |
| Hymba-style (PR cited in Issue #140) | 7L parallel attn+SSM heads, learned mixing        | 1.1828 BPB, 3 seeds |
| S4D-Lin hybrid                       | 2 SSM + 9 Transformer layers, zero-overhead claim | 1.1682 BPB          |
| PR #1227 SSM hybrid                  | Promising at d=192, broken at d=512               | "Scale deception"   |
| @dentity007 SSM PR                   | Proof-of-concept                                  | Not benchmarked     |

For reference: contemporary transformer SOTA at the time these were submitted was ~1.11–1.12. **All four SSM attempts landed 0.05–0.07 BPB behind the comparable transformer.**

The decisive piece of evidence isn't any single SSM submission — it's PR #831 (sseanliu), titled "Why Novel Architectures Fail at 16MB — Throughput-Quantization Co-optimization." This is the single best document for understanding the structural problem. Its argument:

> At 83ms/step, each 1ms of overhead costs ~7 steps. Each step improves BPB by ~0.001. Therefore: any technique must improve BPB by 0.007 per millisecond of overhead.

Then they tested 6 architectural innovations (GatedDeltaNet — SSM-family — among them). GatedDeltaNet "matches per-step quality but is 3.4× slower without torch.compile support." That extra cost wipes out any quality gain.

## Why the SSM submissions failed, decomposed

Five distinct mechanisms, each documented:

**1. The throughput tax.** Mamba's selective scan kernel doesn't compose with `torch.compile` and isn't tensor-core friendly. SSM blocks run 2–4× slower than attention blocks of comparable parameter count on H100. In a 10-minute budget, slower kernel = fewer optimizer steps = worse loss. This is the single biggest factor and the document I gave you yesterday underweighted it.

**2. Quantization hostility.** The leaderboard meta uses INT5/INT6 GPTQ on most weights. SSM weights and activations have heavy outliers; the recurrence amplifies quantization error multiplicatively over time. Documented case from PR #363: even depth recurrence (much milder than Mamba's selective scan) amplified quant error ~900× over 3 cycles. So either you keep SSM in fp16 (eats your byte budget) or you build Quamba2-grade infrastructure (impossible in 5 days).

**3. The associative recall gap.** Confirmed by Zoology and consistent with FineWeb's recall-heavy nature. Partly remediable with BigramHash, but not fully.

**4. Sharp LR cliffs.** Mamba's optimal LR band is narrower than transformer's. In a no-sweep regime this means more risk of landing off-optimal.

**5. Mac iteration loop is broken for Mamba.** `state-spaces/mamba` requires CUDA. You can't iterate Mamba on MPS. You'd be smoke-testing on `mamba-minimal` (pure PyTorch, slow Python scan) and shipping to a different code path on H100. That's a real productivity hit during a 5-day sprint.

## The strongest counter-evidence

One quote from the Issue #140 commentary on the Hymba submission worth taking seriously:

> Key: shallow models win (SSM makes each layer more powerful → 7L beats deeper pure transformers at same step budget).

7 SSM-hybrid layers competed at a roughly comparable level to deeper pure transformers. The 1.1828 isn't "SSM is hopelessly broken" — it's "SSM with 7 layers does about as well as transformer with 11 layers, before the meta optimizations kicked in." That's not a death sentence. It's a real result showing the architectural tradeoff.

Also: the field hasn't done a dedicated "make SSM work on parameter-golf" engineering push. Most SSM submissions are exploratory. The transformer side has had 30+ PRs of optimization layered on. Apples-to-apples requires either bringing the meta techniques to the SSM (hard) or comparing un-optimized baselines (more honest).

## How to approach it for *your* goals

Your stated goals: learn deeply + produce something on the non-record track. This changes the design considerably.

**Don't try to beat 1.0810.** That's the wrong target. Your time is too short, your hardware too limited, the meta too entrenched. Aim instead for one of these two non-record contributions:

**Option A — The clean experiment.** Take the canonical baseline. Drop in *one* S4D layer replacing one attention layer. Touch nothing else. Run on RunPod 1xH100 (cheaper than 8xH100, fine for a non-record run). Compare BPB. Whether positive or negative, write it up honestly. This is how PR #831 became a respected non-record submission — they *systematically demonstrated* what doesn't work and explained why. Negative results from an honest controlled experiment are valuable.

**Option B — The micro-hybrid.** Inspired by Hymba but stripped down: instead of full parallel attn+SSM heads, just add a single SSM-only block at the top of the stack as a "memory readout" — like a learned summarization layer. Smaller scope than Hymba, weirder architectural commitment, more original. Higher implementation risk, more learning, more likely to produce something genuinely yours.

## Challenges you must tackle, ranked

1. **Get an SSM block running on Mac iteration loop.** `mamba-minimal` is the right starting point — single 300-line PyTorch file, runs on MPS. You'll need to verify it computes the same thing as the official `selective_scan_ref`. This is non-trivial; budget half a day.
2. **Decide what \*not\* to quantize.** If your SSM block is small (say 2M of 30M total params), keep it bf16 and quantize only the rest. Document the byte budget carefully.
3. **Pick discretization wisely.** Start with S4D-Lin, not Mamba's selective scan. It's diagonal, ZOH-discretized, two lines of code, easy to debug, and you can add selectivity later if there's time. The document's §1.6 covers this; the Annotated S4D walks the implementation.
4. **Match the loss-comparison surface.** Run the baseline at the same step count, same data, same eval, and report the delta. Don't compare against 1.0810; compare against the baseline you reproduced yourself.
5. **Don't sweep LR exhaustively.** Pick three points (0.005, 0.01, 0.02), run one short smoke test each, pick the best, run the full eval. Done.
6. **Expect to land between 1.20 and 1.30.** That's fine. Your goal isn't a number; it's a clean experiment plus a writeup that demonstrates you understand what you built and why it did what it did.

The honest research verdict: SSMs *can* be made to work in this regime in principle, but the engineering surface is large, the meta isn't on your side, and 5 days isn't enough to crack it. What 5 days *is* enough for is a clean, honest, cited contribution to the non-record track that demonstrates real understanding. That's more valuable to the OpenAI hiring lens than a 14th-place leaderboard entry anyway.