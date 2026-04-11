# Category 3: Quantum Machine Learning (QML)
## Research for Parameter Golf — Model Training Under 16MB / 10 Minutes

**Research date:** March 24, 2026  
**Scope:** Google, IBM, IonQ, Xanadu/PennyLane — plus broader QML landscape. Focus: applicability to training small, highly capable models within extreme resource constraints (16MB, 10 min training).

---

## Executive Summary

Quantum Machine Learning sits at an intersection of genuine promise and present-day overhype. As of early 2026, the field has produced several real, usable results — including quantum-inspired adapters that compress fine-tuning by 40–76% versus LoRA, quantum kernels that outperform classical methods on small/noisy datasets, and hybrid quantum-classical architectures that improve LLM fine-tuning accuracy with fewer trainable parameters.

**For the 16MB / 10-minute constraint specifically:** QML's most actionable benefit is **quantum-inspired techniques running on classical hardware** — tensor network adapters, quantum-inspired orthogonal fine-tuning, and quantum kernel SVMs. These don't require access to a quantum computer today. True quantum hardware (IBM, IonQ, Google) remains NISQ-era (noisy, limited qubits), with practical quantum advantage for training expected 2026–2029. The "quantum-inspired" flavor is immediately deployable.

---

## 1. What Is Quantum Machine Learning?

**Core idea:** Replace or augment classical bits (0 or 1) with qubits that can exist in superposition (both states simultaneously), enabling:
- **Superposition** — evaluate multiple computational paths in parallel
- **Entanglement** — non-local correlations between features, impossible classically
- **Interference** — amplify correct answers, cancel wrong ones
- **Exponential Hilbert space** — n qubits represent 2ⁿ states, enabling exponential compression of feature spaces

**Three flavors of QML:**
1. **Purely quantum** — full quantum circuits, requires quantum hardware (future-facing)
2. **Hybrid quantum-classical** — quantum circuits handle specific subroutines, classical handles the rest (current NISQ era)
3. **Quantum-inspired** — algorithms derived from quantum principles, run on classical hardware (immediately available)

**Key paradigm:** Parameterized Quantum Circuits (PQCs) / Variational Quantum Circuits (VQCs) — trainable quantum circuits whose parameters are optimized via classical optimizers, analogous to neural network weights.

---

## 2. Major Players

### 2.1 Google Quantum AI
**What they're building:** Superconducting qubit systems. The Willow chip (70 qubits) performs calculations in minutes that would take classical supercomputers 10 septillion years. The 68-qubit processor used in generative QML research (Sept 2025).

**Key 2025–2026 breakthrough:** "Generative Quantum Advantage" — first experimental evidence that quantum computers can *learn* probability distributions and generate outputs no classical machine can replicate.

- Research led by Hsin-Yuan Huang, Hartmut Neven, Ryan Babbush (Google Quantum AI)
- Demonstrated three tasks: generating complex bitstring distributions, compressing quantum circuits, learning quantum states
- Models called "instantaneously deep quantum neural networks" — **trainable on classical hardware, requiring quantum for inference**
- Also proved quantum advantage for learning shallow neural networks with natural (non-uniform) data distributions (Nature Communications, Dec 2025)

**Tooling:** TensorFlow Quantum (TFQ) + Cirq
- TFQ integrates quantum computing logic designed in Cirq with TensorFlow APIs
- Supports hybrid quantum-classical model building
- URL: https://www.tensorflow.org/quantum | https://github.com/quantumlib/Cirq

**Why it matters for small models:** Google's 5-stage roadmap to practical quantum advantage positions 2025–2026 as "Stage 2: Scientific value" — quantum computers capable of generating problem instances where quantum outperforms classical. This creates a path where quantum-trained features/representations could eventually compress into a small classical model for deployment.

**Source:** https://thequantuminsider.com/2025/09/15/generative-ai-meets-quantum-advantage-in-googles-latest-study/  
**Nature paper:** https://www.nature.com/articles/s41467-025-68097-2

---

### 2.2 IBM Quantum
**What they're building:** Superconducting qubit processors. The IBM Quantum Nighthawk (120 qubits, 218 tunable couplers, announced Nov 2025) is their most advanced processor.

**2026 target:** IBM expects the first verified cases of quantum advantage to be confirmed by the wider scientific community by end of 2026.

**Hardware milestones:**
- Nighthawk: 120 qubits, 30% higher circuit complexity vs. Heron, supports up to 5,000 two-qubit gates
- Projected: 7,500 gates by end of 2026, 10,000 gates in 2027, 15,000 gates + 1,000+ qubits by 2028
- IBM Quantum Loon processor: experimental, real-time quantum error decoding
- Shift to 300mm wafer fabrication for scaling

**Software: Qiskit Machine Learning**
- Open-source Python library (Apache 2.0 license)
- Components: Quantum Neural Networks (QNNs), Variational Quantum Classifier (VQC), Variational Quantum Regressor (VQR), Quantum Kernel (QK) Methods, Quantum SVM
- PyTorch connector for hybrid quantum-classical neural networks
- Scikit-learn integration
- Handles both real quantum hardware AND classical simulators
- URL: https://github.com/qiskit-community/qiskit-machine-learning
- Paper: https://arxiv.org/html/2505.17756v1

**2027 roadmap:** IBM plans to release advanced Qiskit computational libraries specifically focused on machine learning and optimization.

**Quantum advantage tracker:** IBM, Algorithmiq, Flatiron Institute, and BlueQubit launched open community tracker for verifying quantum advantage claims (2025). Monitors observable estimation, variational problems, classical-verification tasks.

**Energy efficiency note:** A study (Nature Scientific Reports, Dec 2025) benchmarked QNN energy consumption on IBM Qiskit — quantum models on emulated hardware vs. real quantum hardware show different energy profiles, with projected gains as problem sizes scale beyond ~46 qubits.

**Source:** https://thequantuminsider.com/2025/11/12/ibm-reveals-new-quantum-processors-software-and-algorithm-advances/  
**Paper (Qiskit ML):** https://arxiv.org/html/2505.17756v1

---

### 2.3 IonQ
**What they're building:** Trapped-ion quantum computers. The Forte Enterprise system offers 36 algorithmic qubits with high-fidelity operations. Available via major cloud providers (AWS, Azure, Google Cloud).

**Key 2025 breakthrough: Quantum LLM Fine-Tuning**
- Published research demonstrating hybrid quantum-classical architecture for LLM fine-tuning
- Method: Integrate a parameterized quantum circuit as a **classification head** within a pre-trained LLM
- Tested on SST-2 (Stanford Sentiment Treebank) sentiment analysis benchmark
- Results:
  - Quantum-enhanced model **outperformed classical-only methods with similar parameter counts**
  - Classification accuracy increased with number of qubits used
  - Projected significant energy savings for inference as problem size scales beyond 46 qubits
  - Successfully ran on IonQ's trapped-ion quantum hardware

**Also:** Quantum-Enhanced Generative Adversarial Networks (QGANs) for materials science — generated synthetic images of steel microstructures; QGAN-generated images achieved higher quality scores than classical GAN in 70% of test cases. Collaboration with automotive manufacturer + AIST's G-QuAT.

**Why it matters for small models:** The parameterized quantum circuit as a lightweight classification head is directly relevant — a 16MB model could theoretically use a quantum-circuit "head" for inference on key tasks while the base model stays classical and compact.

**Source:** https://quantumzeitgeist.com/ionq-quantum-enhanced-llm-optimization/  
**Detailed report:** https://quantumcomputingreport.com/ionq-demonstrates-hybrid-quantum-ai-applications-in-llm-fine-tuning-and-materials-modeling/

---

### 2.4 Xanadu / PennyLane
**What they're building:** Xanadu builds **photonic quantum computers** (light-based, room-temperature operable). Key hardware:
- **Aurora** (Feb 2026): World's first universal, fault-tolerant photonic quantum computer, announced in Nature. First system combining all subsystems for universal + fault-tolerant computation in photonic architecture.
- Roadmap: Improving error rates and optical loss, scaling up after Aurora

**Latest partnerships (2026):**
- **Xanadu + Lockheed Martin** (Feb 2026): Joint initiative to redefine QML foundations. Focus: generative models that exploit Fourier-based operations inaccessible to classical ML, especially for data-scarce environments. Applications: defense, finance, pharmaceuticals.
- **Xanadu + US Air Force Research Laboratory**: Partnered to advance photonic quantum technologies, evaluating Xanadu's system architectures + QML toolchain.

**PennyLane (the software):**
- Open-source Python library for quantum ML (Xanadu-developed, community maintained)
- Works with PyTorch, TensorFlow, JAX — seamless hybrid classical-quantum training
- Supports quantum hardware: Xanadu Aurora, IBM Q, and more
- **Lightning** simulator suite: runs on CPU and GPU (including AMD GPUs via ROCm)
  - Demonstrated at scale on Frontier Supercomputer (world's fastest)
  - AMD partnership (Dec 2025): PennyLane Lightning on AMD Developer Cloud
- Automatic differentiation for gradient-based optimization of quantum circuits
- Key feature: Differentiable programming — quantum circuits trainable like neural networks

**Key PennyLane capabilities:**
- `StronglyEntanglingLayers` — trainable VQC layers, parameterized like NN weights
- Quantum Convolutional Neural Networks (QCNNs) — generalization from few training examples
- `AngleEmbedding` / `AmplitudeEmbedding` — classical data → quantum state encoding
- Parameter shift rule for analytic gradients

**Xanadu + Lockheed focus on data-scarce environments** is directly relevant — their QML approach specifically targets settings where classical models struggle with insufficient training data.

**Source (Aurora):** https://postquantum.com/quantum-computing-companies/xanadu/  
**Source (Lockheed):** https://www.prnewswire.com/news-releases/xanadu-and-lockheed-martin-launch-joint-research-initiative-to-redefine-the-foundations-of-quantum-machine-learning-302698386.html  
**PennyLane + AMD:** https://www.amd.com/en/developer/resources/technical-articles/2025/fast-track-quantum-simulation-with-xanadu-s-pennylane-lightning-.html

---

## 3. Core QML Techniques Relevant to Small Model Training

### 3.1 Variational Quantum Circuits (VQCs) / Parameterized Quantum Circuits (PQCs)

**What it is:** Quantum circuits with tunable rotation angles as trainable parameters. A VQC encodes input data (via angle or amplitude embedding), applies entangling layers with trainable gates, and produces outputs via quantum measurement. The parameters are optimized using classical optimizers (Adam, gradient descent, etc.).

**Why it matters:** VQCs are the "neural network layers" of quantum computing — you can stack them, compose them, and train them via backprop-equivalent methods (parameter shift rule). They can express functions that would require exponentially larger classical networks.

**Practical architecture:**
```python
# PennyLane VQC example
import pennylane as qml

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.AngleEmbedding(x, wires=range(4))          # Encode input
    qml.StronglyEntanglingLayers(params, wires=range(4))  # Trainable layers
    return qml.expval(qml.PauliZ(0))               # Measure output

# Optimize like a classical neural network
opt = qml.GradientDescentOptimizer()
```

**Key limitation:** The parameter shift rule requires O(N) gradient circuit evaluations for N parameters. A 2025 Nature npj Quantum Information paper estimated this allows only ~9,000 parameter circuits with a single day of computation. This is actually a *feature* for small model training — it enforces extreme parameter efficiency.

**New approach (2025): Density QNNs** — prepares mixtures of trainable unitaries (not pure states). More expressive per-parameter, designed to be compact yet avoid classical simulation or barren plateaus. Paper: https://www.nature.com/articles/s41534-025-01099-6

**Source:** https://medium.com/quantum-computing-and-ai-ml/variational-quantum-circuits-for-machine-learning-aa982a03605f

---

### 3.2 Quantum Neural Networks (QNNs)

**What it is:** Quantum circuits designed to mimic neural network structure — encoding layers, parameterized transformation layers, measurement/readout. Sub-types:
- **Quantum Convolutional Neural Networks (QCNNs)** — hierarchical structure for spatial data
- **Quantum Recurrent Networks** — sequence modeling
- **Quantum Transformers** — attention mechanisms in Hilbert space

**QCNNs — key 2025–2026 results:**
- QCNNs generalize from *very few* training samples — a PennyLane demo (Sept 2025) demonstrates this explicitly for phase detection tasks
- Hybrid QCCNN (Nature Scientific Reports, Aug 2025): leverage high-dimensional Hilbert spaces + entanglement to surpass classical CNNs in image classification accuracy under comparable architectures
- Efficient QCNNs overcoming hardware constraints: arxiv 2505.05957 (May 2025)

**Barren Plateau Problem (and solutions):**
A key challenge: as circuits deepen, gradients vanish exponentially (similar to vanishing gradients in classical NNs). Recent 2025 solutions:
- Neural-network generated quantum states to initialize VQCs — mitigates barren plateaus (arXiv 2411.08238, updated Nov 2025)
- Direct parameterization of unitary matrices instead of stacking gates (arXiv 2508.02459)
- AI-driven Submartingale-based framework (arXiv 2502.13166)
- Geometric optimization on Lie groups explains *why* NN-generated parameters work (arXiv 2512.02078)

**High-expressibility QNNs using only classical resources (2026):** A new Quantum Machine Intelligence paper (Springer, published ~March 2026) demonstrates QNN architectures that are near-term quantum ML architectures leveraging PQCs to *improve upon classical counterparts* — and can be evaluated on classical hardware. URL: https://link.springer.com/article/10.1007/s42484-026-00371-y

---

### 3.3 Quantum Kernel Methods

**What it is:** Map classical data into a quantum Hilbert space using a quantum feature map, then compute kernel (similarity) matrices between data points. Used with classical SVMs (Quantum SVM / QSVM).

**Key advantage:** The kernel computation lives in an exponentially large feature space that's classically intractable to compute directly. This can create decision boundaries impossible for classical kernels.

**2025 practical evidence:**
- Benchmarking study (Quantum Machine Intelligence, Apr 2025, 64 datasets): Fidelity Quantum Kernels (FQKs) and Projected Quantum Kernels (PQKs) tested with 9 encoding circuits up to 15 qubits — provides guidance on when quantum kernels beat classical
- Quantum Kernel Learning for Small Dataset Modeling (Advanced Science, June 2025): Applied to semiconductor fabrication — quantum kernels with trainable alignment layers outperform classical on small datasets
- Quantum Naive Bayes comparison (arXiv 2502.11951): Small quantum circuits approximate probabilistic inference with competitive accuracy vs. classical and **better robustness to noisy data**

**URL (benchmarking):** https://link.springer.com/article/10.1007/s42484-025-00273-5  
**Qiskit implementation:** `QuantumKernel` with `ZZFeatureMap` works with scikit-learn SVC

---

### 3.4 Hybrid Quantum-Classical Architecture

**What it is:** Quantum circuits handle specific computationally intensive subroutines; classical processors handle optimization, preprocessing, inference scaling. The dominant paradigm in NISQ era.

**Current hybrid pattern:**
- Classical: data loading, optimization loop, post-processing
- Quantum: feature mapping, kernel evaluation, parameterized layer execution

**IonQ's LLM fine-tuning architecture:** Quantum circuit as classification head on pre-trained LLM. Train classically, quantum circuit for final prediction. Fewer parameters + better accuracy than purely classical head.

**Google's "train classically, infer quantumly":** Instantaneously deep quantum neural networks trained on classical machines, requiring quantum processor only for inference. Relevant for production deployment scenarios.

---

### 3.5 QAOA (Quantum Approximate Optimization Algorithm)

**What it is:** Alternating layers of problem-specific and mixing Hamiltonians, tuned to solve combinatorial optimization. Applicable to hyperparameter search, architecture search, and training scheduling.

**Why relevant to small model training:**
- Hyperparameter optimization for model architecture is fundamentally a combinatorial search problem
- QAOA can potentially search larger hyperparameter spaces more efficiently than grid search
- IBM has prioritized QAOA in their quantum advantage tracker experiments

---

## 4. Quantum-Inspired Methods (Run on Classical Hardware — Available Now)

This is the **most immediately actionable category** for the 16MB / 10-minute constraint. These techniques are derived from quantum principles but run entirely on classical hardware.

### 4.1 QuIC Adapters (Quantum-Inspired Compound Adapters)
**What it is:** PEFT method inspired by Hamming-weight preserving quantum circuits. Uses quantum-circuit-motivated orthogonal matrix structures for fine-tuning.

**Results:**
- Finetunes models using **less than 0.02% memory footprint of base model**
- Over **40× smaller parameter count than LoRA**
- First-order configuration: matches LoRA performance
- Higher-order: substantial parameter compression with modest performance trade-off
- Preserves pretrained representations via orthogonality constraints
- **Can natively deploy on quantum computers when available**

**Tested on:** LLaMA, Vision Transformers — language, math, reasoning, vision benchmarks.

**Source (paper):** https://arxiv.org/html/2502.06916v2 (QC Ware, Sorbonne Université, Feb 2025)

---

### 4.2 QTHA — Quantum Tensor Hybrid Adaptation
**What it is:** Fine-tuning method combining Quantum Neural Network (QNN) with tensor network decomposition. Decomposes pre-trained weights into quantum neural network + tensor network representations. Uses quantum state superposition to overcome classical rank limitations.

**Results vs. LoRA:**
- Reduces trainable parameters by **76%**
- Reduces training loss by up to **17%**
- Improves test set performance by up to **17%** within the same training steps
- Validated on actual quantum hardware (Origin Quantum, China)
- First engineering-ready foundation for quantum-enhanced LLM fine-tuning

**Source:** https://arxiv.org/html/2503.12790v2 (Origin Quantum + USTC, March 2025)

---

### 4.3 QuanTA (Quantum Tensor Adaptation)
**What it is:** Constructs PEFT adapters via contracted quantum-inspired tensor networks. High-order parameter adjustments using quantum circuit-inspired tensor decompositions.

**Part of a growing family:** QuanTA → QuIC → QTHA — each generation more parameter-efficient.

---

### 4.4 CompactifAI (Multiverse Computing)
**What it is:** Tensor network + singular value truncation for model compression. **60% parameter reduction with 84% energy efficiency gains without sacrificing accuracy** (Multiverse Computing).

**Why relevant:** This is the most aggressively compressed approach in the quantum-inspired space. 84% energy savings with 60% fewer parameters maps directly to faster training within a time budget.

---

### 4.5 QPA — Quantum Parameter Adaptation
**What it is:** Uses Quantum Neural Networks with hardware-efficient ansätze to predict weight parameters for LoRA adapter modules. The QNN generates compact tuning parameters via hybrid quantum-classical mappings.

**Source:** Liu et al., 2025

---

### 4.6 Tensor Network Methods
**What it is:** Mathematical framework from quantum physics (used to study many-body quantum systems) now applied to classical ML. Matrix Product States (MPS), Matrix Product Operators (MPO), Tree Tensor Networks.

**Recent results:**
- Tensor network methods for quantum-inspired image processing (arXiv 2510.23089, Feb 2026): "strike a middle ground between fully-fledged quantum computing and classical computing, significantly speed up certain classical operations"
- MPO representations in QTHA add robustness through localized entanglement regularization

**Available tools:** `quimb`, `tensornetwork` (Google), `tntorch` (PyTorch-native)

---

## 5. Current Limitations (Honest Assessment)

**For actual quantum hardware:**
1. **Qubit count & quality:** NISQ-era machines (20–150 reliable qubits) are too small for most practical ML tasks
2. **Decoherence:** Quantum states collapse quickly; circuit depth is limited
3. **Barren plateaus:** Gradients vanish exponentially in deep circuits (being solved, see 3.2)
4. **Data loading bottleneck:** Encoding classical data into qubits introduces overhead (amplitude encoding: O(2ⁿ) prep time)
5. **Parameter shift rule cost:** O(N) circuit evaluations per gradient step — only viable for small parameter counts (~9,000 max in 1 day)
6. **Noise & error rates:** NISQ hardware has error rates too high for many practical tasks; error correction is IBM's 2029 target

**For quantum-inspired (classical execution):**
1. Not "quantum" — these are classically-inspired approximations
2. Performance gains are real but typically 10–20% improvements, not exponential speedups
3. Tensor network methods hit exponential wall for highly entangled systems

**Bottom line:** True quantum hardware won't train your model today. Quantum-inspired methods absolutely can.

---

## 6. How QML Could Specifically Help: 16MB / 10 Minutes

### Directly Applicable Now (No Quantum Hardware Required)

| Technique | Parameter Reduction | Speed Impact | Applicability |
|-----------|--------------------:|:------------|:-------------|
| QuIC Adapters | 40× smaller than LoRA | ✅ Less memory, faster | Fine-tuning pre-trained base |
| QTHA | 76% fewer trainable params | ✅ Same steps, better loss | Fine-tuning transformers |
| CompactifAI | 60% param reduction | ✅ 84% energy savings | Model compression |
| QPA | Compact LoRA modules | ✅ Moderate | Fine-tuning |
| Tensor Networks | Architectural compression | ✅ Trade expressiveness for size | Core architecture |
| Quantum Kernel SVM | Replaces NN head | ✅ No backprop needed | Small dataset classification |

**Scenario A — Start from scratch, train in 10 min:**
- Use tensor network architectures (MPS layers instead of dense) to dramatically reduce parameter count
- Result: Same expressiveness, fraction of the parameters → fits in 16MB more easily
- Train quantum kernel SVM as the classification head (no gradient computation for that layer)

**Scenario B — Fine-tune a pre-trained model, compress to 16MB:**
- Apply QuIC or QTHA adapters: 40–76% fewer trainable parameters = faster convergence
- Use CompactifAI-style tensor SVD truncation to compress the final model to 16MB
- Combined pipeline: quantum-inspired PEFT + tensor compression

**Scenario C — Architecture search in constrained time:**
- QAOA (via IBM Qiskit on simulator) for hyperparameter optimization
- Evaluate 100s of architectures in parallel (quantum speedup on QAOA)

### Requires Quantum Hardware (Future-Facing, 2026–2028)

| Technique | Provider | Expected Timeline |
|-----------|----------|------------------|
| VQC as feature extractor, classical head | IBM/IonQ/PennyLane | Now (NISQ, noisy) |
| Quantum kernel evaluation | IBM Qiskit | Now (limited scale) |
| Quantum fine-tuning head (IonQ-style) | IonQ | Now (36 AQ) |
| Verified quantum advantage on optimization | IBM | End of 2026 |
| Fault-tolerant quantum ML | IBM | 2029 |

**Practical hybrid approach today:**
1. Use PennyLane (simulator or IBM cloud) to train a small VQC on your task
2. Extract learned quantum feature representations
3. Distill those representations into a small classical model ≤ 16MB
4. The quantum circuit found better features; the classical model deploys them compactly

---

## 7. Tools & How to Access Them

### PennyLane (Xanadu) — Best starting point for QML
- **Install:** `pip install pennylane`
- **GPU acceleration:** `pip install pennylane-lightning-gpu` (NVIDIA) or ROCm (AMD)
- **Backends:** CPU simulator, GPU, IBM Q hardware, Xanadu Aurora, Amazon Braket
- **URL:** https://pennylane.ai
- **Demos:** https://pennylane.ai/qml/demos/ (200+ tutorials)
- **Quantum Convolutional Neural Networks:** https://pennylane.ai/qml/demos/tutorial_learning_few_data
- **Generalization from few data:** https://pennylane.ai/qml/demos/tutorial_annni

### Qiskit Machine Learning (IBM) — Best for quantum hardware access
- **Install:** `pip install qiskit-machine-learning`
- **URL:** https://github.com/qiskit-community/qiskit-machine-learning
- **Free hardware access:** IBM Quantum Experience (cloud, free tier available)
- **Components:** VQC, QNN, QuantumKernel, NeuralNetworkClassifier

### TensorFlow Quantum (Google)
- **Install:** `pip install tensorflow-quantum`
- **URL:** https://www.tensorflow.org/quantum
- **Integrates:** Cirq quantum circuits + TensorFlow training loop
- **Best for:** TF-native workflows wanting quantum layers

### QuIC Adapters
- **Paper:** https://arxiv.org/abs/2502.06916 (QC Ware)
- **Note:** Implementation details in paper; not yet a standalone pip package as of March 2026

### QTHA
- **Paper:** https://arxiv.org/abs/2503.12790 (Origin Quantum)
- **Status:** Research prototype, validated on real quantum hardware

### Tensor Networks (quantum-inspired, classical)
- **`quimb`:** `pip install quimb` — tensor network library
- **`tensornetwork`:** `pip install tensornetwork` (Google, TF/NumPy/JAX backends)
- **`tntorch`:** PyTorch native tensor decompositions

---

## 8. Key Papers & References

| Paper | Year | Key Finding | URL |
|-------|------|-------------|-----|
| Generative Quantum Advantage (Google) | Sept 2025 | First experimental generative QA on 68-qubit processor | arXiv 2509.09033 |
| Quantum advantage for shallow NNs (Nature) | Dec 2025 | Exponential quantum advantage for learning periodic neurons | https://www.nature.com/articles/s41467-025-68097-2 |
| Qiskit Machine Learning (IBM) | May 2025 | Full library description, VQC/QNN/QK algorithms | https://arxiv.org/html/2505.17756v1 |
| QuIC Adapters (QC Ware) | Feb 2025 | 40× LoRA compression via quantum-inspired orthogonal adapters | https://arxiv.org/abs/2502.06916 |
| QTHA (Origin Quantum) | March 2025 | 76% param reduction + 17% perf improvement vs LoRA | https://arxiv.org/abs/2503.12790 |
| Hybrid QML Framework | Sept 2025 | Architecture for CQ paradigm, quantum kernels, VQAs | https://arxiv.org/abs/2502.11951 |
| Density QNNs (npj Quantum Info) | Nov 2025 | New QNN family efficient to train, avoids barren plateaus | https://www.nature.com/articles/s41534-025-01099-6 |
| Neural NN → Barren Plateau Fix | Nov 2025 | NNs generating quantum states mitigate barren plateaus | https://arxiv.org/abs/2411.08238 |
| QCNN for Image Classification | Oct 2025 | QCNNs leverage Hilbert space to surpass classical CNNs | https://www.mdpi.com/2227-7390/13/19/3148 |
| Quantum Kernel Benchmarking (64 datasets) | Apr 2025 | Large-scale FQK/PQK study — when quantum kernels win | https://link.springer.com/article/10.1007/s42484-025-00273-5 |
| IonQ LLM Fine-Tuning | May 2025 | Parameterized quantum circuit as LLM head beats classical | https://quantumzeitgeist.com/ionq-quantum-enhanced-llm-optimization/ |
| Carbon Efficient Quantum AI (IBM Qiskit) | Dec 2025 | Energy consumption study: QNN vs. classical, hardware vs. emulated | https://www.nature.com/articles/s41598-025-28582-6 |

---

## 9. Verdict for Parameter Golf

### What to Use Today

**Tier 1 — Immediately actionable, no quantum hardware:**
- **QuIC Adapters** or **QTHA**: Quantum-inspired PEFT that cuts trainable parameters 40–76% vs. LoRA. If you're fine-tuning toward 16MB, these are the most aggressive compression tools in the PEFT space.
- **Tensor network compression (CompactifAI-style SVD truncation)**: Post-training compression to hit 16MB target. Achieve 60% fewer parameters with 84% energy reduction.
- **Quantum Kernel SVM (Qiskit/PennyLane)**: Replace the classification head with a quantum kernel SVM trained on small data — fewer parameters in the head, potentially better generalization from limited training samples.

**Tier 2 — Accessible today with free cloud quantum access:**
- **PennyLane + IBM Quantum Experience**: Train a small VQC as a feature extractor or classification head. Free tier available. Use simulator for development, submit to real hardware for validation.
- **Qiskit + Nighthawk (IBM cloud)**: Access to 120-qubit Nighthawk processor for quantum kernel computation on your task's feature space.

**Tier 3 — Forward-looking, 2026–2028:**
- IonQ's quantum LLM head architecture (access via AWS/Azure)
- Google's generative quantum models for data augmentation (helps if you're data-scarce)
- IBM's verified quantum advantage experiments for optimization (end of 2026)

### The One Sentence Summary

**In 2026, "quantum ML" for a 16MB / 10-minute constraint means: use quantum-inspired adapters (QuIC/QTHA) to compress fine-tuning by 40–76%, use tensor network truncation to hit your size target, and optionally use PennyLane's quantum kernel SVM as a compact classification head — actual quantum hardware is an optional accelerant, not a requirement.**

---

*Research compiled March 24, 2026. Sources include arXiv preprints, Nature, IBM Quantum Blog, IonQ press releases, Xanadu/PennyLane documentation, and industry analysis.*
