"""
Composable code patches for train_gpt.py.

Each patch function takes the full source text and returns modified source text.
Patches target non-overlapping regions so they can be composed freely for
Pack 2 composite candidates.

Usage:
    from patches import apply_patches
    patched = apply_patches(original_source, ["zloss", "nuclear_norm"])

Revision: 2026-03-24 — original hypothesis set.
Stage2_1 owns the community playbook (LeakyReLU^2, EMA, MuonWD, XSA, GPTQ, etc).
Stage3 tests novel mechanisms from other fields:
  - zloss: logit regularization (from PaLM / large-scale LM stabilization)
  - adaptive_ns_steps: Newton-Schulz step scheduling (from control theory gain scheduling)
  - nuclear_norm: low-rank weight regularization (from rate-distortion / transform coding)
  - weight_perturbation: flat minima search (from Langevin dynamics / channel coding)
  - grad_centralization: gradient preprocessing (from ICCV 2020 computer vision)
"""
from __future__ import annotations

import os


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"Patch '{patch_name}': target string not found in source")
    if count > 1:
        raise ValueError(
            f"Patch '{patch_name}': target string found {count} times (must be unique)"
        )
    return source.replace(old, new)


# ---------------------------------------------------------------------------
# Patch: zloss
# Analogy: PaLM / Gemini z-loss stabilization
# Surface: Loss function (logit magnitudes)
# Target region: GPT.forward loss computation
# ---------------------------------------------------------------------------

def patch_zloss(source: str) -> str:
    """Add z-loss regularization: penalize log(sum(exp(logits)))^2.

    From PaLM (Chowdhery et al., 2022). Keeps logit magnitudes small,
    which stabilizes training AND makes weights more quantization-friendly.
    Distinct from label smoothing (target distribution) and logit softcap
    (hard clamp). Z-loss is a soft differentiable penalty on the partition
    function.
    """
    name = "zloss"

    source = _replace_unique(
        source,
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")",
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        ce = F.cross_entropy(logits.float(), targets, reduction=\"mean\")\n"
        "        _zloss_w = float(os.environ.get(\"ZLOSS_WEIGHT\", \"0.0\"))\n"
        "        if _zloss_w > 0:\n"
        "            log_z = torch.logsumexp(logits.float(), dim=-1)\n"
        "            ce = ce + _zloss_w * (log_z ** 2).mean()\n"
        "        return ce",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: adaptive_ns_steps
# Analogy: Control theory gain scheduling — adapt controller precision
#          to the operating regime
# Surface: Optimizer internals (Newton-Schulz iteration count)
# Target region: Muon.step() where backend_steps is read
# ---------------------------------------------------------------------------

def patch_adaptive_ns_steps(source: str) -> str:
    """Schedule Newton-Schulz iteration count based on training phase.

    Early: more steps (7) — noisier gradients need better orthogonalization.
    Mid: default (5) — standard regime.
    Late/warmdown: fewer steps (3) — weights converging, cheaper ortho is fine,
    and the throughput gain (fewer matmuls per step) means more steps in the
    wallclock budget.

    Inspired by gain scheduling in control theory: the controller is tuned
    differently depending on the operating point of the plant.
    """
    name = "adaptive_ns_steps"

    # Add a global step counter and NS schedule function before Muon class
    source = _replace_unique(
        source,
        "class Muon(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):",
        "_ADAPTIVE_NS = os.environ.get(\"NS_STEPS_EARLY\", \"\") != \"\"\n"
        "_NS_EARLY = int(os.environ.get(\"NS_STEPS_EARLY\", \"7\"))\n"
        "_NS_MID = int(os.environ.get(\"NS_STEPS_MID\", \"5\"))\n"
        "_NS_LATE = int(os.environ.get(\"NS_STEPS_LATE\", \"3\"))\n"
        "_NS_TOTAL_STEPS = [0]  # mutable container for step tracking\n"
        "_NS_MAX_STEPS = [0]    # set from main loop\n"
        "\n"
        "def _get_adaptive_ns_steps() -> int:\n"
        "    if not _ADAPTIVE_NS or _NS_MAX_STEPS[0] <= 0:\n"
        "        return _NS_MID\n"
        "    frac = _NS_TOTAL_STEPS[0] / _NS_MAX_STEPS[0]\n"
        "    if frac < 0.2:\n"
        "        return _NS_EARLY\n"
        "    elif frac > 0.8:\n"
        "        return _NS_LATE\n"
        "    return _NS_MID\n"
        "\n"
        "\n"
        "class Muon(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):",
        name,
    )

    # Replace the fixed backend_steps usage with adaptive call
    source = _replace_unique(
        source,
        "                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)\n"
        "                    # Scale correction from Muon reference implementations.\n"
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        "                    _ns = _get_adaptive_ns_steps() if _ADAPTIVE_NS else backend_steps\n"
        "                    g = zeropower_via_newtonschulz5(g, steps=_ns)\n"
        "                    # Scale correction from Muon reference implementations.\n"
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        name,
    )

    # Increment step counter after optimizer step in main loop
    source = _replace_unique(
        source,
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        zero_grad_all()\n"
        "\n"
        "        step += 1",
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        zero_grad_all()\n"
        "        if _ADAPTIVE_NS:\n"
        "            _NS_TOTAL_STEPS[0] = step + 1\n"
        "\n"
        "        step += 1",
        name,
    )

    # Set max steps from args.iterations
    source = _replace_unique(
        source,
        "    step = 0\n"
        "    while True:",
        "    if _ADAPTIVE_NS:\n"
        "        _NS_MAX_STEPS[0] = args.iterations\n"
        "    step = 0\n"
        "    while True:",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: nuclear_norm
# Analogy: Rate-distortion theory / transform coding — concentrate energy
#          in fewer coefficients for better lossy compression
# Surface: Regularization (weight matrix structure / effective rank)
# Target region: training loop loss computation
# ---------------------------------------------------------------------------

def patch_nuclear_norm(source: str) -> str:
    """Add nuclear norm regularization to matrix parameter weights.

    Penalizes the sum of singular values of weight matrices, which
    encourages low effective rank. In transform coding, concentrating
    energy in fewer basis coefficients improves compression quality.
    Analogously, low-rank weight matrices have fewer significant singular
    values, making quantization less destructive.

    Distinct from weight decay (penalizes Frobenius norm / magnitude) and
    QAT (adds quantization-specific noise). Nuclear norm targets the
    STRUCTURE of the weight matrix, not its magnitude or quantization noise.
    """
    name = "nuclear_norm"

    # Add nuclear norm computation function and flag before the training loop
    source = _replace_unique(
        source,
        "    # -----------------------------\n"
        "    # MAIN TRAINING LOOP\n"
        "    # -----------------------------",
        "    # Nuclear norm regularization\n"
        "    _nucnorm_w = float(os.environ.get(\"NUCLEAR_NORM_WEIGHT\", \"0.0\"))\n"
        "    _nucnorm_params = matrix_params if _nucnorm_w > 0 else []\n"
        "\n"
        "    # -----------------------------\n"
        "    # MAIN TRAINING LOOP\n"
        "    # -----------------------------",
        name,
    )

    # Add nuclear norm penalty to the loss
    source = _replace_unique(
        source,
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()",
        "            # Nuclear norm regularization on matrix params\n"
        "            if _nucnorm_w > 0 and micro_step == 0:\n"
        "                _nn_penalty = sum(\n"
        "                    torch.linalg.svdvals(p.float()).sum()\n"
        "                    for p in _nucnorm_params\n"
        "                ) / max(len(_nucnorm_params), 1)\n"
        "                loss = loss + _nucnorm_w * _nn_penalty\n"
        "            train_loss += loss.detach()\n"
        "            (loss * grad_scale).backward()",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: weight_perturbation
# Analogy: Langevin dynamics / channel coding — add noise during encoding
#          so the decoder (quantizer) receives a robust signal
# Surface: Landscape geometry (bias toward flat minima)
# Target region: after optimizer step in training loop
# ---------------------------------------------------------------------------

def patch_weight_perturbation(source: str) -> str:
    """Add stochastic perturbation to weights after each optimizer step.

    Injects Gaussian noise scaled by learning rate into all matrix parameters.
    Sharp minima are destabilized by noise (loss increases), while flat minima
    are robust (loss barely changes). Over training, this biases convergence
    toward flatter regions that survive quantization better.

    From Langevin dynamics: SGD + properly scaled noise samples from the
    posterior over weights. Also analogous to channel coding: adding noise
    during encoding (training) makes the signal (weights) robust to the
    noisy channel (quantization).

    Distinct from QAT (quantization-specific noise in forward pass),
    dropout (activation noise), and SAM (explicit sharpness minimization
    via adversarial perturbation). This is simpler: just Gaussian noise
    on weights, scaled by LR so it shrinks with the schedule.
    """
    name = "weight_perturbation"

    # Anchor on the step increment + approximate training time line (unique context)
    source = _replace_unique(
        source,
        "        step += 1\n"
        "        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)",
        "        step += 1\n"
        "\n"
        "        # Stochastic weight perturbation (Langevin-inspired)\n"
        "        _swp_scale = float(os.environ.get(\"WEIGHT_PERTURB_SCALE\", \"0.0\"))\n"
        "        if _swp_scale > 0:\n"
        "            with torch.no_grad():\n"
        "                for p in matrix_params:\n"
        "                    noise = torch.randn_like(p) * (scale * _swp_scale)\n"
        "                    p.add_(noise)\n"
        "\n"
        "        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: grad_centralization
# Analogy: ICCV 2020 (Yong et al.) — remove DC component of gradients,
#          proven in CV, untested for LM training
# Surface: Gradient preprocessing (input to optimizer)
# Target region: before optimizer step in training loop
# ---------------------------------------------------------------------------

def patch_grad_centralization(source: str) -> str:
    """Centralize gradients by subtracting their mean before the optimizer step.

    From "Gradient Centralization" (Yong et al., ICCV 2020). Removes the
    DC component of each gradient tensor, focusing updates on the relative
    structure of the gradient rather than its offset. Proven to improve
    convergence speed and generalization in image classification.

    Never tested for language model training. The hypothesis: in LMs,
    the mean gradient component represents a global shift that all neurons
    want to make (e.g., all weights want to grow or shrink together).
    Removing it forces the optimizer to focus on differential updates
    between neurons, which may be more informative.

    Applied only to 2D matrix params (same as Muon). Not applied to
    embeddings or scalars where the mean may carry important signal.
    """
    name = "grad_centralization"

    source = _replace_unique(
        source,
        "        if args.grad_clip_norm > 0:\n"
        "            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n"
        "        for opt in optimizers:\n"
        "            opt.step()",
        "        if args.grad_clip_norm > 0:\n"
        "            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n"
        "        # Gradient centralization (Yong et al., ICCV 2020)\n"
        "        _gc_enabled = os.environ.get(\"GRAD_CENTRALIZATION\", \"0\") == \"1\"\n"
        "        if _gc_enabled:\n"
        "            with torch.no_grad():\n"
        "                for p in matrix_params:\n"
        "                    if p.grad is not None and p.grad.ndim >= 2:\n"
        "                        p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.ndim)), keepdim=True))\n"
        "        for opt in optimizers:\n"
        "            opt.step()",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Registry and apply function
# ---------------------------------------------------------------------------

PATCH_REGISTRY: dict[str, callable] = {
    "zloss": patch_zloss,
    "adaptive_ns_steps": patch_adaptive_ns_steps,
    "nuclear_norm": patch_nuclear_norm,
    "weight_perturbation": patch_weight_perturbation,
    "grad_centralization": patch_grad_centralization,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    """Apply a list of named patches to the source code, in order."""
    for patch_name in patch_names:
        if patch_name not in PATCH_REGISTRY:
            raise ValueError(
                f"Unknown patch: '{patch_name}'. Available: {sorted(PATCH_REGISTRY)}"
            )
        source = PATCH_REGISTRY[patch_name](source)
    return source


def list_patches() -> list[str]:
    """Return sorted list of available patch names."""
    return sorted(PATCH_REGISTRY)
