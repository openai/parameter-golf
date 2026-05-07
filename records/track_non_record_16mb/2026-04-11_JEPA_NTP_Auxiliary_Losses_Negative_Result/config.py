"""
Experiment Configuration
=========================

Defines the 5 experiment configurations discussed in the research plan:

  Baseline   — Pure NTP (L_CE only), no auxiliary losses
  Exp 1      — Spectral Variance Floor only (anti-collapse ablation)
  Exp 2      — Cosine-MSE Auxiliary Head only (latent prediction ablation)
  Exp 3      — Combined (Exp 1 + Exp 2) — mirrors LeWM's two-term structure
  Exp 4      — Layer-Targeted (Exp 3 scoped to middle layers only)

The compound loss formula for Exp 3/4:
    L = L_CE + α · L_cosine-MSE(ĥ_{t+1}, sg(h_{t+1})) + λ · L_spec(Δh_t)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JEPAConfig:
    """Configuration for JEPA-NTP auxiliary losses."""

    # --- Experiment toggles ---
    use_spectral: bool = False          # Enable spectral variance floor loss
    use_cosine_mse: bool = False        # Enable cosine-MSE prediction loss

    # --- Loss weights ---
    alpha: float = 0.1                  # Weight for cosine-MSE loss
    lambda_spec: float = 0.01          # Weight for spectral floor loss
    spectral_eps: float = 0.01         # Eigenvalue floor threshold
    spectral_use_deltas: bool = True   # Apply to Δh (True) or h (False)

    # --- Alpha decay schedule ---
    alpha_decay: bool = True            # Decay alpha over training
    alpha_warmup_frac: float = 0.1     # Fraction of steps before decay starts
    alpha_min: float = 0.01            # Minimum alpha after decay

    # --- Layer targeting ---
    target_layers: Optional[list[int]] = None  # Which layers to hook for aux losses
    # If None, uses middle layers (25-60% of model depth) by default

    # --- Predictor architecture ---
    predictor_hidden_mult: int = 2     # Hidden dim multiplier for predictor MLP
    predictor_dropout: float = 0.1     # Dropout in predictor head

    # --- Logging ---
    log_metrics_every: int = 200       # Steps between metric logging
    log_spectrum: bool = True          # Log full singular value spectrum
    spectrum_top_k: int = 64           # Number of SVs to log

    # --- WandB ---
    use_wandb: bool = True             # Enable WandB logging
    wandb_project: str = "jepa-ntp-parameter-golf"
    wandb_group: str = ""              # Set per experiment
    wandb_tags: list[str] = field(default_factory=list)

    def get_alpha(self, step: int, total_steps: int) -> float:
        """Compute decayed alpha for current step."""
        if not self.alpha_decay:
            return self.alpha
        warmup_steps = int(total_steps * self.alpha_warmup_frac)
        if step < warmup_steps:
            return self.alpha
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / max(decay_steps, 1)
        # Logarithmic decay
        import math
        decay = 1.0 - math.log1p(progress * (math.e - 1))
        decay = max(decay, 0.0)
        return max(self.alpha_min, self.alpha * decay)


# ============================
# Pre-defined experiment configs
# ============================

def baseline_config() -> JEPAConfig:
    """Exp 0: Pure NTP baseline — no auxiliary losses."""
    return JEPAConfig(
        use_spectral=False,
        use_cosine_mse=False,
        use_wandb=True,
        wandb_group="baseline",
        wandb_tags=["baseline", "pure-ntp"],
    )


def exp1_spectral_config() -> JEPAConfig:
    """Exp 1: Spectral Variance Floor only — pure anti-collapse ablation."""
    return JEPAConfig(
        use_spectral=True,
        use_cosine_mse=False,
        lambda_spec=0.01,
        spectral_eps=0.01,
        use_wandb=True,
        wandb_group="exp1_spectral",
        wandb_tags=["exp1", "spectral-floor", "anti-collapse"],
    )


def exp2_cosine_mse_config() -> JEPAConfig:
    """Exp 2: Cosine-MSE Auxiliary Head only — latent prediction ablation."""
    return JEPAConfig(
        use_spectral=False,
        use_cosine_mse=True,
        alpha=0.1,
        alpha_decay=True,
        use_wandb=True,
        wandb_group="exp2_cosine_mse",
        wandb_tags=["exp2", "cosine-mse", "latent-prediction"],
    )


def exp3_combined_config() -> JEPAConfig:
    """Exp 3: Full Two-Term JEPA-NTP — mirrors LeWM's structure."""
    return JEPAConfig(
        use_spectral=True,
        use_cosine_mse=True,
        alpha=0.1,
        lambda_spec=0.01,
        alpha_decay=True,
        use_wandb=True,
        wandb_group="exp3_combined",
        wandb_tags=["exp3", "combined", "jepa-ntp", "two-term"],
    )


def exp4_targeted_config(num_layers: int = 9) -> JEPAConfig:
    """Exp 4: Layer-Targeted — auxiliary losses scoped to middle layers."""
    # Target layers at 25-60% of model depth
    start = max(1, int(num_layers * 0.25))
    end = min(num_layers - 1, int(num_layers * 0.60))
    layers = list(range(start, end + 1))
    return JEPAConfig(
        use_spectral=True,
        use_cosine_mse=True,
        alpha=0.1,
        lambda_spec=0.01,
        alpha_decay=True,
        target_layers=layers,
        use_wandb=True,
        wandb_group="exp4_targeted",
        wandb_tags=["exp4", "layer-targeted", "scoped"],
    )


EXPERIMENT_CONFIGS = {
    "baseline": baseline_config,
    "exp1_spectral": exp1_spectral_config,
    "exp2_cosine_mse": exp2_cosine_mse_config,
    "exp3_combined": exp3_combined_config,
    "exp4_targeted": exp4_targeted_config,
}
