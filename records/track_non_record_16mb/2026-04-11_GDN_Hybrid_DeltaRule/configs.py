"""Model architecture configurations for Direction-5 (GDN-Hybrid + RLS).

Model D is our primary target:
  GDN×5 → shared SWA → GDN×5 → shared SWA (Griffin-style)
  dim=512, qk_gain_init=5.0, bigram 3072×112 + trigram

Model J (Phase 4 if needed):
  12-layer GDN, dim=480, KV-sharing stride=2 (GDN-native depth recurrence)

All models sized to fit ~16MB at int6+zstd-22.
"""
from __future__ import annotations


def model_a_pure_gdn() -> dict:
    """Model A: Pure GDN (Baseline) — 10 layers Gated DeltaNet."""
    return dict(
        arch_name="A_PureGDN",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        qk_gain_init=1.5,
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_d_gdn_hybrid() -> dict:
    """Model D: GDN + Shared SWA — our Direction-5 primary target.

    Architecture: [GDN×5] → [SWA] → [GDN×5] → [SWA_shared]
    This is Griffin-style: interleaved recurrent + local attention with weight sharing.

    Key changes vs PR #1370 Model D:
    - qk_gain_init=5.0 (stronger initial attention, following competition SOTA)
    - bigram_vocab_size=3072, bigram_dim=112, trigram=True (matches Model A best setting)
    - swa_window=512 (standard; can increase for Phase 4)
    """
    return dict(
        arch_name="D_GDN_Hybrid",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=1,
        swa_shared=True,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        qk_gain_init=5.0,          # Direction-5: strong initial attention gain
        meta_tokens=0,
        # Layout: [GDN×5] → [SWA] → [GDN×5] → [SWA_shared]
        layer_layout="gdn5_swa_gdn5_swa_shared",
        bigram_vocab_size=3072,    # Direction-5: full bigram table (vs 2048 default)
        bigram_dim=112,            # Direction-5: matches Model A best setting
        trigram=True,              # Direction-5: add trigram for extra context
    )


def model_d_smoke() -> dict:
    """Model D Smoke: Same architecture, smaller for CPU sanity checks."""
    cfg = model_d_gdn_hybrid()
    cfg["arch_name"] = "D_Smoke"
    cfg["model_dim"] = 128
    cfg["num_heads"] = 4
    cfg["gdn_head_dim"] = 32
    cfg["swa_num_kv_heads"] = 2
    cfg["bigram_vocab_size"] = 512
    cfg["bigram_dim"] = 64
    return cfg


def model_j_kv_share() -> dict:
    """Model J: 12-layer GDN + KV-sharing (Phase 4 depth ablation).

    GDN-native equivalent of transformer depth recurrence.
    Adjacent GDN layer pairs share K/V projections, freeing ~528K params per pair.
    Those params → more layers (12 vs 10) at narrower dim (480 vs 512).

    Note: KV-sharing in GDN requires architectures.py to support it.
    This config is defined here for planning; implement in Phase 4 if needed.
    """
    return dict(
        arch_name="J_GDN_KVShare",
        num_gdn_layers=12,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=480,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=60,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        qk_gain_init=1.5,
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
        kv_share_stride=2,  # share K/V every 2 GDN layers (not yet implemented)
    )


ALL_CONFIGS = {
    "A": model_a_pure_gdn,
    "D": model_d_gdn_hybrid,
    "D_smoke": model_d_smoke,
    "J": model_j_kv_share,
}


def get_config(model_id: str) -> dict:
    """Get config by model ID (A, D, D_smoke, J)."""
    if model_id not in ALL_CONFIGS:
        raise ValueError(f"Unknown model ID '{model_id}'. Choose from {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[model_id]()
