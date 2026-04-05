"""Model architecture configurations for GDN Hybrid experiments.

Each config returns a dict consumed by HybridGDN.__init__.
All models are sized to fit ~16MB at int6+zstd-22.

Models A-H: baseline architecture sweeps.
Models I-K: KV sharing experiments (kv_sharing_stride=2).
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
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_b_deltaproduct() -> dict:
    """Model B: Gated DeltaProduct n_h=2 — rank-2 state transitions."""
    return dict(
        arch_name="B_DeltaProduct",
        num_gdn_layers=10,  # 10 layers to fit param budget
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=480,  # slightly narrower to fit 16MB
        num_heads=8,
        mlp_mult=3.0,
        use_deltaproduct=True,
        dp_num_householder=2,
        dp_allow_neg_eigval=False,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
    )


def model_b2_deltaproduct_neg() -> dict:
    """Model B2: DeltaProduct + negative eigenvalues."""
    cfg = model_b_deltaproduct()
    cfg["arch_name"] = "B2_DeltaProduct_NegEig"
    cfg["dp_allow_neg_eigval"] = True
    return cfg


def model_c_gdn_neg() -> dict:
    """Model C: GDN with negative eigenvalues — richer state dynamics.

    (Originally RWKV-7, replaced because RWKV7 requires Triton kernels with
    no pure-PyTorch fallback available.)
    """
    return dict(
        arch_name="C_GDN_NegEig",
        num_gdn_layers=11,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=True,  # Key difference: negative eigenvalues
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
    )


def model_d_gdn_1swa() -> dict:
    """Model D: GDN + 1 Shared SWA (Zamba-style)."""
    return dict(
        arch_name="D_GDN_1SWA",
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
        meta_tokens=0,
        # Layout: [GDN×5] → [SWA] → [GDN×5] → [SWA_shared]
        layer_layout="gdn5_swa_gdn5_swa_shared",
    )


def model_e_gdn_2swa() -> dict:
    """Model E: GDN + 2 Shared SWA (Hymba-inspired) with meta-tokens."""
    return dict(
        arch_name="E_GDN_2SWA_Hymba",
        num_gdn_layers=9,
        num_mamba_layers=0,
        num_swa_layers=1,  # 1 unique, shared at 2 positions
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
        meta_tokens=4,  # Hymba-style prepended meta-tokens
        # Layout: [GDN×3] → [SWA] → [GDN×3] → [SWA_shared] → [GDN×3]
        layer_layout="gdn3_swa_gdn3_swa_shared_gdn3",
    )


def model_f_mamba2() -> dict:
    """Model F: Mamba-2 Pure (Mamba-3 proxy with RoPE on B/C)."""
    return dict(
        arch_name="F_Mamba2",
        num_gdn_layers=0,
        num_mamba_layers=11,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        mamba_state_size=64,
        mamba_expand=2,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="mamba_only",
    )


def model_g_hybrid() -> dict:
    """Model G: GDN + Mamba-2 + SWA triple hybrid."""
    return dict(
        arch_name="G_GDN_Mamba_SWA",
        num_gdn_layers=6,
        num_mamba_layers=4,
        num_swa_layers=1,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        mamba_state_size=64,
        mamba_expand=2,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        # Layout: [GDN×3] → [Mamba×2] → [SWA] → [GDN×3] → [Mamba×2]
        layer_layout="gdn3_mamba2_swa_gdn3_mamba2",
    )


def model_h_pure_swa() -> dict:
    """Model H: Pure Sliding Window Attention (standard softmax) — control baseline.

    All 10 layers use causal sliding-window softmax attention (no GDN).
    Same MLP, embedding, and normalization as Model A for fair comparison.
    """
    return dict(
        arch_name="H_PureSWA",
        num_gdn_layers=0,
        num_mamba_layers=0,
        num_swa_layers=10,
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
        meta_tokens=0,
        layer_layout="swa_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_i_kv_share() -> dict:
    """Model I: GDN + KV Share — same as A but with kv_sharing_stride=2."""
    return dict(
        arch_name="I_KVShare",
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
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
        kv_sharing_stride=2,
    )


def model_j_kv_share_deeper() -> dict:
    """Model J: GDN + KV Share + Deeper — 12L dim=480, near iso-parameter to A."""
    return dict(
        arch_name="J_KVShare_Deeper",
        num_gdn_layers=12,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=480,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
        kv_sharing_stride=2,
    )


def model_k_kv_share_wider() -> dict:
    """Model K: GDN + KV Share + Wider — 10L dim=544, iso-parameter to A."""
    return dict(
        arch_name="K_KVShare_Wider",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=544,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
        kv_sharing_stride=2,
    )


ALL_CONFIGS = {
    "A": model_a_pure_gdn,
    "B": model_b_deltaproduct,
    "B2": model_b2_deltaproduct_neg,
    "C": model_c_gdn_neg,
    "D": model_d_gdn_1swa,
    "E": model_e_gdn_2swa,
    "F": model_f_mamba2,
    "G": model_g_hybrid,
    "H": model_h_pure_swa,
    "I": model_i_kv_share,
    "J": model_j_kv_share_deeper,
    "K": model_k_kv_share_wider,
}


def get_config(model_id: str) -> dict:
    """Get config by model ID (A, B, B2, C, D, E, F, G, H, I, J, K)."""
    if model_id not in ALL_CONFIGS:
        raise ValueError(f"Unknown model ID '{model_id}'. Choose from {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[model_id]()
