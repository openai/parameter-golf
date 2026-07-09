"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

try:
    from convextok import ConvexTokTokenizer, PRICED_OFFSET
except Exception:  # pragma: no cover - training paths fail loudly when needed.
    ConvexTokTokenizer = None  # type: ignore[assignment]
    PRICED_OFFSET = 260

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Campaign runs can skip the step-0
    # validation and cap the gate set, while final record runs can leave these
    # at defaults for full validation.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    eval_initial = bool(int(os.environ.get("EVAL_INITIAL", "1")))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", "0"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    main_activation_checkpoint = bool(int(os.environ.get("MAIN_ACTIVATION_CHECKPOINT", "0")))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # First-class FineWeb graphification.  This is intentionally separate from
    # the ToricGT sidecar: the main BPB model itself receives TokenGT-style
    # graph-structural features by default.  Disable with FINEWEB_GRAPHIFY=0.
    fineweb_graphify = bool(int(os.environ.get("FINEWEB_GRAPHIFY", "1")))
    tokengt_first_class = bool(
        int(os.environ.get("TOKENGT_FIRST_CLASS", "1" if fineweb_graphify else "0"))
    ) and fineweb_graphify
    tokengt_first_class_lr = float(os.environ.get("TOKENGT_FIRST_CLASS_LR", 2e-4))
    tokengt_graph_radius = int(os.environ.get("TOKENGT_GRAPH_RADIUS", 4))
    tokengt_distance_features = os.environ.get("TOKENGT_DISTANCE_FEATURES", "functional").strip().lower()
    tokengt_token_class_buckets = int(os.environ.get("TOKENGT_TOKEN_CLASS_BUCKETS", 64))
    tokengt_position_buckets = int(os.environ.get("TOKENGT_POSITION_BUCKETS", 256))
    tokengt_structural_weight = float(os.environ.get("TOKENGT_STRUCTURAL_WEIGHT", 0.050))
    tokengt_edge_weight = float(os.environ.get("TOKENGT_EDGE_WEIGHT", 0.035))
    tokengt_torus_weight = float(os.environ.get("TOKENGT_TORUS_WEIGHT", 0.015))
    tokengt_identifier_dim = int(os.environ.get("TOKENGT_IDENTIFIER_DIM", 24))
    tokengt_identifier_weight = float(os.environ.get("TOKENGT_IDENTIFIER_WEIGHT", 0.014))
    tokengt_endpoint_weight = float(os.environ.get("TOKENGT_ENDPOINT_WEIGHT", 0.018))
    tokengt_edge_token_weight = float(os.environ.get("TOKENGT_EDGE_TOKEN_WEIGHT", 0.016))
    convextok_dag_features = bool(int(os.environ.get("CONVEXTOK_DAG_FEATURES", "1")))
    convextok_dag_feature_weight = float(os.environ.get("CONVEXTOK_DAG_FEATURE_WEIGHT", 0.018))
    convextok_toric_reg_weight = float(os.environ.get("CONVEXTOK_TORIC_REG_WEIGHT", "0"))
    convextok_toric_reg_topk = int(os.environ.get("CONVEXTOK_TORIC_REG_TOPK", "96"))
    graph_output_flattening = bool(
        int(
            os.environ.get(
                "OAI_FINEWEB_OUTPUT_FLATTENING",
                os.environ.get("GRAPH_OUTPUT_FLATTENING", "1" if fineweb_graphify else "0"),
            )
        )
    )
    graph_output_flattening_lr = float(os.environ.get("GRAPH_OUTPUT_FLATTENING_LR", tokengt_first_class_lr))
    graph_output_edge_radius = int(os.environ.get("GRAPH_OUTPUT_EDGE_RADIUS", tokengt_graph_radius))
    graph_output_distance_features = os.environ.get(
        "GRAPH_OUTPUT_DISTANCE_FEATURES",
        tokengt_distance_features,
    ).strip().lower()
    graph_output_node_weight = float(os.environ.get("GRAPH_OUTPUT_NODE_WEIGHT", 0.05))
    graph_output_edge_weight = float(os.environ.get("GRAPH_OUTPUT_EDGE_WEIGHT", 0.05))
    graph_output_virtual_edge_tokens = bool(int(os.environ.get("GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS", "1")))
    graph_output_edge_token_weight = float(os.environ.get("GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT", 0.035))
    graph_output_score_correction = bool(int(os.environ.get("GRAPH_OUTPUT_SCORE_CORRECTION", "1")))
    graph_output_score_correction_weight = float(os.environ.get("GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT", 0.024))
    graph_output_calibration_loss_weight = float(os.environ.get("GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT", 0.0))
    graph_output_calibration_margin = float(os.environ.get("GRAPH_OUTPUT_CALIBRATION_MARGIN", 0.0))
    graph_output_calibration_every = int(os.environ.get("GRAPH_OUTPUT_CALIBRATION_EVERY", 25))
    graph_output_calibration_max_sequences = int(os.environ.get("GRAPH_OUTPUT_CALIBRATION_MAX_SEQUENCES", 2))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Bonafide graph LM stream plus ToricGT sidecar.  FineWeb remains the
    # score-defining BPB stream; graph Parquet rows are also graphified into
    # autoregressive node/edge token records and trained directly on the base
    # LM with a small primary-mixture weight.
    toricgt_sidecar = bool(int(os.environ.get("TORICGT_SIDECAR", "0")))
    require_toricgt_sidecar = bool(int(os.environ.get("REQUIRE_TORICGT_SIDECAR", "0")))
    graph_data_path = os.environ.get(
        "GRAPH_DATA_PATH",
        "/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards",
    )
    graph_train_glob = os.environ.get("GRAPH_TRAIN_GLOB", os.path.join(graph_data_path, "train", "*.parquet"))
    late_graph_train_glob = os.environ.get("LATE_GRAPH_TRAIN_GLOB", "").strip()
    late_graph_start_step = int(os.environ.get("LATE_GRAPH_START_STEP", max(1, int(0.70 * iterations))))
    late_graph_mix_ratio = float(os.environ.get("LATE_GRAPH_MIX_RATIO", "1.0"))
    late_graph_upsample_passes = int(os.environ.get("LATE_GRAPH_UPSAMPLE_PASSES", "3"))
    toricblm_structure_readiness_manifest = os.environ.get("TORICBLM_STRUCTURE_READINESS_MANIFEST", "").strip()
    toricblm_structure_flow = bool(int(os.environ.get("TORICBLM_STRUCTURE_FLOW", "0")))
    toricblm_structure_train_glob = os.environ.get("TORICBLM_STRUCTURE_TRAIN_GLOB", "").strip()
    toricblm_structure_seq_len = int(os.environ.get("TORICBLM_STRUCTURE_SEQ_LEN", min(train_seq_len, 512)))
    toricblm_structure_batch_size = int(os.environ.get("TORICBLM_STRUCTURE_BATCH_SIZE", "2"))
    toricblm_structure_every = int(os.environ.get("TORICBLM_STRUCTURE_EVERY", "4"))
    toricblm_structure_max_atoms = int(os.environ.get("TORICBLM_STRUCTURE_MAX_ATOMS", "128"))
    toricblm_structure_head_hidden_dim = int(os.environ.get("TORICBLM_STRUCTURE_HEAD_HIDDEN_DIM", "512"))
    toricblm_structure_lr = float(os.environ.get("TORICBLM_STRUCTURE_LR", "0.00012"))
    toricblm_structure_flow_weight = float(os.environ.get("TORICBLM_STRUCTURE_FLOW_WEIGHT", 0.0))
    toricblm_structure_contact_weight = float(os.environ.get("TORICBLM_STRUCTURE_CONTACT_WEIGHT", 0.0))
    toricblm_structure_distogram_weight = float(os.environ.get("TORICBLM_STRUCTURE_DISTOGRAM_WEIGHT", 0.0))
    toricblm_structure_frame_weight = float(os.environ.get("TORICBLM_STRUCTURE_FRAME_WEIGHT", 0.0))
    toricblm_structure_flow_start_step = int(os.environ.get("TORICBLM_STRUCTURE_FM_START_STEP", late_graph_start_step))
    graph_lm_primary = bool(int(os.environ.get("GRAPH_LM_PRIMARY", "1" if toricgt_sidecar else "0")))
    require_graph_lm_primary = bool(int(os.environ.get("REQUIRE_GRAPH_LM_PRIMARY", "0")))
    graph_lm_seq_len = int(os.environ.get("GRAPH_LM_SEQ_LEN", train_seq_len))
    graph_lm_batch_size = int(os.environ.get("GRAPH_LM_BATCH_SIZE", 2))
    graph_lm_every = int(os.environ.get("GRAPH_LM_EVERY", 1))
    graph_lm_loss_weight = float(os.environ.get("GRAPH_LM_LOSS_WEIGHT", 0.08))
    graph_lm_loss_weight_start = float(
        os.environ.get("GRAPH_LM_LOSS_WEIGHT_START", min(0.025, graph_lm_loss_weight))
    )
    graph_lm_warmup_steps = int(os.environ.get("GRAPH_LM_WARMUP_STEPS", 350))
    graph_lm_hold_steps = int(os.environ.get("GRAPH_LM_HOLD_STEPS", 25))

    # Complete-entry graph/FoT training.  This path keeps row boundaries and
    # trains all segments of long graphified records instead of losing them in a
    # fixed-size token stream.  It is separate from the BPB FineWeb stream so
    # full biological/FoT entries can be learned while validation stays the OAI
    # challenge BPB objective.
    long_entry_training = bool(int(os.environ.get("LONG_ENTRY_TRAINING", "0")))
    long_entry_train_glob = os.environ.get("LONG_ENTRY_TRAIN_GLOB", late_graph_train_glob or graph_train_glob).strip()
    long_entry_every = int(os.environ.get("LONG_ENTRY_EVERY", "8"))
    long_entry_batch_size = int(os.environ.get("LONG_ENTRY_BATCH_SIZE", "1"))
    long_entry_max_tokens = int(os.environ.get("LONG_ENTRY_MAX_TOKENS", "24576"))
    long_entry_segment_len = int(os.environ.get("LONG_ENTRY_SEGMENT_LEN", str(train_seq_len)))
    long_entry_segment_stride = int(os.environ.get("LONG_ENTRY_SEGMENT_STRIDE", str(long_entry_segment_len)))
    long_entry_max_segments = int(os.environ.get("LONG_ENTRY_MAX_SEGMENTS", "24"))
    long_entry_loss_weight = float(os.environ.get("LONG_ENTRY_LOSS_WEIGHT", "0.020"))
    long_entry_next_segment_weight = float(os.environ.get("LONG_ENTRY_NEXT_SEGMENT_WEIGHT", "0.010"))
    long_entry_boundary_weight = float(os.environ.get("LONG_ENTRY_BOUNDARY_WEIGHT", "0.004"))
    long_entry_segment_coverage_weight = float(os.environ.get("LONG_ENTRY_SEGMENT_COVERAGE_WEIGHT", "0.002"))
    long_entry_fot_loss_weight = float(os.environ.get("LONG_ENTRY_FOT_LOSS_WEIGHT", "0.0"))
    long_entry_fot_every = int(os.environ.get("LONG_ENTRY_FOT_EVERY", os.environ.get("LONG_ENTRY_EVERY", "8")))
    long_entry_fot_max_segments = int(os.environ.get("LONG_ENTRY_FOT_MAX_SEGMENTS", "2"))
    long_entry_full_context_every = int(os.environ.get("LONG_ENTRY_FULL_CONTEXT_EVERY", "64"))
    long_entry_full_context_seq_len = int(os.environ.get("LONG_ENTRY_FULL_CONTEXT_SEQ_LEN", "8192"))
    long_entry_full_context_loss_weight = float(os.environ.get("LONG_ENTRY_FULL_CONTEXT_LOSS_WEIGHT", "0.006"))
    long_entry_full_context_min_tokens = int(os.environ.get("LONG_ENTRY_FULL_CONTEXT_MIN_TOKENS", "2048"))
    long_entry_segment_checkpoint = bool(int(os.environ.get("LONG_ENTRY_SEGMENT_CHECKPOINT", "0")))
    long_entry_full_context_checkpoint = bool(int(os.environ.get("LONG_ENTRY_FULL_CONTEXT_CHECKPOINT", "1")))
    aux_grad_routing = bool(int(os.environ.get("AUX_GRAD_ROUTING", "1")))
    aux_grad_route_graph_lm = bool(int(os.environ.get("AUX_GRAD_ROUTE_GRAPH_LM", "1")))
    aux_grad_route_sidecar = bool(int(os.environ.get("AUX_GRAD_ROUTE_SIDECAR", "1")))
    aux_grad_route_teacher = bool(int(os.environ.get("AUX_GRAD_ROUTE_TEACHER", "1")))
    aux_grad_route_structure = bool(int(os.environ.get("AUX_GRAD_ROUTE_STRUCTURE", "1")))
    aux_grad_route_long_entry = bool(int(os.environ.get("AUX_GRAD_ROUTE_LONG_ENTRY", "1")))
    aux_grad_conflict_projection = bool(int(os.environ.get("AUX_GRAD_CONFLICT_PROJECTION", "1")))
    aux_grad_aligned_boost = float(os.environ.get("AUX_GRAD_ALIGNED_BOOST", 1.0))
    bpb_first_aux_staging = bool(int(os.environ.get("BPB_FIRST_AUX_STAGING", "1")))
    bpb_first_core_steps = int(os.environ.get("BPB_FIRST_CORE_STEPS", 250))
    bpb_first_ramp_steps = int(os.environ.get("BPB_FIRST_RAMP_STEPS", 500))
    bpb_first_min_aux_mult = float(os.environ.get("BPB_FIRST_MIN_AUX_MULT", 0.02))
    bpb_first_require_negative_slope = bool(int(os.environ.get("BPB_FIRST_REQUIRE_NEGATIVE_SLOPE", "1")))
    bpb_first_curvature_guard = bool(int(os.environ.get("BPB_FIRST_CURVATURE_GUARD", "1")))
    bpb_first_bad_slope_mult = float(os.environ.get("BPB_FIRST_BAD_SLOPE_MULT", 0.25))
    bpb_first_bad_curvature_mult = float(os.environ.get("BPB_FIRST_BAD_CURVATURE_MULT", 0.35))
    aux_conflict_controller = bool(int(os.environ.get("AUX_CONFLICT_CONTROLLER", "1")))
    aux_conflict_damp_min = float(os.environ.get("AUX_CONFLICT_DAMP_MIN", 0.10))
    aux_conflict_boost_max = float(os.environ.get("AUX_CONFLICT_BOOST_MAX", 1.35))
    teacher_checkpoint = os.environ.get("TEACHER_CHECKPOINT", "")
    teacher_distill_weight = float(os.environ.get("TEACHER_DISTILL_WEIGHT", 0.0))
    teacher_distill_weight_start = float(os.environ.get("TEACHER_DISTILL_WEIGHT_START", teacher_distill_weight))
    teacher_distill_until_step = int(os.environ.get("TEACHER_DISTILL_UNTIL_STEP", 0))
    teacher_distill_decay_steps = int(os.environ.get("TEACHER_DISTILL_DECAY_STEPS", 0))
    teacher_distill_temperature = float(os.environ.get("TEACHER_DISTILL_TEMPERATURE", 2.0))
    teacher_distill_max_sequences = int(os.environ.get("TEACHER_DISTILL_MAX_SEQUENCES", 2))
    fineweb_caseops = bool(int(os.environ.get("FINEWEB_CASEOPS", "0")))
    score_first_tta = bool(int(os.environ.get("SCORE_FIRST_TTA", "0")))
    score_first_tta_steps = int(os.environ.get("SCORE_FIRST_TTA_STEPS", 0))
    score_first_tta_lr = float(os.environ.get("SCORE_FIRST_TTA_LR", 0.0))
    score_first_tta_commit = bool(int(os.environ.get("SCORE_FIRST_TTA_COMMIT", "0")))
    oai_gflownet = bool(int(os.environ.get("OAI_GFLOWNET", "0")))
    oai_gflownet_lr = float(os.environ.get("OAI_GFLOWNET_LR", 2e-4))
    oai_gflownet_every = int(os.environ.get("OAI_GFLOWNET_EVERY", 1))
    oai_gflownet_loss_weight = float(os.environ.get("OAI_GFLOWNET_LOSS_WEIGHT", 0.0))
    oai_gflownet_entropy_weight = float(os.environ.get("OAI_GFLOWNET_ENTROPY_WEIGHT", 0.0))
    oai_gflownet_entropy_target = float(os.environ.get("OAI_GFLOWNET_ENTROPY_TARGET", 1.8))
    oai_gflownet_num_actions = int(os.environ.get("OAI_GFLOWNET_NUM_ACTIONS", 16))
    oai_gflownet_hidden_dim = int(os.environ.get("OAI_GFLOWNET_HIDDEN_DIM", 192))
    oai_gflownet_max_sequences = int(os.environ.get("OAI_GFLOWNET_MAX_SEQUENCES", 2))
    oai_gflownet_max_positions = int(os.environ.get("OAI_GFLOWNET_MAX_POSITIONS", 192))
    oai_embedding_fot = bool(int(os.environ.get("OAI_EMBEDDING_FOT", "0")))
    oai_fot_lr = float(os.environ.get("OAI_FOT_LR", 1.5e-4))
    oai_fot_every = int(os.environ.get("OAI_FOT_EVERY", 1))
    oai_fot_loss_weight = float(os.environ.get("OAI_FOT_LOSS_WEIGHT", 0.0))
    oai_fot_num_trees = int(os.environ.get("OAI_FOT_NUM_TREES", 4))
    oai_fot_max_depth = int(os.environ.get("OAI_FOT_MAX_DEPTH", 5))
    oai_fot_branching = int(os.environ.get("OAI_FOT_BRANCHING", 4))
    oai_fot_topk_trees = int(os.environ.get("OAI_FOT_TOPK_TREES", 2))
    oai_fot_hidden_dim = int(os.environ.get("OAI_FOT_HIDDEN_DIM", 192))
    oai_fot_max_sequences = int(os.environ.get("OAI_FOT_MAX_SEQUENCES", 2))
    oai_fot_max_positions = int(os.environ.get("OAI_FOT_MAX_POSITIONS", 192))
    oai_fot_consensus_buckets = int(os.environ.get("OAI_FOT_CONSENSUS_BUCKETS", 64))
    oai_fot_correction_scale = float(os.environ.get("OAI_FOT_CORRECTION_SCALE", 0.08))
    oai_fot_ucb_exploration = float(os.environ.get("OAI_FOT_UCB_EXPLORATION", 1.25))
    oai_fot_temperature = float(os.environ.get("OAI_FOT_TEMPERATURE", 0.70))
    oai_fot_sparse_weight = float(os.environ.get("OAI_FOT_SPARSE_WEIGHT", 1.0))
    oai_fot_ucb_weight = float(os.environ.get("OAI_FOT_UCB_WEIGHT", 0.5))
    oai_fot_correction_weight = float(os.environ.get("OAI_FOT_CORRECTION_WEIGHT", 0.5))
    oai_fot_consensus_weight = float(os.environ.get("OAI_FOT_CONSENSUS_WEIGHT", 0.75))
    oai_fot_tb_weight = float(os.environ.get("OAI_FOT_TB_WEIGHT", 1.0))
    oai_fot_subtb_weight = float(os.environ.get("OAI_FOT_SUBTB_WEIGHT", 0.0))
    oai_fot_complexity_weight = float(os.environ.get("OAI_FOT_COMPLEXITY_WEIGHT", 0.05))
    oai_fot_reward_advanced_bonus = float(os.environ.get("OAI_FOT_REWARD_ADVANCED_BONUS", 0.0))
    oai_fot_reward_mode = os.environ.get("OAI_FOT_REWARD_MODE", "bpb_delta")
    oai_fot_bpb_delta_weight = float(os.environ.get("OAI_FOT_BPB_DELTA_WEIGHT", 1.0))
    oai_fot_reward_graph_weight = float(os.environ.get("OAI_FOT_REWARD_GRAPH_WEIGHT", 0.10))
    oai_fot_reward_consensus_weight = float(os.environ.get("OAI_FOT_REWARD_CONSENSUS_WEIGHT", 0.20))
    oai_fot_reward_complexity_weight = float(os.environ.get("OAI_FOT_REWARD_COMPLEXITY_WEIGHT", 0.02))
    oai_fot_reward_floor = float(os.environ.get("OAI_FOT_REWARD_FLOOR", 1.0e-4))
    oai_fot_adaptive_control = bool(int(os.environ.get("OAI_FOT_ADAPTIVE_CONTROL", "1")))
    oai_fot_reward_target = float(os.environ.get("OAI_FOT_REWARD_TARGET", 0.08))
    oai_fot_diversity_target = float(os.environ.get("OAI_FOT_DIVERSITY_TARGET", 0.12))
    oai_fot_entropy_high = float(os.environ.get("OAI_FOT_ENTROPY_HIGH", 0.985))
    oai_fot_entropy_low = float(os.environ.get("OAI_FOT_ENTROPY_LOW", 0.75))
    oai_fot_temp_min = float(os.environ.get("OAI_FOT_TEMP_MIN", 0.45))
    oai_fot_temp_max = float(os.environ.get("OAI_FOT_TEMP_MAX", 1.10))
    oai_fot_ucb_min = float(os.environ.get("OAI_FOT_UCB_MIN", 0.40))
    oai_fot_ucb_max = float(os.environ.get("OAI_FOT_UCB_MAX", 1.80))
    oai_mtp = bool(int(os.environ.get("OAI_MTP", "0")))
    oai_mtp_lr = float(os.environ.get("OAI_MTP_LR", tokengt_first_class_lr))
    oai_mtp_every = int(os.environ.get("OAI_MTP_EVERY", 1))
    oai_mtp_loss_weight = float(os.environ.get("OAI_MTP_LOSS_WEIGHT", 0.0))
    oai_mtp_offsets = os.environ.get("OAI_MTP_OFFSETS", "2")
    oai_mtp_max_sequences = int(os.environ.get("OAI_MTP_MAX_SEQUENCES", 2))
    sidecar_seq_len = int(os.environ.get("TORICGT_SIDECAR_SEQ_LEN", min(256, train_seq_len)))
    sidecar_batch_size = int(os.environ.get("TORICGT_SIDECAR_BATCH_SIZE", 8))
    sidecar_every = int(os.environ.get("TORICGT_SIDECAR_EVERY", 1))
    sidecar_lr = float(os.environ.get("TORICGT_SIDECAR_LR", 2e-4))
    sidecar_loss_weight = float(os.environ.get("TORICGT_SIDECAR_LOSS_WEIGHT", 1.0))
    sidecar_loss_weight_start = float(os.environ.get("TORICGT_SIDECAR_LOSS_WEIGHT_START", min(0.05, sidecar_loss_weight)))
    sidecar_warmup_steps = int(os.environ.get("TORICGT_SIDECAR_WARMUP_STEPS", 500))
    sidecar_hold_steps = int(os.environ.get("TORICGT_SIDECAR_HOLD_STEPS", 100))
    graphcg_loss_weight = float(os.environ.get("GRAPHCG_LOSS_WEIGHT", 2e-5))
    analogy_loss_weight = float(os.environ.get("ANALOGY_LOSS_WEIGHT", 2e-5))
    tokengt_graph_loss_weight = float(os.environ.get("TOKENGT_GRAPH_LOSS_WEIGHT", 4e-5))
    trajectory_memory_loss_weight = float(os.environ.get("TRAJECTORY_MEMORY_LOSS_WEIGHT", 2e-5))
    sidecar_compute_all_metrics = int(os.environ.get("TORICGT_SIDECAR_COMPUTE_ALL_METRICS", "1"))
    toric_geometry_loss_weight = float(os.environ.get("TORIC_GEOMETRY_LOSS_WEIGHT", 0.0))
    toric_vector_bundle_loss_weight = float(os.environ.get("TORIC_VECTOR_BUNDLE_LOSS_WEIGHT", 0.0))
    toric_bgg_loss_weight = float(os.environ.get("TORIC_BGG_LOSS_WEIGHT", 0.0))
    koszul_persistence_loss_weight = float(os.environ.get("KOSZUL_PERSISTENCE_LOSS_WEIGHT", 0.0))
    combinatorial_toric_loss_weight = float(os.environ.get("COMBINATORIAL_TORIC_LOSS_WEIGHT", 0.0))
    derived_signature_loss_weight = float(os.environ.get("DERIVED_SIGNATURE_LOSS_WEIGHT", 1e-6))
    advanced_lagrangian_controller = bool(int(os.environ.get("ADVANCED_LAGRANGIAN_CONTROLLER", "1")))
    lagrangian_dual_lr = float(os.environ.get("LAGRANGIAN_DUAL_LR", 0.025))
    lagrangian_decay = float(os.environ.get("LAGRANGIAN_DECAY", 0.985))
    lagrangian_min_multiplier = float(os.environ.get("LAGRANGIAN_MIN_MULTIPLIER", 0.25))
    lagrangian_max_multiplier = float(os.environ.get("LAGRANGIAN_MAX_MULTIPLIER", 2.25))
    lagrangian_bpb_ceiling = float(os.environ.get("LAGRANGIAN_BPB_CEILING", 3.05))
    lagrangian_bpb_softness = float(os.environ.get("LAGRANGIAN_BPB_SOFTNESS", 0.35))
    lagrangian_graphcg_target = float(os.environ.get("LAGRANGIAN_GRAPHCG_TARGET", 0.010))
    lagrangian_analogy_target = float(os.environ.get("LAGRANGIAN_ANALOGY_TARGET", 0.045))
    lagrangian_tokengt_graph_target = float(os.environ.get("LAGRANGIAN_TOKENGT_GRAPH_TARGET", 0.75))
    lagrangian_memory_target = float(os.environ.get("LAGRANGIAN_MEMORY_TARGET", 1.15))
    lagrangian_toric_target = float(os.environ.get("LAGRANGIAN_TORIC_TARGET", 2.0))
    lagrangian_vector_bundle_target = float(os.environ.get("LAGRANGIAN_VECTOR_BUNDLE_TARGET", 0.45))
    lagrangian_bgg_target = float(os.environ.get("LAGRANGIAN_BGG_TARGET", 0.06))
    lagrangian_koszul_target = float(os.environ.get("LAGRANGIAN_KOSZUL_TARGET", 0.035))
    lagrangian_cca_target = float(os.environ.get("LAGRANGIAN_CCA_TARGET", 1.0))
    lagrangian_derived_signature_target = float(os.environ.get("LAGRANGIAN_DERIVED_SIGNATURE_TARGET", 0.025))
    toric_fan_curriculum = bool(int(os.environ.get("TORIC_FAN_CURRICULUM", "1")))
    toric_fan_coarse_steps = int(os.environ.get("TORIC_FAN_COARSE_STEPS", 450))
    toric_fan_intermediate_steps = int(os.environ.get("TORIC_FAN_INTERMEDIATE_STEPS", 950))
    memory_sheaf_gate_min = float(os.environ.get("MEMORY_SHEAF_GATE_MIN", 0.15))
    memory_sheaf_gate_threshold = float(os.environ.get("MEMORY_SHEAF_GATE_THRESHOLD", 0.12))
    memory_sheaf_gate_softness = float(os.environ.get("MEMORY_SHEAF_GATE_SOFTNESS", 0.18))
    memory_sheaf_ce_weight = float(os.environ.get("MEMORY_SHEAF_CE_WEIGHT", 1.0))
    derived_signature_max_vertices = int(os.environ.get("DERIVED_SIGNATURE_MAX_VERTICES", 8))
    derived_signature_max_edges = int(os.environ.get("DERIVED_SIGNATURE_MAX_EDGES", 64))
    toric_geometry_num_exponents = int(os.environ.get("TORIC_GEOMETRY_NUM_EXPONENTS", 16))
    toric_geometry_exponent_dim = int(os.environ.get("TORIC_GEOMETRY_EXPONENT_DIM", 4))
    toric_geometry_probe_rank = int(os.environ.get("TORIC_GEOMETRY_PROBE_RANK", 12))
    toric_geometry_quant_bits = int(os.environ.get("TORIC_GEOMETRY_QUANT_BITS", 6))
    toric_geometry_max_positions = int(os.environ.get("TORIC_GEOMETRY_MAX_POSITIONS", 128))
    toric_vector_bundle_rank = int(os.environ.get("TORIC_VECTOR_BUNDLE_RANK", 8))
    toric_vector_bundle_num_one_dimensional_cones = int(
        os.environ.get("TORIC_VECTOR_BUNDLE_NUM_ONE_DIMENSIONAL_CONES", 8)
    )
    toric_vector_bundle_num_cones = int(os.environ.get("TORIC_VECTOR_BUNDLE_NUM_CONES", 8))
    toric_vector_bundle_filtration_levels = int(os.environ.get("TORIC_VECTOR_BUNDLE_FILTRATION_LEVELS", 3))
    toric_vector_bundle_max_positions = int(os.environ.get("TORIC_VECTOR_BUNDLE_MAX_POSITIONS", 96))
    toric_bgg_num_standard_tokens = int(os.environ.get("TORIC_BGG_NUM_STANDARD_TOKENS", 8))
    toric_bgg_probe_rank = int(os.environ.get("TORIC_BGG_PROBE_RANK", 8))
    toric_bgg_signature_dim = int(os.environ.get("TORIC_BGG_SIGNATURE_DIM", 16))
    toric_bgg_max_positions = int(os.environ.get("TORIC_BGG_MAX_POSITIONS", 64))
    koszul_max_points = int(os.environ.get("KOSZUL_MAX_POINTS", 16))
    koszul_max_windows = int(os.environ.get("KOSZUL_MAX_WINDOWS", 2))
    koszul_window_size = int(os.environ.get("KOSZUL_WINDOW_SIZE", 32))
    koszul_step_stride = int(os.environ.get("KOSZUL_STEP_STRIDE", 16))
    koszul_num_parameters = int(os.environ.get("KOSZUL_NUM_PARAMETERS", 3))
    koszul_temperature = float(os.environ.get("KOSZUL_TEMPERATURE", 0.12))
    koszul_chart_exponents = int(os.environ.get("KOSZUL_CHART_EXPONENTS", 10))
    combinatorial_toric_max_points = int(os.environ.get("COMBINATORIAL_TORIC_MAX_POINTS", 16))
    combinatorial_toric_max_windows = int(os.environ.get("COMBINATORIAL_TORIC_MAX_WINDOWS", 2))
    combinatorial_toric_window_size = int(os.environ.get("COMBINATORIAL_TORIC_WINDOW_SIZE", 32))
    combinatorial_toric_step_stride = int(os.environ.get("COMBINATORIAL_TORIC_STEP_STRIDE", 16))
    combinatorial_toric_num_chambers = int(os.environ.get("COMBINATORIAL_TORIC_NUM_CHAMBERS", 8))
    combinatorial_toric_exponent_dim = int(os.environ.get("COMBINATORIAL_TORIC_EXPONENT_DIM", 4))
    retrieval_conditioned_aux = bool(int(os.environ.get("RETRIEVAL_CONDITIONED_AUX", "1")))
    retrieval_gate_min = float(os.environ.get("RETRIEVAL_GATE_MIN", 0.20))
    retrieval_gate_softness = float(os.environ.get("RETRIEVAL_GATE_SOFTNESS", 0.08))
    retrieval_gate_center = float(os.environ.get("RETRIEVAL_GATE_CENTER", 0.20))
    sidecar_uncertainty_weighting = bool(int(os.environ.get("SIDECAR_UNCERTAINTY_WEIGHTING", "1")))
    sidecar_uncertainty_alpha = float(os.environ.get("SIDECAR_UNCERTAINTY_ALPHA", 0.35))
    sidecar_uncertainty_center = float(os.environ.get("SIDECAR_UNCERTAINTY_CENTER", 2.6))
    sidecar_uncertainty_scale = float(os.environ.get("SIDECAR_UNCERTAINTY_SCALE", 1.0))
    sidecar_uncertainty_max = float(os.environ.get("SIDECAR_UNCERTAINTY_MAX", 2.0))
    graphcg_bpb_orthogonal_weight = float(os.environ.get("GRAPHCG_BPB_ORTHOGONAL_WEIGHT", 0.02))
    graphcg_covariance_conflict_damping = float(os.environ.get("GRAPHCG_COVARIANCE_CONFLICT_DAMPING", 0.50))
    cas_toric_ideal_certificate_path = os.environ.get("CAS_TORIC_IDEAL_CERTIFICATE_PATH", "")
    toricblm_mup = bool(int(os.environ.get("TORICBLM_MUP", "0")))
    mup_base_shapes = os.environ.get("MUP_BASE_SHAPES", "").strip()
    mup_width_mult = float(os.environ.get("MUP_WIDTH_MULT", "1.0"))
    mup_output_mult = float(os.environ.get("MUP_OUTPUT_MULT", "1.0"))
    mup_attention_scale = bool(int(os.environ.get("MUP_ATTENTION_SCALE", "0")))
    mup_attention_base_head_dim = float(os.environ.get("MUP_ATTENTION_BASE_HEAD_DIM", "64.0"))
    mup_rescale_params = bool(int(os.environ.get("MUP_RESCALE_PARAMS", "1")))
    mup_matrix_lr_scale_power = float(os.environ.get("MUP_MATRIX_LR_SCALE_POWER", "0.0"))
    mup_scalar_lr_scale_power = float(os.environ.get("MUP_SCALAR_LR_SCALE_POWER", "0.0"))
    mup_token_lr_scale_power = float(os.environ.get("MUP_TOKEN_LR_SCALE_POWER", "0.0"))
    mup_aux_lr_scale_power = float(os.environ.get("MUP_AUX_LR_SCALE_POWER", "0.0"))
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 0))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/parameter_golf_oai_dense")
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
    start_step = int(os.environ.get("START_STEP", "0"))
    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "1")))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def build_convextok_luts(
    tok: "ConvexTokTokenizer", vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    table_size = max(int(tok.vocab_size), int(vocab_size))
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    lp_scores_np = np.zeros((table_size,), dtype=np.float32)
    rank_scores_np = np.zeros((table_size,), dtype=np.float32)
    priced_np = np.zeros((table_size,), dtype=np.float32)
    lengths = tok.token_byte_lengths()
    base_bytes_np[: min(table_size, lengths.shape[0])] = lengths[: min(table_size, lengths.shape[0])]
    lp_scores = tok.token_lp_scores()
    rank_scores = tok.token_rank_scores()
    priced = tok.token_is_priced()
    lp_scores_np[: min(table_size, lp_scores.shape[0])] = lp_scores[: min(table_size, lp_scores.shape[0])]
    rank_scores_np[: min(table_size, rank_scores.shape[0])] = rank_scores[: min(table_size, rank_scores.shape[0])]
    priced_np[: min(table_size, priced.shape[0])] = priced[: min(table_size, priced.shape[0])]
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.zeros((table_size,), dtype=torch.bool, device=device),
        torch.zeros((table_size,), dtype=torch.bool, device=device),
        torch.tensor(lp_scores_np, dtype=torch.float32, device=device),
        torch.tensor(rank_scores_np, dtype=torch.float32, device=device),
        torch.tensor(priced_np, dtype=torch.float32, device=device),
    )


def build_caseops_token_map(sp: spm.SentencePieceProcessor, vocab_size: int) -> Tensor:
    """Build an involutive, byte-length-preserving ASCII case token map."""

    token_map = torch.arange(vocab_size, dtype=torch.long)
    by_key: dict[tuple[bool, str], int] = {}
    meta: dict[int, tuple[bool, str, int]] = {}
    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id) or sp.is_byte(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        leading = piece.startswith("▁")
        visible = piece[1:] if leading else piece
        if not visible or visible.swapcase() == visible:
            continue
        byte_len = len(visible.encode("utf-8"))
        by_key[(leading, visible)] = token_id
        meta[token_id] = (leading, visible, byte_len)
    for token_id, (leading, visible, byte_len) in meta.items():
        partner = by_key.get((leading, visible.swapcase()))
        if partner is None:
            continue
        _, partner_visible, partner_byte_len = meta.get(partner, (leading, "", -1))
        if partner_visible.swapcase() == visible and partner_byte_len == byte_len:
            token_map[token_id] = partner
            token_map[partner] = token_id
    return token_map


def apply_token_map(tokens: Tensor, token_map: Tensor | None) -> Tensor:
    if token_map is None:
        return tokens
    return token_map[tokens.to(dtype=torch.long)].to(dtype=tokens.dtype)


def load_validation_tokens(
    pattern: str,
    seq_len: int,
    token_map: Tensor | None = None,
    max_tokens: int = 0,
) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to
    # fineweb_val_*. Iterative BPB campaigns may use a deterministic prefix
    # window so validation is frequent enough to guide restarts.
    if max_tokens > 0:
        chunks: list[Tensor] = []
        remaining = max_tokens + 1
        for file in files:
            shard = load_data_shard(file)
            take = min(int(shard.numel()), remaining)
            if take > 0:
                chunks.append(shard[:take])
                remaining -= take
            if remaining <= 0:
                break
        if not chunks:
            raise ValueError(f"Validation token cap {max_tokens} produced no tokens")
        tokens = torch.cat(chunks).contiguous()
    else:
        tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    tokens = apply_token_map(tokens, token_map)
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    tta_optimizer = None
    tta_restore: list[tuple[nn.Parameter, bool]] = []
    tta_param_data_restore: list[tuple[nn.Parameter, Tensor]] = []
    tta_steps_done = 0
    if args.score_first_tta:
        eval_module = model.module if hasattr(model, "module") else model
        eval_base = getattr(eval_module, "_orig_mod", eval_module)
        tta_params: list[nn.Parameter] = []
        for attr in ("graph_output_flattening", "fineweb_tokengt"):
            module = getattr(eval_base, attr, None)
            if module is not None:
                tta_params.extend(list(module.parameters()))
        if tta_params:
            tta_param_ids = {id(param) for param in tta_params}
            for param in eval_base.parameters():
                tta_restore.append((param, bool(param.requires_grad)))
                param.requires_grad_(id(param) in tta_param_ids)
            if not args.score_first_tta_commit:
                tta_param_data_restore = [(param, param.detach().clone()) for param in tta_params]
            tta_optimizer = torch.optim.AdamW(tta_params, lr=max(float(args.score_first_tta_lr), 1e-8))
    try:
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
            if tta_optimizer is not None and tta_steps_done < max(int(args.score_first_tta_steps), 0):
                tta_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    tta_loss = model(x, y)
                tta_loss.backward()
                tta_optimizer.step()
                tta_optimizer.zero_grad(set_to_none=True)
                tta_steps_done += 1
    finally:
        for param, value in tta_param_data_restore:
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        for param, requires_grad in tta_restore:
            param.requires_grad_(requires_grad)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def count_sentencepiece_target_bytes(
    previous_ids: Tensor,
    target_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    """Count decoded target bytes with the same normalization as validation BPB."""

    prev_flat = previous_ids.reshape(-1)
    tgt_flat = target_ids.reshape(-1)
    token_bytes = base_bytes_lut[tgt_flat].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_flat] & ~is_boundary_token_lut[prev_flat]).to(dtype=torch.int16)
    return token_bytes.to(torch.float64).sum()


def sentencepiece_target_byte_lengths(
    previous_ids: Tensor,
    target_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    """Return per-target byte lengths under the validation BPB accounting."""

    prev_flat = previous_ids.reshape(-1)
    tgt_flat = target_ids.reshape(-1)
    token_bytes = base_bytes_lut[tgt_flat].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_flat] & ~is_boundary_token_lut[prev_flat]).to(dtype=torch.int16)
    return token_bytes.to(dtype=torch.float32).view_as(target_ids).clamp_min(1.0)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str, token_map: Tensor | None = None):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.token_map = token_map
        self.file_idx = 0
        self.tokens = apply_token_map(load_data_shard(self.files[0]), self.token_map)
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = apply_token_map(load_data_shard(self.files[self.file_idx]), self.token_map)
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        token_map: Tensor | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, token_map=token_map)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


from toricgt.embedding_forest_of_thought import EmbeddingFoTConfig, EmbeddingForestOfThoughtHead
from toricgt.oai_sidecar import (
    FullEntryGraphParquetStream,
    GraphParquetTokenStream,
    ScheduledGraphParquetTokenStream,
    StructureCoordinateParquetStream,
    ToricGTSidecar,
)
from toricgt.structure_flow_matching import rectified_flow_trajectory_frames, structure_flow_loss

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        *,
        mup_attention_scale: bool = False,
        mup_attention_base_head_dim: float = 64.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.mup_attention_scale = bool(mup_attention_scale)
        self.mup_attention_base_head_dim = float(mup_attention_base_head_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        sdpa_kwargs = {
            "attn_mask": None,
            "is_causal": True,
            "enable_gqa": (self.num_kv_heads != self.num_heads),
        }
        if self.mup_attention_scale:
            sdpa_kwargs["scale"] = math.sqrt(max(self.mup_attention_base_head_dim, 1.0)) / float(self.head_dim)
        y = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class FineWebTokenGTStructuralEmbedding(nn.Module):
    """Causal TokenGT-style graph features for the main FineWeb LM stream.

    Each SP1024 token is a node in a graph.  Directed one-dimensional local
    edges point from already revealed previous tokens into the current token.
    The module emits a compact structural embedding, so graphification is part
    of the primary BPB model instead of an auxiliary sidecar-only loss.
    """

    def __init__(
        self,
        dim: int,
        *,
        vocab_size: int,
        token_class_buckets: int,
        position_buckets: int,
        edge_radius: int,
        distance_features: str,
        structural_weight: float,
        edge_weight: float,
        torus_weight: float,
        identifier_dim: int,
        identifier_weight: float,
        endpoint_weight: float,
        edge_token_weight: float,
        convextok_dag_features: bool,
        convextok_dag_feature_weight: float,
        theta: float = 0.6180339887498948,
    ) -> None:
        super().__init__()
        if token_class_buckets <= 0:
            raise ValueError("TOKENGT_TOKEN_CLASS_BUCKETS must be positive")
        if position_buckets <= 0:
            raise ValueError("TOKENGT_POSITION_BUCKETS must be positive")
        if edge_radius < 0:
            raise ValueError("TOKENGT_GRAPH_RADIUS must be nonnegative")
        if distance_features not in {"embedding", "functional"}:
            raise ValueError("TOKENGT_DISTANCE_FEATURES must be 'embedding' or 'functional'")
        if identifier_dim <= 0:
            raise ValueError("TOKENGT_IDENTIFIER_DIM must be positive")
        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.token_class_buckets = int(token_class_buckets)
        self.position_buckets = int(position_buckets)
        self.edge_radius = int(edge_radius)
        self.distance_features = str(distance_features)
        self.structural_weight = float(structural_weight)
        self.edge_weight = float(edge_weight)
        self.torus_weight = float(torus_weight)
        self.identifier_half_dim = max(1, int(math.ceil(identifier_dim / 2)))
        self.identifier_feature_dim = 2 * self.identifier_half_dim
        self.identifier_weight = float(identifier_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.edge_token_weight = float(edge_token_weight)
        self.convextok_dag_features = bool(convextok_dag_features)
        self.convextok_dag_feature_weight = float(convextok_dag_feature_weight)
        self.theta = float(theta)
        self.token_class_emb = nn.Embedding(self.token_class_buckets, dim)
        self.position_bucket_emb = nn.Embedding(self.position_buckets, dim)
        self.indegree_emb = (
            nn.Embedding(max(1, self.edge_radius + 1), dim) if self.distance_features == "embedding" else None
        )
        self.edge_distance_emb = (
            nn.Embedding(max(1, self.edge_radius + 1), dim) if self.distance_features == "embedding" else None
        )
        self.edge_source_class_emb = nn.Embedding(self.token_class_buckets, dim)
        self.torus_proj = CastedLinear(4, dim, bias=False)
        self.identifier_proj = CastedLinear(self.identifier_feature_dim, dim, bias=False)
        self.endpoint_proj = CastedLinear(2 * self.identifier_feature_dim + 1, dim, bias=False)
        self.edge_token_proj = CastedLinear(dim, dim, bias=False)
        self.convextok_feature_proj = CastedLinear(4, dim, bias=False)
        self.identifier_proj._zero_init = True
        self.endpoint_proj._zero_init = True
        self.edge_token_proj._zero_init = True
        self.convextok_feature_proj._zero_init = True
        self.register_buffer("convextok_lp_scores", torch.zeros(self.vocab_size, dtype=torch.float32), persistent=True)
        self.register_buffer("convextok_rank_scores", torch.zeros(self.vocab_size, dtype=torch.float32), persistent=True)
        self.register_buffer("convextok_priced_flags", torch.zeros(self.vocab_size, dtype=torch.float32), persistent=True)
        self.register_buffer("convextok_byte_lengths", torch.zeros(self.vocab_size, dtype=torch.float32), persistent=True)

    def set_convextok_features(
        self,
        *,
        lp_scores: Tensor,
        rank_scores: Tensor,
        priced_flags: Tensor,
        byte_lengths: Tensor,
    ) -> None:
        for name, source in (
            ("convextok_lp_scores", lp_scores),
            ("convextok_rank_scores", rank_scores),
            ("convextok_priced_flags", priced_flags),
            ("convextok_byte_lengths", byte_lengths),
        ):
            target = getattr(self, name)
            n = min(int(target.numel()), int(source.numel()))
            target.zero_()
            target[:n].copy_(source[:n].detach().to(device=target.device, dtype=target.dtype))

    def _torus_features(self, positions: Tensor, dtype: torch.dtype) -> Tensor:
        pos = positions.to(dtype=torch.float32)
        theta = 2.0 * math.pi * self.theta * pos
        beta = 2.0 * math.pi * 1.4142135623730951 * pos
        features = torch.stack([torch.sin(theta), torch.cos(theta), torch.sin(beta), torch.cos(beta)], dim=-1)
        return features.to(dtype=dtype)

    def _identifier_features(self, positions: Tensor, dtype: torch.dtype) -> Tensor:
        pos = positions.to(dtype=torch.float32)
        freq = torch.arange(1, self.identifier_half_dim + 1, device=positions.device, dtype=torch.float32)
        phase = pos.unsqueeze(-1) * freq.view(*((1,) * pos.ndim), -1) / max(1.0, float(self.position_buckets))
        phase = 2.0 * math.pi * phase
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return (features / math.sqrt(float(self.identifier_feature_dim))).to(dtype=dtype)

    def _distance_features(self, distance_ids: Tensor, dtype: torch.dtype) -> Tensor:
        distance = distance_ids.to(dtype=torch.float32)
        half = (self.dim + 1) // 2
        freq = torch.exp(
            torch.linspace(0.0, -math.log(10000.0), half, device=distance_ids.device, dtype=torch.float32)
        )
        phase = distance.unsqueeze(-1) * freq.view(*((1,) * distance.ndim), -1)
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)[..., : self.dim]
        return (features / math.sqrt(float(self.dim))).to(dtype=dtype)

    def _indegree_signal(self, indegree: Tensor, dtype: torch.dtype) -> Tensor:
        if self.indegree_emb is not None:
            return self.indegree_emb(indegree)
        return self._distance_features(indegree, dtype)

    def _edge_distance_signal(self, distance_ids: Tensor, dtype: torch.dtype) -> Tensor:
        if self.edge_distance_emb is not None:
            return self.edge_distance_emb(distance_ids)
        return self._distance_features(distance_ids, dtype)

    def forward(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        token_class = torch.remainder(input_ids.to(torch.long), self.token_class_buckets)
        positions = torch.arange(seqlen, device=device, dtype=torch.long).view(1, seqlen).expand(bsz, seqlen)
        position_bucket = torch.remainder(positions, self.position_buckets)
        indegree = positions.clamp_max(max(0, self.edge_radius))
        structural = (
            self.token_class_emb(token_class)
            + self.position_bucket_emb(position_bucket)
            + self._indegree_signal(indegree, self.token_class_emb.weight.dtype)
        )
        identifier_features = self._identifier_features(positions, structural.dtype)
        identifier_context = self.identifier_proj(identifier_features)
        if self.edge_radius > 0:
            edge_context = structural.new_zeros((bsz, seqlen, self.dim))
            endpoint_context = structural.new_zeros((bsz, seqlen, self.dim))
            edge_token_context = structural.new_zeros((bsz, seqlen, self.dim))
            valid_count = structural.new_zeros((bsz, seqlen, 1))
            for distance in range(1, self.edge_radius + 1):
                source_class = token_class.new_zeros((bsz, seqlen))
                source_class[:, distance:] = token_class[:, :-distance]
                valid = (positions >= distance).to(dtype=structural.dtype).unsqueeze(-1)
                dist_ids = torch.full_like(token_class, distance)
                local_edge = (
                    self.edge_source_class_emb(source_class) + self._edge_distance_signal(dist_ids, structural.dtype)
                )
                edge_context = edge_context + valid * local_edge
                source_positions = (positions - distance).clamp_min(0)
                source_id = self._identifier_features(source_positions, structural.dtype)
                distance_feature = torch.full(
                    (*positions.shape, 1),
                    float(distance) / float(max(1, self.edge_radius)),
                    device=device,
                    dtype=structural.dtype,
                )
                endpoint_context = endpoint_context + valid * self.endpoint_proj(
                    torch.cat([source_id, identifier_features, distance_feature], dim=-1)
                )
                edge_token_context = edge_token_context + valid * self.edge_token_proj(local_edge)
                valid_count = valid_count + valid
            edge_context = edge_context / valid_count.clamp_min(1.0)
            endpoint_context = endpoint_context / valid_count.clamp_min(1.0)
            edge_token_context = edge_token_context / valid_count.clamp_min(1.0)
        else:
            edge_context = structural.new_zeros((bsz, seqlen, self.dim))
            endpoint_context = structural.new_zeros((bsz, seqlen, self.dim))
            edge_token_context = structural.new_zeros((bsz, seqlen, self.dim))
        torus = self.torus_proj(self._torus_features(positions, structural.dtype))
        if self.convextok_dag_features:
            ids = input_ids.to(dtype=torch.long).clamp_min(0).clamp_max(self.vocab_size - 1)
            max_len = self.convextok_byte_lengths.max().clamp_min(1.0)
            convextok_features = torch.stack(
                [
                    self.convextok_lp_scores[ids],
                    self.convextok_rank_scores[ids],
                    self.convextok_priced_flags[ids],
                    self.convextok_byte_lengths[ids] / max_len,
                ],
                dim=-1,
            ).to(dtype=structural.dtype)
            convextok_context = self.convextok_feature_proj(convextok_features)
        else:
            convextok_context = structural.new_zeros((bsz, seqlen, self.dim))
        return (
            float(self.structural_weight) * structural
            + float(self.edge_weight) * edge_context
            + float(self.torus_weight) * torus
            + float(self.identifier_weight) * identifier_context
            + float(self.endpoint_weight) * endpoint_context
            + float(self.edge_token_weight) * edge_token_context
            + float(self.convextok_dag_feature_weight) * convextok_context
        )


class GraphOutputFlatteningAdapter(nn.Module):
    """Flatten graph-output states back to sequence order for BPB scoring.

    The transformer hidden table is treated as graph-valued output: each token
    row is a node state and local causal one-dimensional edges aggregate from
    already decoded source nodes.  The adapter returns the same row order used
    by SentencePiece BPB evaluation, so the OAI competition objective remains a
    standard next-token sequence score while the model learns a graph-in,
    graph-out representation internally.
    """

    def __init__(
        self,
        dim: int,
        *,
        edge_radius: int,
        distance_features: str,
        node_weight: float,
        edge_weight: float,
        virtual_edge_tokens: bool,
        edge_token_weight: float,
        score_correction: bool,
        score_correction_weight: float,
    ) -> None:
        super().__init__()
        if edge_radius < 0:
            raise ValueError("GRAPH_OUTPUT_EDGE_RADIUS must be nonnegative")
        if distance_features not in {"embedding", "functional"}:
            raise ValueError("GRAPH_OUTPUT_DISTANCE_FEATURES must be 'embedding' or 'functional'")
        self.dim = int(dim)
        self.edge_radius = int(edge_radius)
        self.distance_features = str(distance_features)
        self.node_weight = float(node_weight)
        self.edge_weight = float(edge_weight)
        self.virtual_edge_tokens = bool(virtual_edge_tokens)
        self.edge_token_weight = float(edge_token_weight)
        self.score_correction = bool(score_correction)
        self.score_correction_weight = float(score_correction_weight)
        self.node_proj = CastedLinear(dim, dim, bias=False)
        self.edge_proj = CastedLinear(dim, dim, bias=False)
        self.edge_distance_emb = (
            nn.Embedding(max(1, self.edge_radius + 1), dim) if self.distance_features == "embedding" else None
        )
        self.node_proj._zero_init = True
        self.edge_proj._zero_init = True
        self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.edge_token_gate = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))
        self.score_correction_gate = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

    def _distance_features(self, distance_ids: Tensor, dtype: torch.dtype) -> Tensor:
        distance = distance_ids.to(dtype=torch.float32)
        half = (self.dim + 1) // 2
        freq = torch.exp(
            torch.linspace(0.0, -math.log(10000.0), half, device=distance_ids.device, dtype=torch.float32)
        )
        phase = distance.unsqueeze(-1) * freq.view(*((1,) * distance.ndim), -1)
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)[..., : self.dim]
        return (features / math.sqrt(float(self.dim))).to(dtype=dtype)

    def _edge_distance_signal(self, distance_ids: Tensor, dtype: torch.dtype) -> Tensor:
        if self.edge_distance_emb is not None:
            return self.edge_distance_emb(distance_ids).to(dtype=dtype)
        return self._distance_features(distance_ids, dtype)

    def forward(self, graph_hidden: Tensor) -> Tensor:
        if self.edge_radius > 0:
            edge_context = graph_hidden.new_zeros(graph_hidden.shape)
            edge_token_context = graph_hidden.new_zeros(graph_hidden.shape)
            valid_count = graph_hidden.new_zeros((*graph_hidden.shape[:2], 1))
            for distance in range(1, self.edge_radius + 1):
                valid = torch.zeros((*graph_hidden.shape[:2], 1), device=graph_hidden.device, dtype=graph_hidden.dtype)
                valid[:, distance:, :] = 1
                shifted = graph_hidden.new_zeros(graph_hidden.shape)
                shifted[:, distance:, :] = graph_hidden[:, :-distance, :]
                edge_context = edge_context + valid * shifted
                if self.virtual_edge_tokens:
                    dist_ids = torch.full(
                        graph_hidden.shape[:2],
                        distance,
                        device=graph_hidden.device,
                        dtype=torch.long,
                    )
                    virtual_edge = (
                        shifted
                        + graph_hidden
                        + self._edge_distance_signal(dist_ids, graph_hidden.dtype)
                    )
                    edge_token_context = edge_token_context + valid * virtual_edge
                valid_count = valid_count + valid
            edge_context = edge_context / valid_count.clamp_min(1.0)
            edge_token_context = edge_token_context / valid_count.clamp_min(1.0)
        else:
            edge_context = graph_hidden.new_zeros(graph_hidden.shape)
            edge_token_context = graph_hidden.new_zeros(graph_hidden.shape)
        update = (
            float(self.node_weight) * self.node_proj(graph_hidden)
            + float(self.edge_weight) * self.edge_proj(edge_context)
        )
        if self.virtual_edge_tokens:
            edge_gate = torch.sigmoid(self.edge_token_gate).to(dtype=graph_hidden.dtype)
            update = update + edge_gate * float(self.edge_token_weight) * self.edge_proj(edge_token_context)
        if self.score_correction:
            score_gate = torch.sigmoid(self.score_correction_gate).to(dtype=graph_hidden.dtype)
            update = update + score_gate * float(self.score_correction_weight) * self.node_proj(graph_hidden - edge_context)
        gate = torch.sigmoid(self.gate).to(dtype=graph_hidden.dtype)
        return graph_hidden + gate * update


class OAIEmbeddingGFlowNetHead(nn.Module):
    """Training-only embedding-space GFlowNet head for the OAI baseline.

    Hidden states are treated as a graph-of-thought trajectory.  Token buckets are
    lightweight discrete refinement actions, and mean negative log likelihood
    supplies the terminal reward.  The head is intentionally not serialized in the
    final Parameter-Golf artifact unless a later export path explicitly keeps it.
    """

    def __init__(self, dim: int, *, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        if num_actions < 2:
            raise ValueError("OAI_GFLOWNET_NUM_ACTIONS must be at least 2")
        self.num_actions = int(num_actions)
        self.forward_policy = nn.Sequential(
            RMSNorm(),
            CastedLinear(dim, hidden_dim),
            nn.GELU(),
            CastedLinear(hidden_dim, self.num_actions),
        )
        self.backward_policy = nn.Sequential(
            RMSNorm(),
            CastedLinear(dim, hidden_dim),
            nn.GELU(),
            CastedLinear(hidden_dim, self.num_actions),
        )
        self.log_z = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        hidden: Tensor,
        target_ids: Tensor,
        per_token_nll: Tensor,
        *,
        max_positions: int,
    ) -> dict[str, Tensor]:
        zero = hidden.new_zeros(())
        max_positions = max(2, min(int(max_positions), int(hidden.shape[1])))
        if hidden.shape[0] < 1 or max_positions < 2:
            return {
                "oai_gflownet_loss": zero,
                "oai_gflownet_tb_residual": zero,
                "oai_gflownet_entropy": zero,
                "oai_gflownet_entropy_objective": zero,
                "oai_gflownet_action_diversity": zero,
                "oai_gflownet_reward_mean": zero,
                "oai_gflownet_score_gap": zero,
                "oai_gflownet_log_z": self.log_z.detach().to(device=hidden.device, dtype=hidden.dtype),
            }
        h = hidden[:, :max_positions, :]
        targets = target_ids[:, :max_positions].to(torch.long)
        nll = per_token_nll[:, :max_positions].detach().float()
        f_logits = self.forward_policy(h[:, :-1, :]).float().clamp(-30.0, 30.0)
        b_logits = self.backward_policy(h[:, 1:, :]).float().clamp(-30.0, 30.0)
        actions = torch.remainder(targets[:, 1:], self.num_actions)
        log_pf = F.log_softmax(f_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        log_pb = F.log_softmax(b_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        reward = torch.exp(-nll.mean(dim=1)).clamp_min(1e-8)
        residual = self.log_z.float() + (log_pf - log_pb).mean(dim=1) - reward.log()
        tb_loss = residual.square().mean()
        probs = torch.softmax(f_logits, dim=-1)
        log_probs = torch.log_softmax(f_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean() / math.log(self.num_actions)
        action_mass = probs.mean(dim=(0, 1)).clamp_min(1e-8)
        diversity = -(action_mass * action_mass.log()).sum() / math.log(self.num_actions)
        top2 = torch.topk(f_logits, k=min(2, self.num_actions), dim=-1).values
        score_gap = (top2[..., 0] - top2[..., -1]).mean() if top2.shape[-1] > 1 else zero.float()
        return {
            "oai_gflownet_loss": tb_loss,
            "oai_gflownet_tb_residual": residual.detach().abs().mean(),
            "oai_gflownet_entropy": entropy,
            "oai_gflownet_entropy_objective": (entropy - 1.0).square(),
            "oai_gflownet_action_diversity": diversity.detach(),
            "oai_gflownet_reward_mean": reward.detach().mean(),
            "oai_gflownet_score_gap": score_gap.detach(),
            "oai_gflownet_log_z": self.log_z.detach(),
        }


class OAIStructureCoordinateHead(nn.Module):
    """Training-only coordinate head for real AFDB/PDB structure records."""

    def __init__(self, dim: int, *, hidden_dim: int, distogram_bins: int = 32) -> None:
        super().__init__()
        self.distogram_bins = int(distogram_bins)
        self.dim = int(dim)
        self.velocity = nn.Sequential(
            RMSNorm(),
            CastedLinear(dim, hidden_dim),
            nn.GELU(),
            CastedLinear(hidden_dim, 3),
        )
        self.contact_proj = nn.Sequential(
            RMSNorm(),
            CastedLinear(dim, max(32, hidden_dim // 4)),
            nn.GELU(),
        )
        self.dist_pair = nn.Sequential(
            CastedLinear(2 * dim, hidden_dim),
            nn.GELU(),
            CastedLinear(hidden_dim, self.distogram_bins),
        )

    def forward(
        self,
        hidden: Tensor,
        target_coords: Tensor,
        coord_mask: Tensor,
        *,
        flow_weight: float,
        contact_weight: float,
        distogram_weight: float,
        noise_scale: float = 0.15,
        return_trajectory: bool = False,
        trajectory_steps: int = 16,
    ) -> dict[str, Tensor]:
        max_atoms = min(int(target_coords.shape[1]), int(hidden.shape[1]))
        h = hidden[:, :max_atoms, :]
        coords = target_coords[:, :max_atoms, :].float()
        mask = coord_mask[:, :max_atoms].bool()
        if not bool(mask.any()):
            zero = hidden.float().sum() * 0.0
            out = {
                "structure_loss": zero,
                "flow_matching_loss": zero.detach(),
                "contact_bce_loss": zero.detach(),
                "distogram_loss": zero.detach(),
                "rmsd": zero.detach(),
                "coordinate_count": zero.detach(),
            }
            if return_trajectory:
                out["flow_trajectory"] = torch.empty(
                    target_coords.shape[0],
                    max(2, int(trajectory_steps)),
                    max_atoms,
                    3,
                    dtype=target_coords.dtype,
                    device=target_coords.device,
                )
            return out
        # Center coordinates per structure for stable prediction, but keep the
        # geometry real: targets are AFDB/PDB coordinates after translation only.
        atom_mask = mask.unsqueeze(-1).float()
        center = (coords * atom_mask).sum(dim=1, keepdim=True) / atom_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        coords = coords - center
        noise = torch.randn_like(coords) * float(noise_scale)
        noisy = coords + noise * atom_mask
        pred_velocity = self.velocity(h).float()
        contact_features = self.contact_proj(h).float()
        contact_logits = torch.matmul(contact_features, contact_features.transpose(1, 2)) / math.sqrt(
            max(1, contact_features.shape[-1])
        )
        hi = h[:, :, None, :].float().expand(-1, -1, max_atoms, -1)
        hj = h[:, None, :, :].float().expand(-1, max_atoms, -1, -1)
        pair_features = torch.cat([hi - hj, hi * hj], dim=-1)
        distogram_logits = self.dist_pair(pair_features)
        bin_edges = torch.linspace(2.0, 32.0, self.distogram_bins + 1, device=hidden.device, dtype=torch.float32)
        metrics = structure_flow_loss(
            pred_velocity=pred_velocity,
            noisy_coords=noisy,
            target_coords=coords,
            mask=mask,
            pred_contact_logits=contact_logits,
            pred_distogram_logits=distogram_logits,
            distogram_bin_edges=bin_edges,
            flow_weight=float(flow_weight),
            contact_weight=float(contact_weight),
            distogram_weight=float(distogram_weight),
        )
        out = {
            "structure_loss": metrics.loss,
            "flow_matching_loss": metrics.flow_matching_loss.detach(),
            "contact_bce_loss": metrics.contact_bce_loss.detach(),
            "distogram_loss": metrics.distogram_loss.detach(),
            "rmsd": metrics.rmsd.detach(),
            "coordinate_count": metrics.coordinate_count.detach(),
        }
        if return_trajectory:
            out["flow_trajectory"] = rectified_flow_trajectory_frames(
                noisy.detach(),
                pred_velocity.detach(),
                mask.detach(),
                steps=int(trajectory_steps),
            )
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        mup_attention_scale: bool,
        mup_attention_base_head_dim: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            mup_attention_scale=mup_attention_scale,
            mup_attention_base_head_dim=mup_attention_base_head_dim,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        fineweb_graphify: bool,
        tokengt_first_class: bool,
        tokengt_graph_radius: int,
        tokengt_distance_features: str,
        tokengt_token_class_buckets: int,
        tokengt_position_buckets: int,
        tokengt_structural_weight: float,
        tokengt_edge_weight: float,
        tokengt_torus_weight: float,
        tokengt_identifier_dim: int,
        tokengt_identifier_weight: float,
        tokengt_endpoint_weight: float,
        tokengt_edge_token_weight: float,
        convextok_dag_features: bool,
        convextok_dag_feature_weight: float,
        graph_output_flattening: bool,
        graph_output_edge_radius: int,
        graph_output_distance_features: str,
        graph_output_node_weight: float,
        graph_output_edge_weight: float,
        graph_output_virtual_edge_tokens: bool,
        graph_output_edge_token_weight: float,
        graph_output_score_correction: bool,
        graph_output_score_correction_weight: float,
        toricblm_mup: bool = False,
        mup_width_mult: float = 1.0,
        mup_output_mult: float = 1.0,
        mup_attention_scale: bool = False,
        mup_attention_base_head_dim: float = 64.0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.toricblm_mup = bool(toricblm_mup)
        self.mup_width_mult = max(float(mup_width_mult), 1.0e-8)
        self.mup_output_mult = float(mup_output_mult)
        self.fineweb_graphify = bool(fineweb_graphify)
        self.tokengt_first_class = bool(tokengt_first_class and fineweb_graphify)
        self.graph_output_flattening_enabled = bool(graph_output_flattening and fineweb_graphify)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.fineweb_tokengt = (
            FineWebTokenGTStructuralEmbedding(
                model_dim,
                vocab_size=vocab_size,
                token_class_buckets=tokengt_token_class_buckets,
                position_buckets=tokengt_position_buckets,
                edge_radius=tokengt_graph_radius,
                distance_features=tokengt_distance_features,
                structural_weight=tokengt_structural_weight,
                edge_weight=tokengt_edge_weight,
                torus_weight=tokengt_torus_weight,
                identifier_dim=tokengt_identifier_dim,
                identifier_weight=tokengt_identifier_weight,
                endpoint_weight=tokengt_endpoint_weight,
                edge_token_weight=tokengt_edge_token_weight,
                convextok_dag_features=convextok_dag_features,
                convextok_dag_feature_weight=convextok_dag_feature_weight,
            )
            if self.tokengt_first_class
            else None
        )
        self.graph_output_flattening = (
            GraphOutputFlatteningAdapter(
                model_dim,
                edge_radius=graph_output_edge_radius,
                distance_features=graph_output_distance_features,
                node_weight=graph_output_node_weight,
                edge_weight=graph_output_edge_weight,
                virtual_edge_tokens=graph_output_virtual_edge_tokens,
                edge_token_weight=graph_output_edge_token_weight,
                score_correction=graph_output_score_correction,
                score_correction_weight=graph_output_score_correction_weight,
            )
            if self.graph_output_flattening_enabled
            else None
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    mup_attention_scale,
                    mup_attention_base_head_dim,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _hidden(
        self,
        input_ids: Tensor,
        *,
        flatten_graph_output: bool,
        activation_checkpoint: bool = False,
    ) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.fineweb_tokengt is not None:
            x = x + self.fineweb_tokengt(input_ids).to(dtype=x.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            block = self.blocks[i]
            x = checkpoint(block, x, x0, use_reentrant=False) if activation_checkpoint and self.training else block(x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block = self.blocks[self.num_encoder_layers + i]
            x = checkpoint(block, x, x0, use_reentrant=False) if activation_checkpoint and self.training else block(x, x0)

        x = self.final_norm(x)
        if flatten_graph_output and self.graph_output_flattening is not None:
            x = self.graph_output_flattening(x)
        return x

    def _logits_from_hidden(self, hidden: Tensor) -> Tensor:
        x = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            if self.toricblm_mup:
                x = x * (self.mup_output_mult / self.mup_width_mult)
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            if self.toricblm_mup:
                x = x * (self.mup_output_mult / self.mup_width_mult)
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _loss_from_hidden(self, hidden: Tensor, target_ids: Tensor) -> tuple[Tensor, Tensor]:
        targets = target_ids.reshape(-1)
        logits = self._logits_from_hidden(hidden)
        per_token_nll = F.cross_entropy(logits.float(), targets, reduction="none").view_as(target_ids)
        return per_token_nll.mean(), per_token_nll

    def logits_for_distill(self, input_ids: Tensor, *, flatten_graph_output: bool = True) -> Tensor:
        hidden = self._hidden(input_ids, flatten_graph_output=flatten_graph_output)
        return self._logits_from_hidden(hidden).view(*input_ids.shape, -1)

    def set_convextok_features(
        self,
        *,
        lp_scores: Tensor,
        rank_scores: Tensor,
        priced_flags: Tensor,
        byte_lengths: Tensor,
    ) -> None:
        if self.fineweb_tokengt is None:
            return
        self.fineweb_tokengt.set_convextok_features(
            lp_scores=lp_scores,
            rank_scores=rank_scores,
            priced_flags=priced_flags,
            byte_lengths=byte_lengths,
        )

    def convextok_toric_regularizer(self, *, topk: int = 96) -> Tensor:
        if self.fineweb_tokengt is None or not self.fineweb_tokengt.convextok_dag_features:
            return self.tok_emb.weight.float().sum() * 0.0
        priced = self.fineweb_tokengt.convextok_priced_flags > 0.5
        lp = self.fineweb_tokengt.convextok_lp_scores.float()
        candidates = torch.nonzero(priced & (lp > 0), as_tuple=False).flatten()
        if candidates.numel() < 2:
            return self.tok_emb.weight.float().sum() * 0.0
        order = torch.argsort(lp[candidates], descending=True)
        ids = candidates[order[: max(2, min(int(topk), int(candidates.numel())))]]
        emb = F.normalize(self.tok_emb.weight[ids].float(), dim=-1)
        lengths = self.fineweb_tokengt.convextok_byte_lengths[ids].float()
        max_len = lengths.max().clamp_min(1.0)
        features = torch.stack(
            [
                lp[ids] / lp[ids].max().clamp_min(1e-6),
                self.fineweb_tokengt.convextok_rank_scores[ids].float(),
                lengths / max_len,
            ],
            dim=-1,
        )
        features = F.normalize(features, dim=-1)
        return F.mse_loss(emb @ emb.T, features @ features.T)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self._hidden(
            input_ids,
            flatten_graph_output=True,
            activation_checkpoint=bool(getattr(self, "main_activation_checkpoint", False)),
        )
        loss, _ = self._loss_from_hidden(hidden, target_ids)
        return loss

    def forward_aux(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        flatten_graph_output: bool = False,
        activation_checkpoint: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self._hidden(
            input_ids,
            flatten_graph_output=flatten_graph_output,
            activation_checkpoint=activation_checkpoint,
        )
        loss, per_token_nll = self._loss_from_hidden(hidden, target_ids)
        return loss, hidden, per_token_nll

    def graph_output_flattening_calibration(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        margin: float = 0.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Train graph-output flattening only when it worsens FineWeb CE.

        The raw graph hidden state is detached before the flattening adapter, so
        the calibration pressure is concentrated on the OAI-FineWeb-only
        flattening/score-correction path instead of pulling the main backbone
        away from the score-defining BPB objective.
        """

        if self.graph_output_flattening is None:
            zero = input_ids.new_zeros((), dtype=torch.float32)
            return zero, {
                "graph_output_flattening_calibration_loss": zero,
                "graph_output_flattening_raw_ce": zero,
                "graph_output_flattening_flat_ce": zero,
                "graph_output_flattening_ce_lift": zero,
                "graph_output_flattening_ce_regression": zero,
            }
        with torch.no_grad():
            raw_hidden = self._hidden(input_ids, flatten_graph_output=False)
            raw_loss, _ = self._loss_from_hidden(raw_hidden, target_ids)
        flat_hidden = self.graph_output_flattening(raw_hidden.detach())
        flat_loss, _ = self._loss_from_hidden(flat_hidden, target_ids)
        regression = torch.relu(flat_loss.float() - raw_loss.detach().float() + float(margin))
        lift = raw_loss.detach().float() - flat_loss.detach().float()
        return regression, {
            "graph_output_flattening_calibration_loss": regression.detach().float(),
            "graph_output_flattening_raw_ce": raw_loss.detach().float(),
            "graph_output_flattening_flat_ce": flat_loss.detach().float(),
            "graph_output_flattening_ce_lift": lift,
            "graph_output_flattening_ce_regression": regression.detach().float(),
        }


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if bool(args.compile_model):
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    structure_readiness: dict[str, object] = {}
    if args.toricblm_structure_readiness_manifest:
        manifest_path = Path(args.toricblm_structure_readiness_manifest)
        if manifest_path.exists():
            try:
                structure_readiness = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log0(f"toricblm_structure_readiness:load_failed path:{manifest_path} error:{exc!r}")
        else:
            log0(f"toricblm_structure_readiness:missing_manifest path:{manifest_path}")
    structure_coordinate_records = int(
        structure_readiness.get("coordinate_bearing_records")
        or structure_readiness.get("coordinate_rows")
        or structure_readiness.get("rows")
        or 0
    )
    structure_association_records = int(structure_readiness.get("structure_association_records") or 0)

    wandb_run = None
    if master_process and os.environ.get("WANDB_PROJECT"):
        try:
            import wandb

            wandb_config = {}
            for name in dir(args):
                if name.startswith("_"):
                    continue
                value = getattr(args, name)
                if isinstance(value, (str, int, float, bool)):
                    wandb_config[name] = value

            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=os.environ.get("WANDB_RUN_NAME", args.run_id),
                id=os.environ.get("WANDB_RUN_ID", args.run_id),
                resume=os.environ.get("WANDB_RESUME", "allow"),
                config=wandb_config,
            )
        except Exception as exc:
            log0(f"wandb:init_failed:{exc!r}")

    def wandb_log0(payload: dict[str, float], step_value: int) -> None:
        if not master_process or wandb_run is None:
            return
        try:
            wandb_run.log(payload, step=step_value)
        except Exception as exc:
            log0(f"wandb:log_failed:{exc!r}")

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer_kind = "sentencepiece"
    sp = None
    convextok_tokenizer = None
    convextok_lp_scores = torch.zeros((args.vocab_size,), dtype=torch.float32, device=device)
    convextok_rank_scores = torch.zeros((args.vocab_size,), dtype=torch.float32, device=device)
    convextok_priced_flags = torch.zeros((args.vocab_size,), dtype=torch.float32, device=device)
    if args.tokenizer_path.endswith(".model"):
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        caseops_token_map = build_caseops_token_map(sp, args.vocab_size) if args.fineweb_caseops else None
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            sp, args.vocab_size, device
        )
        convextok_byte_lengths = base_bytes_lut.to(dtype=torch.float32)
    elif args.tokenizer_path.endswith(".convextok.json"):
        if ConvexTokTokenizer is None:
            raise RuntimeError("ConvexTok tokenizer support is unavailable; convextok.py import failed")
        tokenizer_kind = "convextok"
        convextok_tokenizer = ConvexTokTokenizer.load_json(args.tokenizer_path)
        if int(convextok_tokenizer.vocab_size) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match ConvexTok vocab_size={int(convextok_tokenizer.vocab_size)}"
            )
        if args.fineweb_caseops:
            raise ValueError("FINEWEB_CASEOPS is SentencePiece-specific and must be disabled for ConvexTok")
        caseops_token_map = None
        (
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            convextok_lp_scores,
            convextok_rank_scores,
            convextok_priced_flags,
        ) = build_convextok_luts(convextok_tokenizer, args.vocab_size, device)
        convextok_byte_lengths = base_bytes_lut.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported tokenizer path: {args.tokenizer_path}")
    graph_text_tokenizer = sp if tokenizer_kind == "sentencepiece" else convextok_tokenizer
    if graph_text_tokenizer is None:
        raise RuntimeError(f"No graph text tokenizer initialized for tokenizer_kind={tokenizer_kind}")
    caseops_pairs = 0 if caseops_token_map is None else int((caseops_token_map != torch.arange(args.vocab_size)).sum().item())
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(
        args.val_files,
        args.train_seq_len,
        token_map=caseops_token_map,
        max_tokens=max(0, int(args.val_max_tokens)),
    )
    log0(f"val_bpb:enabled tokenizer_kind={tokenizer_kind} tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1} "
        f"max_tokens:{int(args.val_max_tokens)} eval_initial:{int(args.eval_initial)}"
    )
    log0(f"fineweb_caseops:enabled:{int(args.fineweb_caseops)} swapped_token_ids:{caseops_pairs}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        fineweb_graphify=args.fineweb_graphify,
        tokengt_first_class=args.tokengt_first_class,
        tokengt_graph_radius=args.tokengt_graph_radius,
        tokengt_distance_features=args.tokengt_distance_features,
        tokengt_token_class_buckets=args.tokengt_token_class_buckets,
        tokengt_position_buckets=args.tokengt_position_buckets,
        tokengt_structural_weight=args.tokengt_structural_weight,
        tokengt_edge_weight=args.tokengt_edge_weight,
        tokengt_torus_weight=args.tokengt_torus_weight,
        tokengt_identifier_dim=args.tokengt_identifier_dim,
        tokengt_identifier_weight=args.tokengt_identifier_weight,
        tokengt_endpoint_weight=args.tokengt_endpoint_weight,
        tokengt_edge_token_weight=args.tokengt_edge_token_weight,
        convextok_dag_features=args.convextok_dag_features and tokenizer_kind == "convextok",
        convextok_dag_feature_weight=args.convextok_dag_feature_weight,
        graph_output_flattening=args.graph_output_flattening,
        graph_output_edge_radius=args.graph_output_edge_radius,
        graph_output_distance_features=args.graph_output_distance_features,
        graph_output_node_weight=args.graph_output_node_weight,
        graph_output_edge_weight=args.graph_output_edge_weight,
        graph_output_virtual_edge_tokens=args.graph_output_virtual_edge_tokens,
        graph_output_edge_token_weight=args.graph_output_edge_token_weight,
        graph_output_score_correction=args.graph_output_score_correction,
        graph_output_score_correction_weight=args.graph_output_score_correction_weight,
        toricblm_mup=args.toricblm_mup,
        mup_width_mult=args.mup_width_mult,
        mup_output_mult=args.mup_output_mult,
        mup_attention_scale=args.mup_attention_scale,
        mup_attention_base_head_dim=args.mup_attention_base_head_dim,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    base_model.set_convextok_features(
        lp_scores=convextok_lp_scores,
        rank_scores=convextok_rank_scores,
        priced_flags=convextok_priced_flags,
        byte_lengths=convextok_byte_lengths,
    )
    mup_applied = False
    if args.toricblm_mup:
        if args.mup_base_shapes:
            repo_root = Path(__file__).resolve().parents[2]
            mup_path = repo_root / "external" / "mup"
            if mup_path.exists():
                sys.path.insert(0, str(mup_path))
            try:
                from mup import set_base_shapes

                set_base_shapes(
                    base_model,
                    args.mup_base_shapes,
                    rescale_params=bool(args.mup_rescale_params and not args.resume_checkpoint),
                )
                mup_applied = True
            except Exception as exc:
                raise RuntimeError(f"TORICBLM_MUP=1 but applying MUP_BASE_SHAPES failed: {exc!r}") from exc
        else:
            log0("toricblm_mup:enabled_without_base_shapes width_scaled_readout_and_attention_only")
    sidecar = None
    graph_loader = None
    graph_lm_loader = None
    long_entry_loader = None
    oai_gflownet_head = None
    oai_fot_head = None
    structure_loader = None
    structure_head = None

    def build_graph_stream(seq_len: int, batch_size: int):
        if args.late_graph_train_glob:
            return ScheduledGraphParquetTokenStream(
                args.graph_train_glob,
                args.late_graph_train_glob,
                graph_text_tokenizer,
                seq_len,
                batch_size,
                late_start_step=args.late_graph_start_step,
                late_mix_ratio=args.late_graph_mix_ratio,
            )
        return GraphParquetTokenStream(
            args.graph_train_glob,
            graph_text_tokenizer,
            seq_len,
            batch_size,
        )

    if args.graph_lm_primary:
        if args.graph_lm_batch_size < 1:
            raise ValueError("GRAPH_LM_BATCH_SIZE must be at least 1")
        if args.graph_lm_seq_len > args.train_seq_len:
            raise ValueError("GRAPH_LM_SEQ_LEN must not exceed TRAIN_SEQ_LEN")
        try:
            graph_lm_loader = build_graph_stream(args.graph_lm_seq_len, args.graph_lm_batch_size)
        except Exception:
            if args.require_graph_lm_primary:
                raise
            log0("graph_lm_primary:init_failed_nonfatal enabled=0")
            graph_lm_loader = None
    if args.toricgt_sidecar:
        if args.sidecar_batch_size < 2:
            raise ValueError("TORICGT_SIDECAR_BATCH_SIZE must be at least 2 so the retrieval head has candidates")
        if args.sidecar_seq_len > args.train_seq_len:
            raise ValueError("TORICGT_SIDECAR_SEQ_LEN must not exceed TRAIN_SEQ_LEN")
        try:
            sidecar = ToricGTSidecar(args.model_dim, args).to(device)
            graph_loader = build_graph_stream(args.sidecar_seq_len, args.sidecar_batch_size)
        except Exception:
            if args.require_toricgt_sidecar:
                raise
            log0("toricgt_sidecar:init_failed_nonfatal enabled=0")
            sidecar = None
            graph_loader = None
    if args.long_entry_training:
        if not args.long_entry_train_glob:
            raise ValueError("LONG_ENTRY_TRAINING=1 requires LONG_ENTRY_TRAIN_GLOB, LATE_GRAPH_TRAIN_GLOB, or GRAPH_TRAIN_GLOB")
        if args.long_entry_segment_len < 32:
            raise ValueError("LONG_ENTRY_SEGMENT_LEN must be at least 32")
        if args.long_entry_segment_stride < 1:
            raise ValueError("LONG_ENTRY_SEGMENT_STRIDE must be positive")
        if args.long_entry_max_tokens < args.long_entry_segment_len:
            raise ValueError("LONG_ENTRY_MAX_TOKENS must be at least LONG_ENTRY_SEGMENT_LEN")
        long_entry_loader = FullEntryGraphParquetStream(
            args.long_entry_train_glob,
            graph_text_tokenizer,
            batch_size=args.long_entry_batch_size,
            max_tokens=args.long_entry_max_tokens,
        )
    if args.oai_gflownet:
        oai_gflownet_head = OAIEmbeddingGFlowNetHead(
            args.model_dim,
            num_actions=args.oai_gflownet_num_actions,
            hidden_dim=args.oai_gflownet_hidden_dim,
    ).to(device)
    base_model.main_activation_checkpoint = bool(args.main_activation_checkpoint)
    if args.oai_embedding_fot:
        oai_fot_head = EmbeddingForestOfThoughtHead(
            EmbeddingFoTConfig(
                dim=args.model_dim,
                num_trees=args.oai_fot_num_trees,
                max_depth=args.oai_fot_max_depth,
                branching=args.oai_fot_branching,
                topk_trees=args.oai_fot_topk_trees,
                hidden_dim=args.oai_fot_hidden_dim,
                max_positions=args.oai_fot_max_positions,
                consensus_buckets=args.oai_fot_consensus_buckets,
                correction_scale=args.oai_fot_correction_scale,
                ucb_exploration=args.oai_fot_ucb_exploration,
                temperature=args.oai_fot_temperature,
                sparse_weight=args.oai_fot_sparse_weight,
                ucb_weight=args.oai_fot_ucb_weight,
                correction_weight=args.oai_fot_correction_weight,
                consensus_weight=args.oai_fot_consensus_weight,
                tb_weight=args.oai_fot_tb_weight,
                subtb_weight=args.oai_fot_subtb_weight,
                complexity_weight=args.oai_fot_complexity_weight,
                reward_advanced_bonus=args.oai_fot_reward_advanced_bonus,
                reward_mode=args.oai_fot_reward_mode,
                bpb_delta_weight=args.oai_fot_bpb_delta_weight,
                reward_graph_weight=args.oai_fot_reward_graph_weight,
                reward_consensus_weight=args.oai_fot_reward_consensus_weight,
                reward_complexity_weight=args.oai_fot_reward_complexity_weight,
                reward_floor=args.oai_fot_reward_floor,
            )
        ).to(device)
    if args.toricblm_structure_flow and args.toricblm_structure_train_glob:
        if args.toricblm_structure_seq_len > args.train_seq_len:
            raise ValueError("TORICBLM_STRUCTURE_SEQ_LEN must not exceed TRAIN_SEQ_LEN")
        structure_loader = StructureCoordinateParquetStream(
            args.toricblm_structure_train_glob,
            graph_text_tokenizer,
            args.toricblm_structure_seq_len,
            args.toricblm_structure_batch_size,
            max_atoms=args.toricblm_structure_max_atoms,
        )
        structure_head = OAIStructureCoordinateHead(
            args.model_dim,
            hidden_dim=args.toricblm_structure_head_hidden_dim,
        ).to(device)
    if args.require_toricgt_sidecar and sidecar is None:
        raise RuntimeError("REQUIRE_TORICGT_SIDECAR=1 but the ToricGT sidecar is disabled")
    if args.require_graph_lm_primary and graph_lm_loader is None:
        raise RuntimeError("REQUIRE_GRAPH_LM_PRIMARY=1 but the graph LM primary stream is disabled")
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            base_model.load_state_dict(checkpoint["model"], strict=True)
            if sidecar is not None and checkpoint.get("sidecar") is not None:
                sidecar.load_state_dict(checkpoint["sidecar"], strict=True)
            if oai_gflownet_head is not None and checkpoint.get("oai_gflownet") is not None:
                oai_gflownet_head.load_state_dict(checkpoint["oai_gflownet"], strict=True)
            if oai_fot_head is not None and checkpoint.get("oai_fot") is not None:
                oai_fot_head.load_state_dict(checkpoint["oai_fot"], strict=True)
            if structure_head is not None and checkpoint.get("structure_head") is not None:
                structure_head.load_state_dict(checkpoint["structure_head"], strict=True)
            if args.start_step <= 0:
                args.start_step = int(checkpoint.get("step", 0))
        else:
            base_model.load_state_dict(checkpoint, strict=True)
        log0(f"resume_checkpoint:{resume_path} start_step:{args.start_step}")
    teacher_model: GPT | None = None
    if args.teacher_checkpoint and args.teacher_distill_weight > 0.0:
        teacher_path = Path(args.teacher_checkpoint)
        teacher_model = GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            fineweb_graphify=args.fineweb_graphify,
            tokengt_first_class=args.tokengt_first_class,
            tokengt_graph_radius=args.tokengt_graph_radius,
            tokengt_distance_features=args.tokengt_distance_features,
            tokengt_token_class_buckets=args.tokengt_token_class_buckets,
            tokengt_position_buckets=args.tokengt_position_buckets,
            tokengt_structural_weight=args.tokengt_structural_weight,
            tokengt_edge_weight=args.tokengt_edge_weight,
            tokengt_torus_weight=args.tokengt_torus_weight,
            tokengt_identifier_dim=args.tokengt_identifier_dim,
            tokengt_identifier_weight=args.tokengt_identifier_weight,
            tokengt_endpoint_weight=args.tokengt_endpoint_weight,
            tokengt_edge_token_weight=args.tokengt_edge_token_weight,
            convextok_dag_features=args.convextok_dag_features and tokenizer_kind == "convextok",
            convextok_dag_feature_weight=args.convextok_dag_feature_weight,
            graph_output_flattening=args.graph_output_flattening,
            graph_output_edge_radius=args.graph_output_edge_radius,
            graph_output_distance_features=args.graph_output_distance_features,
            graph_output_node_weight=args.graph_output_node_weight,
            graph_output_edge_weight=args.graph_output_edge_weight,
            graph_output_virtual_edge_tokens=args.graph_output_virtual_edge_tokens,
            graph_output_edge_token_weight=args.graph_output_edge_token_weight,
            graph_output_score_correction=args.graph_output_score_correction,
            graph_output_score_correction_weight=args.graph_output_score_correction_weight,
            toricblm_mup=args.toricblm_mup,
            mup_width_mult=args.mup_width_mult,
            mup_output_mult=args.mup_output_mult,
            mup_attention_scale=args.mup_attention_scale,
            mup_attention_base_head_dim=args.mup_attention_base_head_dim,
        ).to(device).bfloat16()
        teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher_state = teacher_payload["model"] if isinstance(teacher_payload, dict) and "model" in teacher_payload else teacher_payload
        teacher_missing, teacher_unexpected = teacher_model.load_state_dict(teacher_state, strict=False)
        for module in teacher_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(teacher_model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        log0(
            f"teacher_distill:enabled checkpoint:{teacher_path} weight:{args.teacher_distill_weight} "
            f"temp:{args.teacher_distill_temperature} missing:{len(teacher_missing)} unexpected:{len(teacher_unexpected)}"
        )
    else:
        log0(
            "teacher_distill:"
            f"enabled:0 checkpoint:{args.teacher_checkpoint or 'none'} weight:{args.teacher_distill_weight}"
        )
    compiled_model: nn.Module = torch.compile(base_model, dynamic=False, fullgraph=True) if args.compile_model else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    width_mult_for_lr = max(float(args.mup_width_mult), 1.0e-8) if args.toricblm_mup else 1.0

    def scaled_lr(base_lr: float, power: float) -> float:
        return float(base_lr) / (width_mult_for_lr ** float(power))

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    token_lr = scaled_lr(token_lr, args.mup_token_lr_scale_power)
    matrix_lr = scaled_lr(args.matrix_lr, args.mup_matrix_lr_scale_power)
    scalar_lr = scaled_lr(args.scalar_lr, args.mup_scalar_lr_scale_power)
    head_lr = scaled_lr(args.head_lr, args.mup_token_lr_scale_power)
    tokengt_lr = scaled_lr(args.tokengt_first_class_lr, args.mup_aux_lr_scale_power)
    flattening_lr = scaled_lr(args.graph_output_flattening_lr, args.mup_aux_lr_scale_power)
    sidecar_lr = scaled_lr(args.sidecar_lr, args.mup_aux_lr_scale_power)
    gflownet_lr = scaled_lr(args.oai_gflownet_lr, args.mup_aux_lr_scale_power)
    fot_lr = scaled_lr(args.oai_fot_lr, args.mup_aux_lr_scale_power)
    structure_lr = scaled_lr(args.toricblm_structure_lr, args.mup_aux_lr_scale_power)
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    first_class_tokengt_params = (
        list(base_model.fineweb_tokengt.parameters())
        if getattr(base_model, "fineweb_tokengt", None) is not None
        else []
    )
    graph_output_flattening_params = (
        list(base_model.graph_output_flattening.parameters())
        if getattr(base_model, "graph_output_flattening", None) is not None
        else []
    )
    graph_structure_groups = []
    if first_class_tokengt_params:
        graph_structure_groups.append(
            {
                "params": first_class_tokengt_params,
                "lr": tokengt_lr,
                "base_lr": tokengt_lr,
            }
        )
    if graph_output_flattening_params:
        graph_structure_groups.append(
            {
                "params": graph_output_flattening_params,
                "lr": flattening_lr,
                "base_lr": flattening_lr,
            }
        )
    if graph_structure_groups:
        optimizer_first_class_tokengt = torch.optim.AdamW(
            graph_structure_groups,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_first_class_tokengt)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": head_lr, "base_lr": head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    if sidecar is not None:
        optimizer_sidecar = torch.optim.AdamW(
            [{"params": list(sidecar.parameters()), "lr": sidecar_lr, "base_lr": sidecar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_sidecar)
    if oai_gflownet_head is not None:
        optimizer_oai_gflownet = torch.optim.AdamW(
            [
                {
                    "params": list(oai_gflownet_head.parameters()),
                    "lr": gflownet_lr,
                    "base_lr": gflownet_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_oai_gflownet)
    if oai_fot_head is not None:
        optimizer_oai_fot = torch.optim.AdamW(
            [
                {
                    "params": list(oai_fot_head.parameters()),
                    "lr": fot_lr,
                    "base_lr": fot_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_oai_fot)
    if structure_head is not None:
        optimizer_structure = torch.optim.AdamW(
            [
                {
                    "params": list(structure_head.parameters()),
                    "lr": structure_lr,
                    "base_lr": structure_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_structure)

    n_params = sum(p.numel() for p in base_model.parameters())
    first_class_tokengt_param_count = sum(p.numel() for p in first_class_tokengt_params)
    graph_output_flattening_param_count = sum(p.numel() for p in graph_output_flattening_params)
    sidecar_params = sum(p.numel() for p in sidecar.parameters()) if sidecar is not None else 0
    oai_gflownet_params = sum(p.numel() for p in oai_gflownet_head.parameters()) if oai_gflownet_head is not None else 0
    oai_fot_params = sum(p.numel() for p in oai_fot_head.parameters()) if oai_fot_head is not None else 0
    structure_head_params = sum(p.numel() for p in structure_head.parameters()) if structure_head is not None else 0
    log0(f"model_params:{n_params}")
    log0(
        "toricblm_mup:"
        f"enabled:{int(args.toricblm_mup)} base_shapes:{args.mup_base_shapes or 'none'} "
        f"base_shapes_applied:{int(mup_applied)} width_mult:{args.mup_width_mult} "
        f"output_mult:{args.mup_output_mult} attention_scale:{int(args.mup_attention_scale)} "
        f"attention_base_head_dim:{args.mup_attention_base_head_dim} "
        f"rescale_params:{int(args.mup_rescale_params and not args.resume_checkpoint)} "
        f"lr_scale_powers:matrix={args.mup_matrix_lr_scale_power},scalar={args.mup_scalar_lr_scale_power},"
        f"token={args.mup_token_lr_scale_power},aux={args.mup_aux_lr_scale_power} "
        f"effective_lrs:token={token_lr},matrix={matrix_lr},scalar={scalar_lr},"
        f"head={head_lr},tokengt={tokengt_lr},flattening={flattening_lr},"
        f"sidecar={sidecar_lr},gflownet={gflownet_lr},fot={fot_lr},structure={structure_lr}"
    )
    log0(
        "fineweb_graphification:"
        f"enabled:{int(args.fineweb_graphify)} tokengt_first_class:{int(args.tokengt_first_class)} "
        f"params:{first_class_tokengt_param_count} radius:{args.tokengt_graph_radius} "
        f"distance_features:{args.tokengt_distance_features} "
        f"token_class_buckets:{args.tokengt_token_class_buckets} position_buckets:{args.tokengt_position_buckets} "
        f"weights:structural={args.tokengt_structural_weight},edge={args.tokengt_edge_weight},"
        f"torus={args.tokengt_torus_weight},identifier={args.tokengt_identifier_weight},"
        f"endpoint={args.tokengt_endpoint_weight},edge_token={args.tokengt_edge_token_weight} "
        f"convextok_dag_features:{int(args.convextok_dag_features and tokenizer_kind == 'convextok')} "
        f"convextok_dag_feature_weight:{args.convextok_dag_feature_weight} "
        f"identifier_dim:{args.tokengt_identifier_dim} lr:{tokengt_lr}"
    )
    log0(
        "graph_output_flattening:"
        f"enabled:{int(base_model.graph_output_flattening is not None)} "
        f"params:{graph_output_flattening_param_count} radius:{args.graph_output_edge_radius} "
        f"distance_features:{args.graph_output_distance_features} "
        f"weights:node={args.graph_output_node_weight},edge={args.graph_output_edge_weight},"
        f"virtual_edge={args.graph_output_edge_token_weight},"
        f"score_correction={args.graph_output_score_correction_weight} "
        f"virtual_edge_tokens:{int(args.graph_output_virtual_edge_tokens)} "
        f"score_correction_enabled:{int(args.graph_output_score_correction)} "
        f"lr:{flattening_lr} "
        f"calibration_weight:{args.graph_output_calibration_loss_weight} "
        f"calibration_every:{args.graph_output_calibration_every} "
        f"flatten_order=tokenizer_sequence scope=oai_fineweb_bpb_only"
    )
    log0(
        "graph_lm_primary:"
        f"enabled:{int(graph_lm_loader is not None)} required:{int(args.require_graph_lm_primary)} "
        f"seq_len:{args.graph_lm_seq_len} batch_size:{args.graph_lm_batch_size} every:{args.graph_lm_every} "
        f"loss_weight_start:{args.graph_lm_loss_weight_start} loss_weight_peak:{args.graph_lm_loss_weight} "
        f"warmup_steps:{args.graph_lm_warmup_steps} hold_steps:{args.graph_lm_hold_steps} "
        f"graph_glob:{args.graph_train_glob} "
        f"late_graph_glob:{args.late_graph_train_glob or 'disabled'} "
        f"late_start_step:{args.late_graph_start_step} "
        f"late_mix_ratio:{args.late_graph_mix_ratio} "
        f"late_upsample_passes:{args.late_graph_upsample_passes} "
        f"stream:{graph_lm_loader.describe() if graph_lm_loader is not None else 'disabled'}"
    )
    log0(
        "long_entry_full_row_training:"
        f"enabled:{int(long_entry_loader is not None)} "
        f"glob:{args.long_entry_train_glob or 'disabled'} "
        f"stream:{long_entry_loader.describe() if long_entry_loader is not None else 'disabled'} "
        f"every:{args.long_entry_every} batch_size:{args.long_entry_batch_size} "
        f"max_tokens:{args.long_entry_max_tokens} segment_len:{args.long_entry_segment_len} "
        f"segment_stride:{args.long_entry_segment_stride} max_segments:{args.long_entry_max_segments} "
        f"loss_weight:{args.long_entry_loss_weight} "
        f"next_segment_weight:{args.long_entry_next_segment_weight} "
        f"boundary_weight:{args.long_entry_boundary_weight} "
        f"coverage_weight:{args.long_entry_segment_coverage_weight} "
        f"fot_loss_weight:{args.long_entry_fot_loss_weight} "
        f"fot_every:{args.long_entry_fot_every} "
        f"fot_max_segments:{args.long_entry_fot_max_segments} "
        f"full_context_every:{args.long_entry_full_context_every} "
        f"full_context_seq_len:{args.long_entry_full_context_seq_len} "
        f"full_context_loss_weight:{args.long_entry_full_context_loss_weight} "
        f"full_context_min_tokens:{args.long_entry_full_context_min_tokens} "
        f"segment_checkpoint:{int(args.long_entry_segment_checkpoint)} "
        f"full_context_checkpoint:{int(args.long_entry_full_context_checkpoint)}"
    )
    log0(
        "toricblm_structure_training:"
        f"flow_enabled:{int(args.toricblm_structure_flow)} "
        f"head_enabled:{int(structure_head is not None)} "
        f"params:{structure_head_params} "
        f"train_glob:{args.toricblm_structure_train_glob or 'disabled'} "
        f"stream:{structure_loader.describe() if structure_loader is not None else 'disabled'} "
        f"seq_len:{args.toricblm_structure_seq_len} "
        f"batch_size:{args.toricblm_structure_batch_size} "
        f"every:{args.toricblm_structure_every} "
        f"max_atoms:{args.toricblm_structure_max_atoms} "
        f"lr:{structure_lr} "
        f"flow_weight:{args.toricblm_structure_flow_weight} "
        f"contact_weight:{args.toricblm_structure_contact_weight} "
        f"distogram_weight:{args.toricblm_structure_distogram_weight} "
        f"frame_weight:{args.toricblm_structure_frame_weight} "
        f"start_step:{args.toricblm_structure_flow_start_step} "
        f"readiness_manifest:{args.toricblm_structure_readiness_manifest or 'none'} "
        f"coordinate_records:{structure_coordinate_records} "
        f"association_records:{structure_association_records} "
        f"target_shortfalls:{json.dumps(structure_readiness.get('target_shortfalls', {}), sort_keys=True)} "
        f"status:{structure_readiness.get('structure_flow_status', 'unknown')}"
    )
    log0(
        "toricgt_sidecar:"
        f"enabled:{int(sidecar is not None)} required:{int(args.require_toricgt_sidecar)} "
        f"params:{sidecar_params} graph_glob:{args.graph_train_glob} "
        f"late_graph_glob:{args.late_graph_train_glob or 'disabled'} "
        f"late_start_step:{args.late_graph_start_step} "
        f"late_mix_ratio:{args.late_graph_mix_ratio} "
        f"seq_len:{args.sidecar_seq_len} batch_size:{args.sidecar_batch_size} every:{args.sidecar_every} "
        f"weights:graphcg={args.graphcg_loss_weight},analogy={args.analogy_loss_weight},"
        f"tokengt_graph={args.tokengt_graph_loss_weight},trajectory_memory={args.trajectory_memory_loss_weight},"
        f"toric_geometry={args.toric_geometry_loss_weight},vector_bundle_1d_cone_ce={args.toric_vector_bundle_loss_weight},"
        f"toric_bgg={args.toric_bgg_loss_weight},koszul={args.koszul_persistence_loss_weight},"
        f"combinatorial_toric={args.combinatorial_toric_loss_weight},"
        f"derived_signature={args.derived_signature_loss_weight},"
        f"compute_all_metrics={args.sidecar_compute_all_metrics}"
    )
    log0(
        "advanced_bpb_controls:"
        f"aux_grad_routing:{int(args.aux_grad_routing)} graph_lm:{int(args.aux_grad_route_graph_lm)} "
        f"sidecar:{int(args.aux_grad_route_sidecar)} teacher:{int(args.aux_grad_route_teacher)} "
        f"long_entry:{int(args.aux_grad_route_long_entry)} "
        f"retrieval_conditioned_aux:{int(args.retrieval_conditioned_aux)} "
        f"uncertainty_weighting:{int(args.sidecar_uncertainty_weighting)} "
        f"bpb_first_staging:{int(args.bpb_first_aux_staging)} "
        f"bpb_first_core/ramp/min:{args.bpb_first_core_steps}/{args.bpb_first_ramp_steps}/{args.bpb_first_min_aux_mult} "
        f"aux_conflict_controller:{int(args.aux_conflict_controller)} "
        f"aux_conflict_damp/boost:{args.aux_conflict_damp_min}/{args.aux_conflict_boost_max} "
        f"lagrangian_controller:{int(args.advanced_lagrangian_controller)} "
        f"lagrangian_dual_lr:{args.lagrangian_dual_lr} "
        f"lagrangian_bpb_ceiling:{args.lagrangian_bpb_ceiling} "
        f"toric_fan_curriculum:{int(args.toric_fan_curriculum)} "
        f"fan_steps:{args.toric_fan_coarse_steps}/{args.toric_fan_intermediate_steps} "
        f"memory_sheaf_gate:min={args.memory_sheaf_gate_min},threshold={args.memory_sheaf_gate_threshold},"
        f"softness={args.memory_sheaf_gate_softness} "
        f"graphcg_bpb_orthogonal_weight:{args.graphcg_bpb_orthogonal_weight} "
        f"score_first_tta:{int(args.score_first_tta)} "
        f"score_first_tta_commit:{int(args.score_first_tta_commit)}"
    )
    log0(
        "oai_transfer_heads:"
        f"gflownet_enabled:{int(oai_gflownet_head is not None)} "
        f"gflownet_params:{oai_gflownet_params} "
        f"gflownet_weight:{args.oai_gflownet_loss_weight} "
        f"gflownet_entropy_weight:{args.oai_gflownet_entropy_weight} "
        f"gflownet_actions:{args.oai_gflownet_num_actions} "
        f"fot_enabled:{int(oai_fot_head is not None)} "
        f"fot_params:{oai_fot_params} "
        f"fot_weight:{args.oai_fot_loss_weight} "
        f"fot_reward_mode:{args.oai_fot_reward_mode} "
        f"fot_bpb_delta_weight:{args.oai_fot_bpb_delta_weight} "
        f"fot_adaptive:{int(args.oai_fot_adaptive_control)} "
        f"fot_reward/diversity_targets:{args.oai_fot_reward_target}/{args.oai_fot_diversity_target} "
        f"fot_trees:{args.oai_fot_num_trees} "
        f"fot_depth:{args.oai_fot_max_depth} "
        f"fot_branching:{args.oai_fot_branching} "
        f"mtp_enabled:{int(args.oai_mtp)} mtp_weight:{args.oai_mtp_loss_weight} "
        f"mtp_offsets:{args.oai_mtp_offsets}"
    )
    log0(
        "toricgt_sidecar_heads:"
        f"graphcg_full_rank:{int(sidecar.graphcg_basis.shape[0]) if sidecar is not None else 0} "
        f"analogy_lattice:{int(sidecar is not None)} "
        f"tokengt_causal_graph:{int(sidecar is not None)} "
        f"trajectory_memory_retrieval:{int(sidecar is not None)} "
        f"toric_geometry:{int(sidecar is not None)} "
        f"toric_vector_bundle_1d_cone_sheaf:{int(sidecar is not None)} "
        f"toric_bgg_category_o:{int(sidecar is not None)} "
        f"koszul_persistence:{int(sidecar is not None)} "
        f"combinatorial_toric_commutative_algebra:{int(sidecar is not None)} "
        f"derived_signature_distillation:{int(sidecar is not None)}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"compile_model:{int(args.compile_model)}")
    log0(f"main_activation_checkpoint:{int(args.main_activation_checkpoint)}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{matrix_lr} scalar_lr:{scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f} start_step:{args.start_step}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, token_map=caseops_token_map)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    base_grad_params = [param for param in base_model.parameters() if param.requires_grad]

    def snapshot_base_grads() -> list[Tensor | None]:
        return [None if param.grad is None else param.grad.detach().clone() for param in base_grad_params]

    def zero_base_grads() -> None:
        for param in base_grad_params:
            param.grad = None

    def restore_base_grads(grads: list[Tensor | None]) -> None:
        for param, grad in zip(base_grad_params, grads, strict=True):
            param.grad = None if grad is None else grad.clone()

    def add_base_grads(grads: list[Tensor | None]) -> None:
        for param, grad in zip(base_grad_params, grads, strict=True):
            if grad is None:
                continue
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.add_(grad.to(dtype=param.grad.dtype))

    def route_aux_grads(
        primary: list[Tensor | None] | None,
        aux: list[Tensor | None],
        *,
        name: str,
    ) -> tuple[list[Tensor | None], dict[str, float]]:
        if primary is None or not args.aux_grad_routing:
            aux_norm = math.sqrt(
                sum(float(grad.float().pow(2).sum().item()) for grad in aux if grad is not None)
            )
            return aux, {
                f"aux_grad_routing/{name}_enabled": 0.0,
                f"aux_grad_routing/{name}_cosine": 0.0,
                f"aux_grad_routing/{name}_conflict": 0.0,
                f"aux_grad_routing/{name}_projection_coeff": 0.0,
                f"aux_grad_routing/{name}_routed_norm": aux_norm,
            }
        metric_device = next((grad.device for grad in aux if grad is not None), device)
        dot_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        primary_norm_sq_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        aux_norm_sq_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        for p_grad, a_grad in zip(primary, aux, strict=True):
            if p_grad is None or a_grad is None:
                continue
            pf = p_grad.float()
            af = a_grad.float()
            dot_t = dot_t + (pf * af).sum()
            primary_norm_sq_t = primary_norm_sq_t + pf.pow(2).sum()
            aux_norm_sq_t = aux_norm_sq_t + af.pow(2).sum()
        dot = float(dot_t.detach().item())
        primary_norm_sq = float(primary_norm_sq_t.detach().item())
        aux_norm_sq = float(aux_norm_sq_t.detach().item())
        cosine = dot / (math.sqrt(max(primary_norm_sq, 1e-30)) * math.sqrt(max(aux_norm_sq, 1e-30)))
        conflict = dot < 0.0
        coeff = dot / max(primary_norm_sq, 1e-30) if conflict and args.aux_grad_conflict_projection else 0.0
        if args.aux_conflict_controller:
            if conflict:
                route_scale = max(float(args.aux_conflict_damp_min), 1.0 + min(float(cosine), 0.0))
            else:
                route_scale = min(float(args.aux_conflict_boost_max), float(args.aux_grad_aligned_boost) * (1.0 + 0.25 * max(float(cosine), 0.0)))
        else:
            route_scale = float(args.aux_grad_aligned_boost)
        routed: list[Tensor | None] = []
        routed_norm_sq_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        for p_grad, a_grad in zip(primary, aux, strict=True):
            if a_grad is None:
                routed.append(None)
                continue
            if conflict and args.aux_grad_conflict_projection and p_grad is not None:
                r = (route_scale * (a_grad.float() - coeff * p_grad.float())).to(dtype=a_grad.dtype)
            else:
                r = (route_scale * a_grad.float()).to(dtype=a_grad.dtype)
            routed_norm_sq_t = routed_norm_sq_t + r.float().pow(2).sum()
            routed.append(r)
        routed_norm_sq = float(routed_norm_sq_t.detach().item())
        return routed, {
            f"aux_grad_routing/{name}_enabled": 1.0,
            f"aux_grad_routing/{name}_cosine": float(cosine),
            f"aux_grad_routing/{name}_conflict": float(conflict),
            f"aux_grad_routing/{name}_projection_coeff": float(coeff),
            f"aux_grad_routing/{name}_route_scale": float(route_scale),
            f"aux_grad_routing/{name}_controller_enabled": float(args.aux_conflict_controller),
            f"aux_grad_routing/{name}_primary_norm": math.sqrt(max(primary_norm_sq, 0.0)),
            f"aux_grad_routing/{name}_aux_norm": math.sqrt(max(aux_norm_sq, 0.0)),
            f"aux_grad_routing/{name}_routed_norm": math.sqrt(max(routed_norm_sq, 0.0)),
        }

    def split_long_entry_segments(entry: Tensor) -> list[Tensor]:
        segment_len = max(2, int(args.long_entry_segment_len))
        stride = max(1, int(args.long_entry_segment_stride))
        max_segments = int(args.long_entry_max_segments)
        if entry.numel() < 2:
            return []
        starts = list(range(0, max(1, int(entry.numel()) - 1), stride))
        if starts and starts[-1] + segment_len + 1 < int(entry.numel()):
            starts.append(max(0, int(entry.numel()) - segment_len - 1))
        if max_segments > 0 and len(starts) > max_segments:
            chosen = torch.linspace(0, len(starts) - 1, steps=max_segments).round().long().tolist()
            starts = [starts[idx] for idx in chosen]
        segments: list[Tensor] = []
        seen_starts: set[int] = set()
        for start in starts:
            if start in seen_starts:
                continue
            seen_starts.add(start)
            end = min(int(entry.numel()), start + segment_len + 1)
            segment = entry[start:end]
            if segment.numel() >= 2:
                segments.append(segment)
        return segments

    def compute_long_entry_objective(entry_batch: dict[str, object], step_value: int) -> tuple[Tensor, dict[str, float]]:
        zero = next(base_model.parameters()).sum() * 0.0
        entries = entry_batch["entries"]
        if not isinstance(entries, list) or not entries:
            return zero, {"long_entry_full_row/enabled": 1.0, "long_entry_full_row/rows": 0.0}
        segment_losses: list[Tensor] = []
        next_nce_losses: list[Tensor] = []
        boundary_losses: list[Tensor] = []
        coverage_losses: list[Tensor] = []
        fot_losses: list[Tensor] = []
        fot_rewards: list[Tensor] = []
        fot_diversities: list[Tensor] = []
        segment_counts: list[int] = []
        token_count = 0
        byte_count = torch.zeros((), device=device, dtype=torch.float64)
        truncated_count = 0
        original_lengths = entry_batch.get("original_lengths", [])
        lengths = entry_batch.get("lengths", [])
        truncated = entry_batch.get("truncated", [])

        for row_index, entry in enumerate(entries):
            if not torch.is_tensor(entry):
                continue
            if isinstance(truncated, list) and row_index < len(truncated) and bool(truncated[row_index]):
                truncated_count += 1
            segments = split_long_entry_segments(entry)
            segment_counts.append(len(segments))
            summaries: list[Tensor] = []
            first_states: list[Tensor] = []
            last_states: list[Tensor] = []
            row_segment_losses: list[Tensor] = []
            for segment in segments:
                sx = segment[:-1].view(1, -1)
                sy = segment[1:].view(1, -1)
                lm_loss, hidden, segment_nll = base_model.forward_aux(
                    sx,
                    sy,
                    flatten_graph_output=True,
                    activation_checkpoint=bool(args.long_entry_segment_checkpoint),
                )
                row_segment_losses.append(lm_loss)
                if (
                    oai_fot_head is not None
                    and float(args.long_entry_fot_loss_weight) > 0.0
                    and int(args.long_entry_fot_every) > 0
                    and step_value % int(args.long_entry_fot_every) == 0
                    and len(fot_losses) < max(1, int(args.long_entry_fot_max_segments))
                ):
                    segment_bytes = sentencepiece_target_byte_lengths(
                        sx,
                        sy,
                        base_bytes_lut,
                        has_leading_space_lut,
                        is_boundary_token_lut,
                    )
                    lm_weight = base_model.tok_emb.weight if base_model.tie_embeddings else base_model.lm_head.weight
                    fot_control = current_fot_control()
                    fot_out = oai_fot_head(
                        hidden,
                        sy,
                        segment_nll,
                        target_byte_lengths=segment_bytes,
                        lm_head_weight=lm_weight,
                        logit_softcap=float(args.logit_softcap),
                        temperature_multiplier=float(fot_control["temperature_multiplier"]),
                        ucb_multiplier=float(fot_control["ucb_multiplier"]),
                        sparse_multiplier=float(fot_control["sparse_multiplier"]),
                    )
                    fot_losses.append(fot_out["oai_fot_loss"])
                    fot_rewards.append(fot_out["oai_fot_reward_mean"])
                    fot_diversities.append(fot_out["oai_fot_tree_diversity"])
                summaries.append(hidden.float().mean(dim=1).squeeze(0))
                first_states.append(hidden[:, 0, :].float().squeeze(0))
                last_states.append(hidden[:, -1, :].float().squeeze(0))
                token_count += int(sy.numel())
                byte_count = byte_count + count_sentencepiece_target_bytes(
                    sx,
                    sy,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
            if row_segment_losses:
                segment_losses.append(torch.stack(row_segment_losses).mean())
            if len(summaries) >= 2:
                summary = F.normalize(torch.stack(summaries), dim=-1)
                logits = summary[:-1] @ summary[1:].T / 0.20
                targets = torch.arange(logits.shape[0], device=logits.device)
                next_nce_losses.append(F.cross_entropy(logits.float(), targets))
                left = F.normalize(torch.stack(last_states[:-1]), dim=-1)
                right = F.normalize(torch.stack(first_states[1:]), dim=-1)
                boundary_losses.append((1.0 - (left * right).sum(dim=-1)).mean())
                coverage = summary.float().var(dim=0, unbiased=False).mean()
                coverage_losses.append(torch.relu(summary.new_tensor(0.010) - coverage))

        ce = torch.stack(segment_losses).mean() if segment_losses else zero
        next_nce = torch.stack(next_nce_losses).mean() if next_nce_losses else zero
        boundary = torch.stack(boundary_losses).mean() if boundary_losses else zero
        coverage = torch.stack(coverage_losses).mean() if coverage_losses else zero
        long_entry_fot = torch.stack(fot_losses).mean() if fot_losses else zero
        long_entry_fot_reward = torch.stack(fot_rewards).mean() if fot_rewards else zero
        long_entry_fot_diversity = torch.stack(fot_diversities).mean() if fot_diversities else zero
        full_context_loss = zero
        full_context_tokens = 0
        full_context_active = False
        if (
            int(args.long_entry_full_context_every) > 0
            and step_value % int(args.long_entry_full_context_every) == 0
            and float(args.long_entry_full_context_loss_weight) > 0.0
        ):
            full_cap = max(2, int(args.long_entry_full_context_seq_len))
            min_tokens = max(2, int(args.long_entry_full_context_min_tokens))
            candidates = [entry for entry in entries if torch.is_tensor(entry) and int(entry.numel()) >= min_tokens]
            if candidates:
                full_entry = candidates[0][: full_cap + 1]
                if full_entry.numel() >= 2:
                    fx = full_entry[:-1].view(1, -1)
                    fy = full_entry[1:].view(1, -1)
                    full_context_loss, _, _ = base_model.forward_aux(
                        fx,
                        fy,
                        flatten_graph_output=True,
                        activation_checkpoint=bool(args.long_entry_full_context_checkpoint),
                    )
                    full_context_tokens = int(fy.numel())
                    full_context_active = True

        weighted = aux_stage_multiplier * (
            float(args.long_entry_loss_weight) * ce
            + float(args.long_entry_next_segment_weight) * next_nce
            + float(args.long_entry_boundary_weight) * boundary
            + float(args.long_entry_segment_coverage_weight) * coverage
            + float(args.long_entry_fot_loss_weight) * long_entry_fot
            + float(args.long_entry_full_context_loss_weight) * full_context_loss
        )
        bits_per_token = float(ce.detach().float().item() / math.log(2.0)) if segment_losses else 0.0
        tokens_per_byte = float(token_count / max(float(byte_count.detach().float().item()), 1.0))
        original_mean = 0.0
        emitted_mean = 0.0
        if isinstance(original_lengths, list) and original_lengths:
            original_mean = float(sum(int(x) for x in original_lengths) / len(original_lengths))
        if isinstance(lengths, list) and lengths:
            emitted_mean = float(sum(int(x) for x in lengths) / len(lengths))
        metrics = {
            "long_entry_full_row/enabled": 1.0,
            "long_entry_full_row/loss": float(ce.detach().float().item()),
            "long_entry_full_row/weighted_loss": float(weighted.detach().float().item()),
            "long_entry_full_row/bpb": float(bits_per_token * tokens_per_byte),
            "long_entry_full_row/bits_per_token": float(bits_per_token),
            "long_entry_full_row/tokens_per_byte": float(tokens_per_byte),
            "long_entry_full_row/rows": float(len(entries)),
            "long_entry_full_row/segments": float(sum(segment_counts)),
            "long_entry_full_row/segments_per_row": float(sum(segment_counts) / max(1, len(segment_counts))),
            "long_entry_full_row/target_tokens": float(token_count),
            "long_entry_full_row/target_bytes": float(byte_count.detach().float().item()),
            "long_entry_full_row/next_segment_nce": float(next_nce.detach().float().item()),
            "long_entry_full_row/boundary_continuity": float(boundary.detach().float().item()),
            "long_entry_full_row/segment_coverage_loss": float(coverage.detach().float().item()),
            "long_entry_full_row/fot_loss": float(long_entry_fot.detach().float().item()),
            "long_entry_full_row/fot_segments": float(len(fot_losses)),
            "long_entry_full_row/fot_reward_mean": float(long_entry_fot_reward.detach().float().item()),
            "long_entry_full_row/fot_tree_diversity": float(long_entry_fot_diversity.detach().float().item()),
            "long_entry_full_row/truncated_rows": float(truncated_count),
            "long_entry_full_row/original_tokens_mean": original_mean,
            "long_entry_full_row/emitted_tokens_mean": emitted_mean,
            "long_entry_full_row/full_context_active": float(full_context_active),
            "long_entry_full_row/full_context_loss": float(full_context_loss.detach().float().item()),
            "long_entry_full_row/full_context_tokens": float(full_context_tokens),
            "long_entry_full_row/loss_weight": float(args.long_entry_loss_weight * aux_stage_multiplier),
            "long_entry_full_row/next_segment_weight": float(args.long_entry_next_segment_weight * aux_stage_multiplier),
            "long_entry_full_row/boundary_weight": float(args.long_entry_boundary_weight * aux_stage_multiplier),
            "long_entry_full_row/coverage_weight": float(args.long_entry_segment_coverage_weight * aux_stage_multiplier),
            "long_entry_full_row/fot_weight": float(args.long_entry_fot_loss_weight * aux_stage_multiplier),
            "long_entry_full_row/fot_every": float(args.long_entry_fot_every),
            "long_entry_full_row/fot_max_segments": float(args.long_entry_fot_max_segments),
            "long_entry_full_row/full_context_weight": float(args.long_entry_full_context_loss_weight * aux_stage_multiplier),
        }
        return weighted, metrics

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def linear_schedule(step_value: int, start: float, peak: float, warmup_steps: int, hold_steps: int = 0) -> float:
        if warmup_steps <= 0:
            return peak
        if step_value < hold_steps:
            return start
        frac = min(max((step_value - hold_steps) / max(warmup_steps, 1), 0.0), 1.0)
        return start + frac * (peak - start)

    def teacher_weight(step_value: int) -> float:
        if teacher_model is None or args.teacher_distill_weight <= 0.0:
            return 0.0
        if args.teacher_distill_until_step > 0 and step_value >= args.teacher_distill_until_step:
            return 0.0
        peak = float(args.teacher_distill_weight)
        start = float(args.teacher_distill_weight_start)
        if args.teacher_distill_decay_steps <= 0:
            return peak
        frac = min(max(step_value / max(args.teacher_distill_decay_steps, 1), 0.0), 1.0)
        return start + frac * (peak - start)

    def parsed_mtp_offsets() -> list[int]:
        offsets: list[int] = []
        for raw in str(args.oai_mtp_offsets).replace(";", ",").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if value >= 2:
                offsets.append(value)
        return sorted(set(offsets))

    mtp_offsets = parsed_mtp_offsets()
    bpb_control_state: dict[str, float | None] = {
        "last_bpb": None,
        "last_slope": None,
        "slope_ema": None,
    }
    fot_control_state: dict[str, float | None] = {
        "reward_ema": None,
        "entropy_ema": None,
        "diversity_ema": None,
        "loss_multiplier": 1.0,
        "temperature_multiplier": 1.0,
        "ucb_multiplier": 1.0,
        "sparse_multiplier": 1.0,
    }

    def update_bpb_aux_control(step_value: int, train_bpb_value: float) -> dict[str, float]:
        if not args.bpb_first_aux_staging:
            return {
                "enabled": 0.0,
                "stage_multiplier": 1.0,
                "phase_multiplier": 1.0,
                "slope_multiplier": 1.0,
                "curvature_multiplier": 1.0,
                "slope_ema": 0.0,
                "curvature": 0.0,
            }
        min_mult = max(0.0, min(float(args.bpb_first_min_aux_mult), 1.0))
        if step_value < int(args.bpb_first_core_steps):
            phase_multiplier = min_mult
        else:
            frac = min(
                max((step_value - int(args.bpb_first_core_steps)) / max(int(args.bpb_first_ramp_steps), 1), 0.0),
                1.0,
            )
            phase_multiplier = min_mult + frac * (1.0 - min_mult)
        last_bpb = bpb_control_state.get("last_bpb")
        current_slope = 0.0 if last_bpb is None else float(train_bpb_value) - float(last_bpb)
        old_slope_ema = bpb_control_state.get("slope_ema")
        slope_ema = current_slope if old_slope_ema is None else 0.90 * float(old_slope_ema) + 0.10 * current_slope
        last_slope = bpb_control_state.get("last_slope")
        curvature = 0.0 if last_slope is None else slope_ema - float(last_slope)
        slope_multiplier = 1.0
        if args.bpb_first_require_negative_slope and step_value > 20 and slope_ema >= 0.0:
            slope_multiplier = float(args.bpb_first_bad_slope_mult)
        curvature_multiplier = 1.0
        if args.bpb_first_curvature_guard and step_value > 40 and curvature > 0.015:
            curvature_multiplier = float(args.bpb_first_bad_curvature_mult)
        stage_multiplier = max(0.0, min(1.0, phase_multiplier * slope_multiplier * curvature_multiplier))
        bpb_control_state["last_bpb"] = float(train_bpb_value)
        bpb_control_state["last_slope"] = float(slope_ema)
        bpb_control_state["slope_ema"] = float(slope_ema)
        return {
            "enabled": 1.0,
            "stage_multiplier": float(stage_multiplier),
            "phase_multiplier": float(phase_multiplier),
            "slope_multiplier": float(slope_multiplier),
            "curvature_multiplier": float(curvature_multiplier),
            "slope_ema": float(1000.0 * slope_ema),
            "curvature": float(1000.0 * curvature),
        }

    def current_fot_control() -> dict[str, float]:
        return {
            "enabled": float(args.oai_fot_adaptive_control),
            "loss_multiplier": float(fot_control_state.get("loss_multiplier") or 1.0),
            "temperature_multiplier": float(fot_control_state.get("temperature_multiplier") or 1.0),
            "ucb_multiplier": float(fot_control_state.get("ucb_multiplier") or 1.0),
            "sparse_multiplier": float(fot_control_state.get("sparse_multiplier") or 1.0),
            "reward_ema": float(fot_control_state.get("reward_ema") or 0.0),
            "entropy_ema": float(fot_control_state.get("entropy_ema") or 0.0),
            "diversity_ema": float(fot_control_state.get("diversity_ema") or 0.0),
        }

    def update_fot_control_from_metrics(metrics: dict[str, float]) -> dict[str, float]:
        if not args.oai_fot_adaptive_control:
            return current_fot_control()
        reward = float(metrics.get("oai_fot/reward_mean", 0.0))
        entropy = float(metrics.get("oai_fot/activation_entropy", 0.0))
        diversity = float(metrics.get("oai_fot/tree_diversity", 0.0))
        for key, value in (("reward_ema", reward), ("entropy_ema", entropy), ("diversity_ema", diversity)):
            old = fot_control_state.get(key)
            fot_control_state[key] = value if old is None else 0.85 * float(old) + 0.15 * value
        reward_ema = float(fot_control_state["reward_ema"] or 0.0)
        entropy_ema = float(fot_control_state["entropy_ema"] or 0.0)
        diversity_ema = float(fot_control_state["diversity_ema"] or 0.0)
        loss_mult = float(fot_control_state.get("loss_multiplier") or 1.0)
        temp_mult = float(fot_control_state.get("temperature_multiplier") or 1.0)
        ucb_mult = float(fot_control_state.get("ucb_multiplier") or 1.0)
        sparse_mult = float(fot_control_state.get("sparse_multiplier") or 1.0)
        if reward_ema < float(args.oai_fot_reward_target) and entropy_ema > float(args.oai_fot_entropy_high):
            temp_mult *= 0.96
            ucb_mult *= 0.96
            loss_mult *= 0.92
        if reward_ema >= float(args.oai_fot_reward_target) and diversity_ema < float(args.oai_fot_diversity_target):
            ucb_mult *= 1.04
            sparse_mult *= 1.03
            loss_mult *= 1.02
        if reward_ema < 0.5 * float(args.oai_fot_reward_target) and diversity_ema < float(args.oai_fot_diversity_target):
            loss_mult *= 0.88
        if entropy_ema < float(args.oai_fot_entropy_low) and reward_ema >= 0.75 * float(args.oai_fot_reward_target):
            temp_mult *= 1.03
        fot_control_state["loss_multiplier"] = max(0.10, min(1.50, loss_mult))
        fot_control_state["temperature_multiplier"] = max(float(args.oai_fot_temp_min), min(float(args.oai_fot_temp_max), temp_mult))
        fot_control_state["ucb_multiplier"] = max(float(args.oai_fot_ucb_min), min(float(args.oai_fot_ucb_max), ucb_mult))
        fot_control_state["sparse_multiplier"] = max(0.40, min(1.80, sparse_mult))
        return current_fot_control()

    def save_checkpoint(step_value: int, val_loss_value: float | None = None, val_bpb_value: float | None = None) -> None:
        if not master_process or args.checkpoint_every <= 0:
            return
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step_value),
            "model": base_model.state_dict(),
            "sidecar": sidecar.state_dict() if sidecar is not None else None,
            "oai_gflownet": oai_gflownet_head.state_dict() if oai_gflownet_head is not None else None,
            "oai_fot": oai_fot_head.state_dict() if oai_fot_head is not None else None,
            "structure_head": structure_head.state_dict() if structure_head is not None else None,
            "toricgt_sidecar_enabled": bool(sidecar is not None),
            "oai_gflownet_enabled": bool(oai_gflownet_head is not None),
            "oai_embedding_fot_enabled": bool(oai_fot_head is not None),
            "toricblm_structure_flow_enabled": bool(structure_head is not None),
            "graph_lm_primary_enabled": bool(graph_lm_loader is not None),
            "long_entry_full_row_enabled": bool(long_entry_loader is not None),
            "graph_output_flattening_enabled": bool(base_model.graph_output_flattening is not None),
            "val_loss": val_loss_value,
            "val_bpb": val_bpb_value,
            "run_id": args.run_id,
            "hyperparameters": {
                key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool))
            },
        }
        path = ckpt_dir / f"{args.run_id}_step_{int(step_value):06d}.pt"
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        log0(f"checkpoint_saved:{path} step:{step_value} val_bpb:{val_bpb_value}")

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0 and args.start_step <= 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, token_map=caseops_token_map)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = int(args.start_step)
    while True:
        last_step = step >= args.iterations or (stop_after_step is not None and step >= stop_after_step)

        cadence_validate = args.val_loss_every > 0 and step % args.val_loss_every == 0
        initial_validate = step == args.start_step
        should_validate = last_step or (cadence_validate and (args.eval_initial or not initial_validate))
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(step - args.start_step, 1):.2f}ms"
            )
            wandb_log0(
                {
                    "val/loss": float(val_loss),
                    "val/bpb": float(val_bpb),
                    "oai_competition/bpb": float(val_bpb),
                    "bpb/oai_competition": float(val_bpb),
                    "00_primary/val_bpb": float(val_bpb),
                    "score_first_tta/eval_enabled": float(args.score_first_tta),
                    "score_first_tta/eval_steps": float(args.score_first_tta_steps),
                    "score_first_tta/eval_lr": float(args.score_first_tta_lr),
                    "score_first_tta/eval_commit": float(args.score_first_tta_commit),
                },
                step,
            )
            save_checkpoint(step, val_loss, val_bpb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        train_token_count = torch.zeros((), device=device, dtype=torch.float64)
        train_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        graph_lm_metrics: dict[str, float] = {}
        long_entry_metrics: dict[str, float] = {}
        sidecar_metrics: dict[str, float] = {}
        teacher_metrics: dict[str, float] = {}
        oai_gflownet_metrics: dict[str, float] = {}
        oai_fot_metrics: dict[str, float] = {}
        oai_mtp_metrics: dict[str, float] = {}
        structure_metrics: dict[str, float] = {}
        aux_grad_metrics: dict[str, float] = {}
        bpb_aux_metrics: dict[str, float] = {}
        convextok_metrics: dict[str, float] = {}
        last_fineweb_x: Tensor | None = None
        last_fineweb_y: Tensor | None = None
        flattening_calibration_metrics: dict[str, float] = {}
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            last_fineweb_x = x
            last_fineweb_y = y
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            batch_token_count = float(y.numel())
            train_loss_sum += loss.detach().to(torch.float64) * batch_token_count
            train_token_count += batch_token_count
            train_byte_count += count_sentencepiece_target_bytes(
                x,
                y,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            (loss * grad_scale).backward()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_byte_count, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_sum / train_token_count.clamp_min(1.0)).to(dtype=torch.float32)
        train_bits_per_token = float(train_loss.detach().float().item() / math.log(2.0))
        train_tokens_per_byte = float((train_token_count / train_byte_count.clamp_min(1.0)).detach().float().item())
        train_bpb = float(train_bits_per_token * train_tokens_per_byte)
        bpb_aux_control = update_bpb_aux_control(step, train_bpb)
        aux_stage_multiplier = float(bpb_aux_control["stage_multiplier"])
        bpb_aux_metrics = {f"bpb_aux_control/{key}": float(value) for key, value in bpb_aux_control.items()}
        primary_grads = snapshot_base_grads() if args.aux_grad_routing else None

        if tokenizer_kind == "convextok" and args.convextok_toric_reg_weight > 0.0:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                convextok_toric_reg = base_model.convextok_toric_regularizer(
                    topk=max(2, int(args.convextok_toric_reg_topk))
                )
                convextok_toric_weight = aux_stage_multiplier * float(args.convextok_toric_reg_weight)
                weighted_convextok_toric_reg = convextok_toric_weight * convextok_toric_reg
            weighted_convextok_toric_reg.backward()
            convextok_metrics = {
                "convextok_toric/regularizer": float(convextok_toric_reg.detach().float().item()),
                "convextok_toric/regularizer_weight": float(convextok_toric_weight),
                "convextok_toric/regularizer_topk": float(args.convextok_toric_reg_topk),
            }

        if (
            oai_gflownet_head is not None
            and args.oai_gflownet_every > 0
            and step % args.oai_gflownet_every == 0
            and last_fineweb_x is not None
            and last_fineweb_y is not None
            and args.oai_gflownet_loss_weight > 0.0
        ):
            seq_count = min(max(1, int(args.oai_gflownet_max_sequences)), int(last_fineweb_x.shape[0]))
            gx = last_fineweb_x[:seq_count]
            gy = last_fineweb_y[:seq_count]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, g_hidden, g_nll = base_model.forward_aux(gx, gy, flatten_graph_output=True)
                g_out = oai_gflownet_head(
                    g_hidden,
                    gy,
                    g_nll,
                    max_positions=int(args.oai_gflownet_max_positions),
                )
                g_entropy = g_out["oai_gflownet_entropy"].float()
                g_entropy_objective = (g_entropy - float(args.oai_gflownet_entropy_target)).pow(2)
                weighted_gflownet_loss = (
                    aux_stage_multiplier * float(args.oai_gflownet_loss_weight) * g_out["oai_gflownet_loss"]
                    + aux_stage_multiplier * float(args.oai_gflownet_entropy_weight) * g_entropy_objective
                )
            if args.aux_grad_routing:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_gflownet_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="oai_gflownet")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_gflownet_loss.backward()
            oai_gflownet_metrics = {
                "oai_gflownet/enabled": 1.0,
                "oai_gflownet/loss": float(g_out["oai_gflownet_loss"].detach().float().item()),
                "oai_gflownet/weighted_loss": float(weighted_gflownet_loss.detach().float().item()),
                "oai_gflownet/loss_weight": float(args.oai_gflownet_loss_weight * aux_stage_multiplier),
                "oai_gflownet/loss_weight_config": float(args.oai_gflownet_loss_weight),
                "oai_gflownet/bpb_stage_multiplier": float(aux_stage_multiplier),
                "oai_gflownet/entropy": float(g_entropy.detach().float().item()),
                "oai_gflownet/entropy_weight": float(args.oai_gflownet_entropy_weight),
                "oai_gflownet/entropy_target": float(args.oai_gflownet_entropy_target),
                "oai_gflownet/entropy_objective": float(g_entropy_objective.detach().float().item()),
                "oai_gflownet/action_diversity": float(
                    g_out["oai_gflownet_action_diversity"].detach().float().item()
                ),
                "oai_gflownet/reward_mean": float(g_out["oai_gflownet_reward_mean"].detach().float().item()),
                "oai_gflownet/tb_residual": float(g_out["oai_gflownet_tb_residual"].detach().float().item()),
                "oai_gflownet/score_gap": float(g_out["oai_gflownet_score_gap"].detach().float().item()),
                "oai_gflownet/log_z": float(g_out["oai_gflownet_log_z"].detach().float().item()),
                "oai_gflownet/sequences": float(seq_count),
                "oai_gflownet/max_positions": float(args.oai_gflownet_max_positions),
                "oai_gflownet/num_actions": float(args.oai_gflownet_num_actions),
            }

        if (
            oai_fot_head is not None
            and args.oai_fot_every > 0
            and step % args.oai_fot_every == 0
            and last_fineweb_x is not None
            and last_fineweb_y is not None
            and args.oai_fot_loss_weight > 0.0
        ):
            seq_count = min(max(1, int(args.oai_fot_max_sequences)), int(last_fineweb_x.shape[0]))
            fx = last_fineweb_x[:seq_count]
            fy = last_fineweb_y[:seq_count]
            advanced_signal = None
            if oai_gflownet_metrics:
                advanced_signal = torch.full(
                    (seq_count,),
                    float(oai_gflownet_metrics.get("oai_gflownet/reward_mean", 0.0)),
                    device=device,
                    dtype=torch.float32,
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, f_hidden, f_nll = base_model.forward_aux(fx, fy, flatten_graph_output=True)
                f_bytes = sentencepiece_target_byte_lengths(
                    fx,
                    fy,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                fot_control_before = current_fot_control()
                lm_weight = base_model.tok_emb.weight if base_model.tie_embeddings else base_model.lm_head.weight
                f_out = oai_fot_head(
                    f_hidden,
                    fy,
                    f_nll,
                    advanced_signal=advanced_signal,
                    target_byte_lengths=f_bytes,
                    lm_head_weight=lm_weight,
                    logit_softcap=float(args.logit_softcap),
                    temperature_multiplier=float(fot_control_before["temperature_multiplier"]),
                    ucb_multiplier=float(fot_control_before["ucb_multiplier"]),
                    sparse_multiplier=float(fot_control_before["sparse_multiplier"]),
                )
                effective_fot_weight = (
                    aux_stage_multiplier
                    * float(args.oai_fot_loss_weight)
                    * float(fot_control_before["loss_multiplier"])
                )
                weighted_fot_loss = effective_fot_weight * f_out["oai_fot_loss"]
            if args.aux_grad_routing:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_fot_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="oai_fot")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_fot_loss.backward()
            oai_fot_metrics = {
                "oai_fot/enabled": 1.0,
                "oai_fot/loss": float(f_out["oai_fot_loss"].detach().float().item()),
                "oai_fot/weighted_loss": float(weighted_fot_loss.detach().float().item()),
                "oai_fot/loss_weight": float(effective_fot_weight),
                "oai_fot/loss_weight_config": float(args.oai_fot_loss_weight),
                "oai_fot/bpb_stage_multiplier": float(aux_stage_multiplier),
                "oai_fot/adaptive_control_enabled": float(args.oai_fot_adaptive_control),
                "oai_fot/runtime_loss_multiplier": float(fot_control_before["loss_multiplier"]),
                "oai_fot/runtime_temperature_multiplier": float(fot_control_before["temperature_multiplier"]),
                "oai_fot/runtime_ucb_multiplier": float(fot_control_before["ucb_multiplier"]),
                "oai_fot/runtime_sparse_multiplier": float(fot_control_before["sparse_multiplier"]),
                "oai_fot/sparse_activation_loss": float(
                    f_out["oai_fot_sparse_activation_loss"].detach().float().item()
                ),
                "oai_fot/ucb_loss": float(f_out["oai_fot_ucb_loss"].detach().float().item()),
                "oai_fot/self_correction_loss": float(
                    f_out["oai_fot_self_correction_loss"].detach().float().item()
                ),
                "oai_fot/consensus_loss": float(f_out["oai_fot_consensus_loss"].detach().float().item()),
                "oai_fot/tb_loss": float(f_out["oai_fot_tb_loss"].detach().float().item()),
                "oai_fot/subtb_loss": float(f_out["oai_fot_subtb_loss"].detach().float().item()),
                "oai_fot/complexity_loss": float(f_out["oai_fot_complexity_loss"].detach().float().item()),
                "oai_fot/activation_entropy": float(
                    f_out["oai_fot_activation_entropy"].detach().float().item()
                ),
                "oai_fot/active_tree_count": float(f_out["oai_fot_active_tree_count"].detach().float().item()),
                "oai_fot/active_mass_topk": float(f_out["oai_fot_active_mass_topk"].detach().float().item()),
                "oai_fot/tree_diversity": float(f_out["oai_fot_tree_diversity"].detach().float().item()),
                "oai_fot/value_mean": float(f_out["oai_fot_value_mean"].detach().float().item()),
                "oai_fot/reward_mean": float(f_out["oai_fot_reward_mean"].detach().float().item()),
                "oai_fot/reward_bpb_delta": float(
                    f_out["oai_fot_reward_bpb_delta"].detach().float().item()
                ),
                "oai_fot/reward_raw_byte_nll": float(
                    f_out["oai_fot_reward_raw_byte_nll"].detach().float().item()
                ),
                "oai_fot/reward_corrected_byte_nll": float(
                    f_out["oai_fot_reward_corrected_byte_nll"].detach().float().item()
                ),
                "oai_fot/corrected_ce": float(f_out["oai_fot_corrected_ce"].detach().float().item()),
                "oai_fot/tb_residual": float(f_out["oai_fot_tb_residual"].detach().float().item()),
                "oai_fot/correction_cosine": float(
                    f_out["oai_fot_correction_cosine"].detach().float().item()
                ),
                "oai_fot/correction_bpb_proxy_lift": float(
                    f_out["oai_fot_correction_bpb_proxy_lift"].detach().float().item()
                ),
                "oai_fot/consensus_margin": float(f_out["oai_fot_consensus_margin"].detach().float().item()),
                "oai_fot/consensus_entropy": float(f_out["oai_fot_consensus_entropy"].detach().float().item()),
                "oai_fot/consensus_tree_agreement": float(
                    f_out["oai_fot_consensus_tree_agreement"].detach().float().item()
                ),
                "oai_fot/log_z": float(f_out["oai_fot_log_z"].detach().float().item()),
                "oai_fot/sequences": float(seq_count),
                "oai_fot/max_positions": float(args.oai_fot_max_positions),
                "oai_fot/num_trees": float(f_out["oai_fot_num_trees"].detach().float().item()),
                "oai_fot/node_count": float(f_out["oai_fot_node_count"].detach().float().item()),
                "oai_fot/topk_trees": float(f_out["oai_fot_topk_trees"].detach().float().item()),
            }
            oai_fot_metrics.update(
                {
                    f"oai_fot/runtime_after_{key}": value
                    for key, value in update_fot_control_from_metrics(oai_fot_metrics).items()
                }
            )

        if (
            args.oai_mtp
            and args.oai_mtp_every > 0
            and step % args.oai_mtp_every == 0
            and args.oai_mtp_loss_weight > 0.0
            and mtp_offsets
            and last_fineweb_x is not None
            and last_fineweb_y is not None
        ):
            seq_count = min(max(1, int(args.oai_mtp_max_sequences)), int(last_fineweb_x.shape[0]))
            mx = last_fineweb_x[:seq_count]
            my = last_fineweb_y[:seq_count]
            mtp_losses: list[Tensor] = []
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                mtp_hidden = base_model._hidden(mx, flatten_graph_output=True)
                for offset in mtp_offsets:
                    if offset > mtp_hidden.shape[1]:
                        continue
                    mtp_target = my[:, offset - 1 :]
                    mtp_source = mtp_hidden[:, : mtp_target.shape[1], :]
                    mtp_logits = base_model._logits_from_hidden(mtp_source)
                    mtp_losses.append(F.cross_entropy(mtp_logits.float(), mtp_target.reshape(-1)))
                mtp_loss = torch.stack(mtp_losses).mean() if mtp_losses else mtp_hidden.new_zeros(())
                weighted_mtp_loss = aux_stage_multiplier * float(args.oai_mtp_loss_weight) * mtp_loss
            if float(weighted_mtp_loss.detach().float().item()) > 0.0:
                if args.aux_grad_routing:
                    saved_base_grads = snapshot_base_grads()
                    zero_base_grads()
                    weighted_mtp_loss.backward()
                    aux = snapshot_base_grads()
                    routed, metrics = route_aux_grads(primary_grads, aux, name="oai_mtp")
                    restore_base_grads(saved_base_grads)
                    add_base_grads(routed)
                    aux_grad_metrics.update(metrics)
                else:
                    weighted_mtp_loss.backward()
            oai_mtp_metrics = {
                "oai_mtp/enabled": 1.0,
                "oai_mtp/loss": float(mtp_loss.detach().float().item()),
                "oai_mtp/weighted_loss": float(weighted_mtp_loss.detach().float().item()),
                "oai_mtp/loss_weight": float(args.oai_mtp_loss_weight * aux_stage_multiplier),
                "oai_mtp/loss_weight_config": float(args.oai_mtp_loss_weight),
                "oai_mtp/bpb_stage_multiplier": float(aux_stage_multiplier),
                "oai_mtp/offset_count": float(len(mtp_losses)),
                "oai_mtp/max_offset": float(max(mtp_offsets) if mtp_offsets else 0),
                "oai_mtp/sequences": float(seq_count),
            }

        if (
            args.graph_output_calibration_loss_weight > 0.0
            and last_fineweb_x is not None
            and last_fineweb_y is not None
            and args.graph_output_calibration_every > 0
            and step % args.graph_output_calibration_every == 0
        ):
            seq_count = min(
                max(1, int(args.graph_output_calibration_max_sequences)),
                int(last_fineweb_x.shape[0]),
            )
            cx = last_fineweb_x[:seq_count]
            cy = last_fineweb_y[:seq_count]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                calibration_loss, calibration_out = base_model.graph_output_flattening_calibration(
                    cx,
                    cy,
                    margin=float(args.graph_output_calibration_margin),
                )
                weighted_calibration_loss = float(args.graph_output_calibration_loss_weight) * calibration_loss
            if float(weighted_calibration_loss.detach().float().item()) > 0.0:
                weighted_calibration_loss.backward()
            flattening_calibration_metrics = {
                "graph_output_flattening/calibration_loss": float(calibration_loss.detach().float().item()),
                "graph_output_flattening/calibration_weighted_loss": float(
                    weighted_calibration_loss.detach().float().item()
                ),
                "graph_output_flattening/calibration_raw_ce": float(
                    calibration_out["graph_output_flattening_raw_ce"].detach().float().item()
                ),
                "graph_output_flattening/calibration_flat_ce": float(
                    calibration_out["graph_output_flattening_flat_ce"].detach().float().item()
                ),
                "graph_output_flattening/ce_lift": float(
                    calibration_out["graph_output_flattening_ce_lift"].detach().float().item()
                ),
                "graph_output_flattening/ce_regression": float(
                    calibration_out["graph_output_flattening_ce_regression"].detach().float().item()
                ),
                "graph_output_flattening/calibration_margin": float(args.graph_output_calibration_margin),
                "graph_output_flattening/calibration_weight": float(args.graph_output_calibration_loss_weight),
                "graph_output_flattening/calibration_sequences": float(seq_count),
            }

        current_teacher_weight = teacher_weight(step)
        if teacher_model is not None and current_teacher_weight > 0.0 and last_fineweb_x is not None:
            teacher_model.eval()
            seq_count = min(max(1, int(args.teacher_distill_max_sequences)), last_fineweb_x.shape[0])
            dx = last_fineweb_x[:seq_count]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                student_logits = base_model.logits_for_distill(dx, flatten_graph_output=True)
                with torch.no_grad():
                    teacher_logits = teacher_model.logits_for_distill(dx, flatten_graph_output=True)
                temp = max(float(args.teacher_distill_temperature), 1e-4)
                teacher_flat = teacher_logits.float().reshape(-1, teacher_logits.shape[-1])
                student_flat = student_logits.float().reshape(-1, student_logits.shape[-1])
                teacher_probs = torch.softmax(teacher_flat / temp, dim=-1)
                student_log_probs = torch.log_softmax(student_flat / temp, dim=-1)
                teacher_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
                weighted_teacher_loss = float(current_teacher_weight) * teacher_kl
            if args.aux_grad_routing and args.aux_grad_route_teacher:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_teacher_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="teacher")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_teacher_loss.backward()
            teacher_metrics = {
                "teacher_distill/enabled": 1.0,
                "teacher_distill/kl": float(teacher_kl.detach().float().item()),
                "teacher_distill/weighted_loss": float(weighted_teacher_loss.detach().float().item()),
                "teacher_distill/weight": float(current_teacher_weight),
                "teacher_distill/temperature": float(args.teacher_distill_temperature),
                "teacher_distill/sequences": float(seq_count),
            }

        scheduled_graph_lm_weight = linear_schedule(
            step,
            args.graph_lm_loss_weight_start,
            args.graph_lm_loss_weight,
            args.graph_lm_warmup_steps,
            args.graph_lm_hold_steps,
        )
        if (
            graph_lm_loader is not None
            and args.graph_lm_every > 0
            and step % args.graph_lm_every == 0
            and scheduled_graph_lm_weight > 0
        ):
            gx, gy = graph_lm_loader.next_batch(device, step=step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                graph_lm_loss, _, _ = base_model.forward_aux(gx, gy)
                effective_graph_lm_weight = aux_stage_multiplier * float(scheduled_graph_lm_weight)
                weighted_graph_lm_loss = effective_graph_lm_weight * graph_lm_loss
            graph_byte_count = count_sentencepiece_target_bytes(
                gx,
                gy,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            graph_bits_per_token = float(graph_lm_loss.detach().float().item() / math.log(2.0))
            graph_tokens_per_byte = float(gy.numel() / graph_byte_count.detach().float().clamp_min(1.0).item())
            graph_lm_bpb = float(graph_bits_per_token * graph_tokens_per_byte)
            if args.aux_grad_routing and args.aux_grad_route_graph_lm:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_graph_lm_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="graph_lm")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_graph_lm_loss.backward()
            graph_lm_metrics = {
                "graph_lm_primary/loss": float(graph_lm_loss.detach().float().item()),
                "graph_lm_primary/weighted_loss": float(weighted_graph_lm_loss.detach().float().item()),
                "graph_lm_primary/bpb": float(graph_lm_bpb),
                "graph_lm_primary/bits_per_token": float(graph_bits_per_token),
                "graph_lm_primary/tokens_per_byte": float(graph_tokens_per_byte),
                "graph_lm_primary/target_tokens": float(gy.numel()),
                "graph_lm_primary/target_bytes": float(graph_byte_count.detach().float().item()),
                "graph_lm_primary/loss_weight": float(effective_graph_lm_weight),
                "graph_lm_primary/scheduled_loss_weight": float(scheduled_graph_lm_weight),
                "graph_lm_primary/bpb_stage_multiplier": float(aux_stage_multiplier),
                "graph_lm_primary/loss_weight_peak": float(args.graph_lm_loss_weight),
                "graph_lm_primary/loss_weight_start": float(args.graph_lm_loss_weight_start),
                "graph_lm_primary/warmup_steps": float(args.graph_lm_warmup_steps),
                "graph_lm_primary/seq_len": float(args.graph_lm_seq_len),
                "graph_lm_primary/batch_size": float(args.graph_lm_batch_size),
                "graph_lm_primary/late_stage_active": float(
                    getattr(graph_lm_loader, "last_stream", "base") == "late"
                ),
            }

        if (
            long_entry_loader is not None
            and args.long_entry_every > 0
            and step % args.long_entry_every == 0
            and (
                args.long_entry_loss_weight
                + args.long_entry_next_segment_weight
                + args.long_entry_boundary_weight
                + args.long_entry_segment_coverage_weight
                + args.long_entry_fot_loss_weight
                + args.long_entry_full_context_loss_weight
            )
            > 0.0
        ):
            entry_batch = long_entry_loader.next_entries(device, step=step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                weighted_long_entry_loss, long_entry_metrics = compute_long_entry_objective(entry_batch, step)
            if args.aux_grad_routing and args.aux_grad_route_long_entry:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_long_entry_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="long_entry")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_long_entry_loss.backward()

        if (
            structure_loader is not None
            and structure_head is not None
            and args.toricblm_structure_flow
            and args.toricblm_structure_every > 0
            and step >= args.toricblm_structure_flow_start_step
            and step % args.toricblm_structure_every == 0
            and (
                args.toricblm_structure_flow_weight
                + args.toricblm_structure_contact_weight
                + args.toricblm_structure_distogram_weight
            )
            > 0.0
        ):
            sx, sy, target_coords, coord_mask, plddt = structure_loader.next_batch(device, step=step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                structure_lm_loss, structure_hidden, _structure_nll = base_model.forward_aux(
                    sx,
                    sy,
                    flatten_graph_output=False,
                )
                structure_out = structure_head(
                    structure_hidden,
                    target_coords,
                    coord_mask,
                    flow_weight=float(args.toricblm_structure_flow_weight),
                    contact_weight=float(args.toricblm_structure_contact_weight),
                    distogram_weight=float(args.toricblm_structure_distogram_weight),
                )
                weighted_structure_loss = aux_stage_multiplier * structure_out["structure_loss"]
            if args.aux_grad_routing and args.aux_grad_route_structure:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_structure_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="structure")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_structure_loss.backward()
            structure_metrics = {
                "toricblm_structure/head_enabled": 1.0,
                "toricblm_structure/coordinate_losses_active": 1.0,
                "toricblm_structure/lm_loss": float(structure_lm_loss.detach().float().item()),
                "toricblm_structure/loss": float(structure_out["structure_loss"].detach().float().item()),
                "toricblm_structure/weighted_loss": float(weighted_structure_loss.detach().float().item()),
                "toricblm_structure/flow_matching_loss": float(
                    structure_out["flow_matching_loss"].detach().float().item()
                ),
                "toricblm_structure/contact_bce_loss": float(
                    structure_out["contact_bce_loss"].detach().float().item()
                ),
                "toricblm_structure/distogram_loss": float(
                    structure_out["distogram_loss"].detach().float().item()
                ),
                "toricblm_structure/rmsd": float(structure_out["rmsd"].detach().float().item()),
                "toricblm_structure/coordinate_count_batch": float(
                    structure_out["coordinate_count"].detach().float().item()
                ),
                "toricblm_structure/batch_size": float(args.toricblm_structure_batch_size),
                "toricblm_structure/max_atoms": float(args.toricblm_structure_max_atoms),
                "toricblm_structure/every": float(args.toricblm_structure_every),
                "toricblm_structure/bpb_stage_multiplier": float(aux_stage_multiplier),
                "toricblm_structure/mean_plddt_batch": float(
                    plddt[coord_mask].detach().float().mean().item() if bool(coord_mask.any()) else 0.0
                ),
            }

        if sidecar is not None and graph_loader is not None and args.sidecar_every > 0 and step % args.sidecar_every == 0:
            sidecar.train()
            sx, sy = graph_loader.next_batch(device, step=step)
            scheduled_sidecar_weight = linear_schedule(
                step,
                args.sidecar_loss_weight_start,
                args.sidecar_loss_weight,
                args.sidecar_warmup_steps,
                args.sidecar_hold_steps,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                sidecar_lm_loss, sidecar_hidden, sidecar_nll = base_model.forward_aux(sx, sy)
                positions = torch.arange(sx.shape[1], device=device, dtype=torch.float32).unsqueeze(0).expand(sx.shape[0], -1)
                sidecar_out = sidecar(sidecar_hidden, sy, positions, sidecar_nll, step=step)
                effective_sidecar_weight = aux_stage_multiplier * float(scheduled_sidecar_weight)
                sidecar_loss = effective_sidecar_weight * sidecar_out["toricgt_sidecar_loss"]
            if args.aux_grad_routing and args.aux_grad_route_sidecar:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                sidecar_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="sidecar")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                sidecar_loss.backward()
            sidecar_metrics = {
                "toricgt_sidecar/lm_loss": float(sidecar_lm_loss.detach().float().item()),
                "toricgt_sidecar/loss": float(sidecar_loss.detach().float().item()),
                "toricgt_sidecar/loss_weight": float(effective_sidecar_weight),
                "toricgt_sidecar/scheduled_loss_weight": float(scheduled_sidecar_weight),
                "toricgt_sidecar/bpb_stage_multiplier": float(aux_stage_multiplier),
                "toricgt_sidecar/loss_weight_peak": float(args.sidecar_loss_weight),
                "toricgt_sidecar/loss_weight_start": float(args.sidecar_loss_weight_start),
                "toricgt_sidecar/warmup_steps": float(args.sidecar_warmup_steps),
                "toricgt_sidecar/hold_steps": float(args.sidecar_hold_steps),
                "toricgt_sidecar/late_stage_active": float(getattr(graph_loader, "last_stream", "base") == "late"),
            }
            for name, value in sidecar_out.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    sidecar_metrics[f"toricgt_sidecar/{name}"] = float(value.detach().float().item())
            if args.require_toricgt_sidecar:
                required = (
                    "toricgt_sidecar/graphcg_loss",
                    "toricgt_sidecar/analogy_lattice_loss",
                    "toricgt_sidecar/tokengt_graph_loss",
                    "toricgt_sidecar/trajectory_memory_loss",
                    "toricgt_sidecar/toric_geometry_loss",
                    "toricgt_sidecar/toric_vector_bundle_1d_cone_ce_loss",
                    "toricgt_sidecar/toric_bgg_loss",
                    "toricgt_sidecar/koszul_persistence_loss",
                    "toricgt_sidecar/toric_cca_topology_loss",
                    "toricgt_sidecar/derived_signature_loss",
                )
                missing = [name for name in required if name not in sidecar_metrics]
                if missing:
                    raise RuntimeError(f"ToricGT sidecar missing required active metrics: {missing}")

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            graph_lm_brief = ""
            if graph_lm_metrics:
                graph_lm_brief = (
                    f" graph_lm_loss:{graph_lm_metrics.get('graph_lm_primary/loss', 0.0):.6f}"
                    f" graph_lm_bpb:{graph_lm_metrics.get('graph_lm_primary/bpb', 0.0):.4f}"
                    f" graph_lm_w:{graph_lm_metrics.get('graph_lm_primary/loss_weight', 0.0):.4f}"
                )
            long_entry_brief = ""
            if long_entry_metrics:
                long_entry_brief = (
                    f" long_entry_loss:{long_entry_metrics.get('long_entry_full_row/loss', 0.0):.6f}"
                    f" long_entry_bpb:{long_entry_metrics.get('long_entry_full_row/bpb', 0.0):.4f}"
                    f" long_entry_segments:{long_entry_metrics.get('long_entry_full_row/segments', 0.0):.0f}"
                    f" long_entry_full:{long_entry_metrics.get('long_entry_full_row/full_context_active', 0.0):.0f}"
                )
            teacher_brief = ""
            if teacher_metrics:
                teacher_brief = (
                    f" teacher_kl:{teacher_metrics.get('teacher_distill/kl', 0.0):.6f}"
                    f" teacher_w:{teacher_metrics.get('teacher_distill/weight', 0.0):.4f}"
                )
            gflownet_brief = ""
            if oai_gflownet_metrics:
                gflownet_brief = (
                    f" oai_gfn:{oai_gflownet_metrics.get('oai_gflownet/loss', 0.0):.6f}"
                    f" gfn_H:{oai_gflownet_metrics.get('oai_gflownet/entropy', 0.0):.4f}"
                    f" gfn_R:{oai_gflownet_metrics.get('oai_gflownet/reward_mean', 0.0):.4f}"
                )
            fot_brief = ""
            if oai_fot_metrics:
                fot_brief = (
                    f" oai_fot:{oai_fot_metrics.get('oai_fot/loss', 0.0):.6f}"
                    f" fot_H:{oai_fot_metrics.get('oai_fot/activation_entropy', 0.0):.4f}"
                    f" fot_R:{oai_fot_metrics.get('oai_fot/reward_mean', 0.0):.4f}"
                    f" fot_div:{oai_fot_metrics.get('oai_fot/tree_diversity', 0.0):.4f}"
                )
            mtp_brief = ""
            if oai_mtp_metrics:
                mtp_brief = (
                    f" mtp:{oai_mtp_metrics.get('oai_mtp/loss', 0.0):.6f}"
                    f" mtp_w:{oai_mtp_metrics.get('oai_mtp/loss_weight', 0.0):.4f}"
                )
            convextok_brief = ""
            if convextok_metrics:
                convextok_brief = (
                    f" convextok_toric:{convextok_metrics.get('convextok_toric/regularizer', 0.0):.6f}"
                    f" convextok_w:{convextok_metrics.get('convextok_toric/regularizer_weight', 0.0):.6f}"
                )
            structure_brief = ""
            if structure_metrics:
                structure_brief = (
                    f" structure_loss:{structure_metrics.get('toricblm_structure/loss', 0.0):.6f}"
                    f" structure_rmsd:{structure_metrics.get('toricblm_structure/rmsd', 0.0):.4f}"
                    f" structure_coords:{structure_metrics.get('toricblm_structure/coordinate_count_batch', 0.0):.0f}"
                )
            sidecar_brief = ""
            if sidecar_metrics:
                sidecar_brief = (
                    f" sidecar_loss:{sidecar_metrics.get('toricgt_sidecar/loss', 0.0):.6f}"
                    f" graphcg:{sidecar_metrics.get('toricgt_sidecar/graphcg_loss', 0.0):.6f}"
                    f" analogy:{sidecar_metrics.get('toricgt_sidecar/analogy_lattice_loss', 0.0):.6f}"
                    f" tokengt_graph:{sidecar_metrics.get('toricgt_sidecar/tokengt_graph_loss', 0.0):.6f}"
                    f" memory:{sidecar_metrics.get('toricgt_sidecar/trajectory_memory_loss', 0.0):.6f}"
                    f" toric:{sidecar_metrics.get('toricgt_sidecar/toric_geometry_loss', 0.0):.6f}"
                    f" vb1d:{sidecar_metrics.get('toricgt_sidecar/toric_vector_bundle_1d_cone_ce_loss', 0.0):.6f}"
                    f" bgg:{sidecar_metrics.get('toricgt_sidecar/toric_bgg_loss', 0.0):.6f}"
                    f" koszul:{sidecar_metrics.get('toricgt_sidecar/koszul_persistence_loss', 0.0):.6f}"
                    f" cca:{sidecar_metrics.get('toricgt_sidecar/toric_cca_topology_loss', 0.0):.6f}"
                    f" derived:{sidecar_metrics.get('toricgt_sidecar/derived_signature_loss', 0.0):.6f}"
                )
            graph_equiv_bpb_terms = [
                float(value)
                for value in (
                    graph_lm_metrics.get("graph_lm_primary/bpb"),
                    long_entry_metrics.get("long_entry_full_row/bpb"),
                )
                if value is not None and math.isfinite(float(value)) and float(value) > 0.0
            ]
            graph_equiv_bpb = (
                sum(graph_equiv_bpb_terms) / max(1, len(graph_equiv_bpb_terms))
                if graph_equiv_bpb_terms
                else float("nan")
            )
            graph_equiv_primary_bpb = float(
                long_entry_metrics.get(
                    "long_entry_full_row/bpb",
                    graph_lm_metrics.get("graph_lm_primary/bpb", float("nan")),
                )
            )
            graph_equiv_brief = ""
            if math.isfinite(graph_equiv_bpb):
                graph_equiv_brief = f" graph_equiv_bpb:{graph_equiv_bpb:.4f}"
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_bpb:{train_bpb:.4f} train_bpt:{train_bits_per_token:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms "
                f"step_avg:{approx_training_time_ms / max(step - args.start_step, 1):.2f}ms"
                f" aux_stage:{bpb_aux_metrics.get('bpb_aux_control/stage_multiplier', 1.0):.4f}"
                f"{graph_lm_brief}"
                f"{long_entry_brief}"
                f"{teacher_brief}"
                f"{gflownet_brief}"
                f"{fot_brief}"
                f"{mtp_brief}"
                f"{convextok_brief}"
                f"{structure_brief}"
                f"{graph_equiv_brief}"
                f"{sidecar_brief}"
            )
            payload = {
                "train/loss": float(train_loss.item()),
                "train/bpb": float(train_bpb),
                "train/bits_per_token": float(train_bits_per_token),
                "train/tokens_per_byte": float(train_tokens_per_byte),
                "train/target_tokens": float(train_token_count.detach().float().item()),
                "train/target_bytes": float(train_byte_count.detach().float().item()),
                "train/step_avg_ms": float(approx_training_time_ms / max(step - args.start_step, 1)),
                "graph_equiv/bpb": float(graph_equiv_bpb) if math.isfinite(graph_equiv_bpb) else float("nan"),
                "graph_equiv/primary_bpb": float(graph_equiv_primary_bpb)
                if math.isfinite(graph_equiv_primary_bpb)
                else float("nan"),
                "graph_equiv/sources": float(len(graph_equiv_bpb_terms)),
                "graph_equiv/graph_in_graph_out": 1.0,
                "graph_equiv/sequence_flattening_for_bpb": float(base_model.graph_output_flattening is not None),
                "fineweb_graphify/enabled": float(args.fineweb_graphify),
                "fineweb_graphify/tokengt_first_class": float(args.tokengt_first_class),
                "fineweb_graphify/graph_radius": float(args.tokengt_graph_radius),
                "fineweb_graphify/token_class_buckets": float(args.tokengt_token_class_buckets),
                "fineweb_graphify/position_buckets": float(args.tokengt_position_buckets),
                "fineweb_graphify/structural_weight": float(args.tokengt_structural_weight),
                "fineweb_graphify/edge_weight": float(args.tokengt_edge_weight),
                "fineweb_graphify/torus_weight": float(args.tokengt_torus_weight),
                "fineweb_graphify/identifier_dim": float(args.tokengt_identifier_dim),
                "fineweb_graphify/identifier_weight": float(args.tokengt_identifier_weight),
                "fineweb_graphify/endpoint_weight": float(args.tokengt_endpoint_weight),
                "fineweb_graphify/edge_token_weight": float(args.tokengt_edge_token_weight),
                "fineweb_graphify/first_class_lr": float(args.tokengt_first_class_lr),
                "fineweb_graphify/convextok_dag_features": float(args.convextok_dag_features and tokenizer_kind == "convextok"),
                "fineweb_graphify/convextok_dag_feature_weight": float(args.convextok_dag_feature_weight),
                "tokenizer/kind_convextok": float(tokenizer_kind == "convextok"),
                "tokenizer/kind_sentencepiece": float(tokenizer_kind == "sentencepiece"),
                "tokenizer/priced_offset": float(PRICED_OFFSET),
                "tokenizer_toric/lp_face_mass": float(convextok_lp_scores.detach().float().sum().item()),
                "tokenizer_toric/priced_vocab_fraction": float(convextok_priced_flags.detach().float().mean().item()),
                "tokenizer_toric/mean_token_byte_length": float(convextok_byte_lengths.detach().float().mean().item()),
                "tokenizer_tropical/mean_token_lp_score": float(convextok_lp_scores.detach().float().mean().item()),
                "convextok_toric/regularizer_weight_config": float(args.convextok_toric_reg_weight),
                "convextok_toric/regularizer_topk_config": float(args.convextok_toric_reg_topk),
                "graph_output_flattening/enabled": float(base_model.graph_output_flattening is not None),
                "graph_output_flattening/edge_radius": float(args.graph_output_edge_radius),
                "graph_output_flattening/node_weight": float(args.graph_output_node_weight),
                "graph_output_flattening/edge_weight": float(args.graph_output_edge_weight),
                "graph_output_flattening/virtual_edge_tokens": float(args.graph_output_virtual_edge_tokens),
                "graph_output_flattening/edge_token_weight": float(args.graph_output_edge_token_weight),
                "graph_output_flattening/score_correction": float(args.graph_output_score_correction),
                "graph_output_flattening/score_correction_weight": float(args.graph_output_score_correction_weight),
                "graph_output_flattening/lr": float(args.graph_output_flattening_lr),
                "graph_output_flattening/oai_fineweb_bpb_only": float(base_model.graph_output_flattening is not None),
                "graph_output_flattening/calibration_enabled": float(
                    args.graph_output_calibration_loss_weight > 0.0
                    and base_model.graph_output_flattening is not None
                ),
                "graph_output_flattening/calibration_every": float(args.graph_output_calibration_every),
                "graph_lm_primary/enabled": float(graph_lm_loader is not None),
                "graph_lm_primary/required": float(args.require_graph_lm_primary),
                "graph_lm_primary/every": float(args.graph_lm_every),
                "graph_lm_primary/loss_weight_peak_config": float(args.graph_lm_loss_weight),
                "graph_lm_primary/loss_weight_start_config": float(args.graph_lm_loss_weight_start),
                "graph_lm_primary/late_stage_configured": float(bool(args.late_graph_train_glob)),
                "graph_lm_primary/late_start_step_config": float(args.late_graph_start_step),
                "graph_lm_primary/late_mix_ratio_config": float(args.late_graph_mix_ratio),
                "graph_lm_primary/late_upsample_passes_config": float(args.late_graph_upsample_passes),
                "long_entry_full_row/enabled_config": float(long_entry_loader is not None),
                "long_entry_full_row/every_config": float(args.long_entry_every),
                "long_entry_full_row/batch_size_config": float(args.long_entry_batch_size),
                "long_entry_full_row/max_tokens_config": float(args.long_entry_max_tokens),
                "long_entry_full_row/segment_len_config": float(args.long_entry_segment_len),
                "long_entry_full_row/segment_stride_config": float(args.long_entry_segment_stride),
                "long_entry_full_row/max_segments_config": float(args.long_entry_max_segments),
                "long_entry_full_row/loss_weight_config": float(args.long_entry_loss_weight),
                "long_entry_full_row/next_segment_weight_config": float(args.long_entry_next_segment_weight),
                "long_entry_full_row/boundary_weight_config": float(args.long_entry_boundary_weight),
                "long_entry_full_row/coverage_weight_config": float(args.long_entry_segment_coverage_weight),
                "long_entry_full_row/fot_loss_weight_config": float(args.long_entry_fot_loss_weight),
                "long_entry_full_row/fot_every_config": float(args.long_entry_fot_every),
                "long_entry_full_row/fot_max_segments_config": float(args.long_entry_fot_max_segments),
                "long_entry_full_row/full_context_every_config": float(args.long_entry_full_context_every),
                "long_entry_full_row/full_context_seq_len_config": float(args.long_entry_full_context_seq_len),
                "long_entry_full_row/full_context_min_tokens_config": float(args.long_entry_full_context_min_tokens),
                "long_entry_full_row/full_context_loss_weight_config": float(args.long_entry_full_context_loss_weight),
                "long_entry_full_row/segment_checkpoint_config": float(args.long_entry_segment_checkpoint),
                "long_entry_full_row/full_context_checkpoint_config": float(args.long_entry_full_context_checkpoint),
                "toricblm_structure/flow_enabled_config": float(args.toricblm_structure_flow),
                "toricblm_structure/head_enabled_config": float(structure_head is not None),
                "toricblm_structure/train_glob_configured": float(bool(args.toricblm_structure_train_glob)),
                "toricblm_structure/every_config": float(args.toricblm_structure_every),
                "toricblm_structure/max_atoms_config": float(args.toricblm_structure_max_atoms),
                "toricblm_structure/batch_size_config": float(args.toricblm_structure_batch_size),
                "toricblm_structure/seq_len_config": float(args.toricblm_structure_seq_len),
                "toricblm_structure/lr_config": float(args.toricblm_structure_lr),
                "toricblm_structure/flow_weight_config": float(args.toricblm_structure_flow_weight),
                "toricblm_structure/contact_weight_config": float(args.toricblm_structure_contact_weight),
                "toricblm_structure/distogram_weight_config": float(args.toricblm_structure_distogram_weight),
                "toricblm_structure/frame_weight_config": float(args.toricblm_structure_frame_weight),
                "toricblm_structure/start_step_config": float(args.toricblm_structure_flow_start_step),
                "toricblm_structure/coordinate_bearing_records": float(
                    structure_coordinate_records
                ),
                "toricblm_structure/structure_association_records": float(
                    structure_association_records
                ),
                "toricblm_structure/readiness_records": float(
                    structure_readiness.get("records") or structure_readiness.get("rows") or 0
                ),
                "toricblm_structure/target_shortfall_count": float(
                    len(structure_readiness.get("target_shortfalls", {}) or {})
                ),
                "toricblm_structure/coordinate_losses_active": float(
                    args.toricblm_structure_flow and structure_loader is not None and structure_head is not None
                ),
                "toricgt_sidecar/enabled": float(sidecar is not None),
                "toricgt_sidecar/loss_weight_peak_config": float(args.sidecar_loss_weight),
                "toricgt_sidecar/loss_weight_start_config": float(args.sidecar_loss_weight_start),
                "toricgt_sidecar/loss_warmup_steps_config": float(args.sidecar_warmup_steps),
                "toricgt_sidecar/graphcg_active": float(sidecar is not None),
                "toricgt_sidecar/analogy_active": float(sidecar is not None),
                "toricgt_sidecar/memory_retrieval_active": float(sidecar is not None),
                "toricgt_sidecar/toric_geometry_weight": float(args.toric_geometry_loss_weight),
                "toricgt_sidecar/toric_vector_bundle_1d_cone_ce_weight": float(args.toric_vector_bundle_loss_weight),
                "toricgt_sidecar/toric_bgg_weight": float(args.toric_bgg_loss_weight),
                "toricgt_sidecar/koszul_persistence_weight": float(args.koszul_persistence_loss_weight),
                "toricgt_sidecar/combinatorial_toric_weight": float(args.combinatorial_toric_loss_weight),
                "toricgt_sidecar/derived_signature_weight": float(args.derived_signature_loss_weight),
                "toricgt_sidecar/advanced_lagrangian_controller_config": float(args.advanced_lagrangian_controller),
                "toricgt_sidecar/lagrangian_dual_lr_config": float(args.lagrangian_dual_lr),
                "toricgt_sidecar/lagrangian_decay_config": float(args.lagrangian_decay),
                "toricgt_sidecar/lagrangian_min_multiplier_config": float(args.lagrangian_min_multiplier),
                "toricgt_sidecar/lagrangian_max_multiplier_config": float(args.lagrangian_max_multiplier),
                "toricgt_sidecar/lagrangian_bpb_ceiling_config": float(args.lagrangian_bpb_ceiling),
                "toricgt_sidecar/toric_fan_curriculum_config": float(args.toric_fan_curriculum),
                "toricgt_sidecar/toric_fan_coarse_steps_config": float(args.toric_fan_coarse_steps),
                "toricgt_sidecar/toric_fan_intermediate_steps_config": float(args.toric_fan_intermediate_steps),
                "toricgt_sidecar/memory_sheaf_gate_min_config": float(args.memory_sheaf_gate_min),
                "toricgt_sidecar/memory_sheaf_gate_threshold_config": float(args.memory_sheaf_gate_threshold),
                "toricgt_sidecar/memory_sheaf_gate_softness_config": float(args.memory_sheaf_gate_softness),
                "toricgt_sidecar/derived_signature_max_vertices_config": float(args.derived_signature_max_vertices),
                "toricgt_sidecar/derived_signature_max_edges_config": float(args.derived_signature_max_edges),
                "teacher_distill/configured": float(args.teacher_distill_weight > 0.0 and bool(args.teacher_checkpoint)),
                "teacher_distill/target_weight": float(args.teacher_distill_weight),
                "fineweb_caseops/enabled": float(args.fineweb_caseops),
                "fineweb_caseops/swapped_token_ids": float(caseops_pairs),
                "aux_grad_routing/enabled": float(args.aux_grad_routing),
                "aux_grad_routing/long_entry_config": float(args.aux_grad_route_long_entry),
                "score_first_tta/enabled": float(args.score_first_tta),
                "score_first_tta/steps": float(args.score_first_tta_steps),
                "score_first_tta/lr": float(args.score_first_tta_lr),
                "score_first_tta/commit": float(args.score_first_tta_commit),
                "bpb_aux_control/staging_enabled_config": float(args.bpb_first_aux_staging),
                "bpb_aux_control/core_steps_config": float(args.bpb_first_core_steps),
                "bpb_aux_control/ramp_steps_config": float(args.bpb_first_ramp_steps),
                "bpb_aux_control/min_aux_mult_config": float(args.bpb_first_min_aux_mult),
                "aux_grad_routing/conflict_controller_config": float(args.aux_conflict_controller),
                "aux_grad_routing/conflict_damp_min_config": float(args.aux_conflict_damp_min),
                "aux_grad_routing/conflict_boost_max_config": float(args.aux_conflict_boost_max),
                "oai_gflownet/enabled": float(oai_gflownet_head is not None),
                "oai_gflownet/loss_weight_config": float(args.oai_gflownet_loss_weight),
                "oai_gflownet/entropy_weight_config": float(args.oai_gflownet_entropy_weight),
                "oai_gflownet/every": float(args.oai_gflownet_every),
                "oai_gflownet/lr": float(args.oai_gflownet_lr),
                "oai_fot/enabled": float(oai_fot_head is not None),
                "oai_fot/loss_weight_config": float(args.oai_fot_loss_weight),
                "oai_fot/every": float(args.oai_fot_every),
                "oai_fot/lr": float(args.oai_fot_lr),
                "oai_fot/num_trees_config": float(args.oai_fot_num_trees),
                "oai_fot/max_depth_config": float(args.oai_fot_max_depth),
                "oai_fot/branching_config": float(args.oai_fot_branching),
                "oai_fot/topk_trees_config": float(args.oai_fot_topk_trees),
                "oai_fot/max_sequences_config": float(args.oai_fot_max_sequences),
                "oai_fot/max_positions_config": float(args.oai_fot_max_positions),
                "oai_fot/sparse_weight_config": float(args.oai_fot_sparse_weight),
                "oai_fot/ucb_weight_config": float(args.oai_fot_ucb_weight),
                "oai_fot/correction_weight_config": float(args.oai_fot_correction_weight),
                "oai_fot/consensus_weight_config": float(args.oai_fot_consensus_weight),
                "oai_fot/tb_weight_config": float(args.oai_fot_tb_weight),
                "oai_fot/subtb_weight_config": float(args.oai_fot_subtb_weight),
                "oai_fot/complexity_weight_config": float(args.oai_fot_complexity_weight),
                "oai_fot/reward_advanced_bonus_config": float(args.oai_fot_reward_advanced_bonus),
                "oai_fot/reward_mode_bpb_delta": float(str(args.oai_fot_reward_mode).strip().lower() in {"bpb_delta", "ce_delta", "byte_delta"}),
                "oai_fot/bpb_delta_weight_config": float(args.oai_fot_bpb_delta_weight),
                "oai_fot/reward_graph_weight_config": float(args.oai_fot_reward_graph_weight),
                "oai_fot/reward_consensus_weight_config": float(args.oai_fot_reward_consensus_weight),
                "oai_fot/reward_complexity_weight_config": float(args.oai_fot_reward_complexity_weight),
                "oai_fot/adaptive_control_config": float(args.oai_fot_adaptive_control),
                "oai_fot/reward_target_config": float(args.oai_fot_reward_target),
                "oai_fot/diversity_target_config": float(args.oai_fot_diversity_target),
                "oai_fot/entropy_high_config": float(args.oai_fot_entropy_high),
                "oai_fot/entropy_low_config": float(args.oai_fot_entropy_low),
                "oai_mtp/enabled": float(args.oai_mtp),
                "oai_mtp/loss_weight_config": float(args.oai_mtp_loss_weight),
                "oai_mtp/every": float(args.oai_mtp_every),
                "toricblm_mup/enabled": float(args.toricblm_mup),
                "toricblm_mup/base_shapes_applied": float(mup_applied),
                "toricblm_mup/width_mult": float(args.mup_width_mult),
                "toricblm_mup/output_mult": float(args.mup_output_mult),
                "toricblm_mup/attention_scale": float(args.mup_attention_scale),
                "toricblm_mup/effective_token_lr": float(token_lr),
                "toricblm_mup/effective_matrix_lr": float(matrix_lr),
                "toricblm_mup/effective_scalar_lr": float(scalar_lr),
                "toricblm_mup/effective_aux_lr": float(sidecar_lr),
            }
            payload.update(bpb_aux_metrics)
            payload.update(convextok_metrics)
            payload.update(flattening_calibration_metrics)
            payload.update(graph_lm_metrics)
            payload.update(long_entry_metrics)
            payload.update(teacher_metrics)
            payload.update(oai_gflownet_metrics)
            payload.update(oai_fot_metrics)
            payload.update(oai_mtp_metrics)
            payload.update(structure_metrics)
            payload.update(sidecar_metrics)
            payload.update(aux_grad_metrics)
            wandb_log0(payload, step)

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
