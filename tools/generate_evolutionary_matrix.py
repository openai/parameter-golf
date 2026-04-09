#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvoMatrixRun:
    name: str
    stage: str
    intent: str
    command: str
    flags: tuple[str, ...]
    tags: tuple[str, ...] = ()

    def output_path(self, output_dir: str) -> str:
        return f"{output_dir}/{self.name}.json"

    def command_parts(self, python_bin: str, script_path: str, output_dir: str, enwik8_path: str) -> tuple[str, ...]:
        result_path = self.output_path(output_dir)
        parts = [python_bin, script_path, self.command, *self.flags]
        if self.command != "vmap-throughput":
            parts.extend(("--enwik8-path", enwik8_path))
        parts.extend(("--output-json", result_path))
        return tuple(parts)

    def shell_command(self, python_bin: str, script_path: str, output_dir: str, enwik8_path: str) -> str:
        return " ".join(shlex.quote(part) for part in self.command_parts(python_bin, script_path, output_dir, enwik8_path))


def parse_csv(spec: str | None) -> tuple[str, ...]:
    if spec is None:
        return ()
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def shared_model_flags(
    *,
    device: str = "cuda",
    dtype: str = "fp16",
    model_dim: int = 512,
    num_layers: int = 9,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: int = 3,
    vocab_size: int = 256,
    seq_len: int = 256,
    stride: int = 64,
    batch_size: int = 8,
    qk_gain_init: float = 1.5,
    spine_variant: str = "plain",
    xsa_last_n: int = 4,
) -> tuple[str, ...]:
    flags = [
        "--device",
        device,
        "--dtype",
        dtype,
        "--model-dim",
        str(model_dim),
        "--num-layers",
        str(num_layers),
        "--num-heads",
        str(num_heads),
        "--num-kv-heads",
        str(num_kv_heads),
        "--mlp-mult",
        str(mlp_mult),
        "--vocab-size",
        str(vocab_size),
        "--seq-len",
        str(seq_len),
        "--stride",
        str(stride),
        "--batch-size",
        str(batch_size),
        "--qk-gain-init",
        str(qk_gain_init),
        "--spine-variant",
        spine_variant,
    ]
    if spine_variant == "xsa":
        flags.extend(("--xsa-last-n", str(xsa_last_n)))
    return tuple(flags)


def make_run(
    *,
    name: str,
    stage: str,
    intent: str,
    command: str,
    flags: tuple[str, ...],
    tags: tuple[str, ...],
) -> EvoMatrixRun:
    return EvoMatrixRun(name=name, stage=stage, intent=intent, command=command, flags=flags, tags=tags)


def make_evolution_run(
    *,
    name: str,
    stage: str,
    intent: str,
    strategy: str,
    percentile: float = 50.0,
    base_train_seconds: int = 120,
    generations: int = 8,
    population_size: int = 16,
    tournament_size: int = 4,
    mutation_std: float = 5e-4,
    mutation_fraction: float = 0.05,
    seed: int = 1337,
    eval_batches: int = 32,
    tags: tuple[str, ...] = (),
    shared_flags: tuple[str, ...] | None = None,
) -> EvoMatrixRun:
    flags = (shared_flags or shared_model_flags()) + (
        "--seed",
        str(seed),
        "--base-train-seconds",
        str(base_train_seconds),
        "--generations",
        str(generations),
        "--population-size",
        str(population_size),
        "--tournament-size",
        str(tournament_size),
        "--crossover-strategy",
        strategy,
        "--crossover-percentile",
        str(percentile),
        "--mutation-std",
        str(mutation_std),
        "--mutation-fraction",
        str(mutation_fraction),
        "--eval-batches",
        str(eval_batches),
    )
    return make_run(
        name=name,
        stage=stage,
        intent=intent,
        command="evolution-loop",
        flags=flags,
        tags=tags + ("evolution", f"strategy:{strategy}"),
    )


def make_committee_viability_run(
    *,
    name: str,
    stage: str,
    intent: str,
    copies: int,
    train_seconds: float,
    ensemble_topks: str,
    seed: int,
    tags: tuple[str, ...] = (),
    shared_flags: tuple[str, ...] | None = None,
) -> EvoMatrixRun:
    flags = (shared_flags or shared_model_flags(dtype="bf16")) + (
        "--seed",
        str(seed),
        "--copies",
        str(copies),
        "--train-seconds",
        f"{train_seconds:g}",
        "--crossover-strategies",
        "layer_swap",
        "--percentiles",
        "50",
        "--eval-batches",
        "8",
        "--pair-limit",
        "0",
        "--ensemble-topks",
        ensemble_topks,
        "--member-train-mode",
        "parallel_vmap",
    )
    return make_run(
        name=name,
        stage=stage,
        intent=intent,
        command="crossover-viability",
        flags=flags,
        tags=tags + ("committee", "frontier", "seed"),
    )


def make_committee_schedule_run(
    *,
    name: str,
    stage: str,
    intent: str,
    stage_copies: str,
    stage_train_seconds: str,
    ensemble_topks: str,
    seed: int,
    spawn_noise_std: float = 0.0,
    tags: tuple[str, ...] = (),
    shared_flags: tuple[str, ...] | None = None,
) -> EvoMatrixRun:
    flags = (shared_flags or shared_model_flags(dtype="bf16")) + (
        "--seed",
        str(seed),
        "--stage-copies",
        stage_copies,
        "--stage-train-seconds",
        stage_train_seconds,
        "--eval-batches",
        "8",
        "--ensemble-topks",
        ensemble_topks,
        "--spawn-noise-std",
        f"{spawn_noise_std:g}",
    )
    return make_run(
        name=name,
        stage=stage,
        intent=intent,
        command="committee-schedule",
        flags=flags,
        tags=tags + ("committee", "frontier", "staged", "seed"),
    )


def build_matrix() -> list[EvoMatrixRun]:
    standard = shared_model_flags()
    train_stable = shared_model_flags(dtype="bf16")
    xsa = shared_model_flags(spine_variant="xsa", xsa_last_n=4)
    xsa_train_stable = shared_model_flags(dtype="bf16", spine_variant="xsa", xsa_last_n=4)
    parallel_member_flags = ("--member-train-mode", "parallel_vmap")
    viability_standard = standard + parallel_member_flags
    viability_train_stable = train_stable + parallel_member_flags
    viability_xsa_train_stable = xsa_train_stable + parallel_member_flags
    runs: list[EvoMatrixRun] = [
        make_run(
            name="throughput_fp16_standard",
            stage="0-throughput",
            intent="Measure full-population vmap scaling at the target 16MB-ish transformer shape in fp16.",
            command="vmap-throughput",
            flags=standard
            + (
                "--population-scales",
                "8,64,256,1024,4096,16384",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "core", "fp16"),
        ),
        make_run(
            name="throughput_bf16_standard",
            stage="0-throughput",
            intent="Repeat the full population ladder in bf16 to see whether stability changes the scaling envelope.",
            command="vmap-throughput",
            flags=shared_model_flags(dtype="bf16")
            + (
                "--population-scales",
                "8,64,256,1024,4096,16384",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "core", "bf16"),
        ),
        make_run(
            name="throughput_bf16_standard_chunk256",
            stage="0-throughput",
            intent="Repeat the bf16 ladder with chunked population evaluation so the larger scales can run past the first memory wall.",
            command="vmap-throughput",
            flags=shared_model_flags(dtype="bf16")
            + (
                "--population-scales",
                "8,64,256,1024,4096,16384",
                "--population-chunk-size",
                "256",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "core", "bf16", "chunked"),
        ),
        make_run(
            name="throughput_fp16_longseq",
            stage="0-throughput",
            intent="Stress the same population ladder at sequence length 512 to surface attention-dominated scaling.",
            command="vmap-throughput",
            flags=shared_model_flags(seq_len=512, batch_size=4)
            + (
                "--population-scales",
                "8,64,256,1024,4096",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "deep", "attention", "fp16"),
        ),
        make_run(
            name="throughput_fp16_bigbatch",
            stage="0-throughput",
            intent="Push batch-heavy evaluation at sequence length 256 to see whether wider token batches improve population efficiency.",
            command="vmap-throughput",
            flags=shared_model_flags(batch_size=16)
            + (
                "--population-scales",
                "8,64,256,1024,4096",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "deep", "batch", "fp16"),
        ),
        make_run(
            name="throughput_xsa_fp16_standard",
            stage="0-throughput",
            intent="Check whether the XSA spine variant changes batched population throughput enough to matter for evolutionary search.",
            command="vmap-throughput",
            flags=xsa
            + (
                "--population-scales",
                "8,64,256,1024,4096",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "deep", "xsa", "fp16"),
        ),
        make_run(
            name="throughput_fp16_seq1024",
            stage="0-throughput",
            intent="Add a long-context stress point at sequence length 1024 to expose where population scaling stops being practical.",
            command="vmap-throughput",
            flags=shared_model_flags(seq_len=1024, batch_size=2)
            + (
                "--population-scales",
                "8,64,256,1024",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "4",
            ),
            tags=("throughput", "deep", "stress", "fp16"),
        ),
        make_run(
            name="throughput_bf16_standard_chunk128",
            stage="0-throughput",
            intent="Probe a smaller chunk size to measure whether chunk overhead or memory headroom dominates at larger populations.",
            command="vmap-throughput",
            flags=shared_model_flags(dtype="bf16")
            + (
                "--population-scales",
                "256,1024,4096,16384",
                "--population-chunk-size",
                "128",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "deep", "bf16", "chunked", "chunk128"),
        ),
        make_run(
            name="throughput_bf16_standard_chunk512",
            stage="0-throughput",
            intent="Probe a larger chunk size to see how much throughput we can buy before memory pressure returns.",
            command="vmap-throughput",
            flags=shared_model_flags(dtype="bf16")
            + (
                "--population-scales",
                "256,1024,4096,16384",
                "--population-chunk-size",
                "512",
                "--noise-std",
                "0.001",
                "--warmup-repeats",
                "2",
                "--timed-repeats",
                "5",
            ),
            tags=("throughput", "deep", "bf16", "chunked", "chunk512"),
        ),
        make_run(
            name="viability_short_all_seed1337",
            stage="1-viability",
            intent="Cheap all-strategy viability sweep to identify obvious collapse modes before spending minutes per parent.",
            command="crossover-viability",
            flags=viability_train_stable
            + (
                "--seed",
                "1337",
                "--copies",
                "8",
                "--train-seconds",
                "30",
                "--crossover-strategies",
                "weight_overlap,delta_overlap,sign_consensus,delta_sign_consensus,tensor_swap,layer_swap,delta_importance",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "16",
            ),
            tags=("viability", "core", "seed", "all-strategies"),
        ),
        make_run(
            name="viability_short_all_seed2025",
            stage="1-viability",
            intent="Repeat the short all-strategy viability sweep with a different training order seed for robustness.",
            command="crossover-viability",
            flags=viability_train_stable
            + (
                "--seed",
                "2025",
                "--copies",
                "8",
                "--train-seconds",
                "30",
                "--crossover-strategies",
                "weight_overlap,delta_overlap,sign_consensus,delta_sign_consensus,tensor_swap,layer_swap,delta_importance",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "16",
            ),
            tags=("viability", "core", "seed", "all-strategies"),
        ),
        make_run(
            name="viability_short_all_seed4242",
            stage="1-viability",
            intent="A third short all-strategy seed sweep so collapse or viability is not inferred from one lucky ordering.",
            command="crossover-viability",
            flags=viability_train_stable
            + (
                "--seed",
                "4242",
                "--copies",
                "8",
                "--train-seconds",
                "30",
                "--crossover-strategies",
                "weight_overlap,delta_overlap,sign_consensus,delta_sign_consensus,tensor_swap,layer_swap,delta_importance",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "16",
            ),
            tags=("viability", "core", "seed", "all-strategies"),
        ),
        make_run(
            name="viability_short_core_copies12",
            stage="1-viability",
            intent="Increase parent diversity cheaply with 12 copies on the most plausible crossover families.",
            command="crossover-viability",
            flags=viability_train_stable
            + (
                "--seed",
                "1337",
                "--copies",
                "12",
                "--train-seconds",
                "45",
                "--crossover-strategies",
                "delta_overlap,delta_importance,delta_sign_consensus,layer_swap",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "16",
            ),
            tags=("viability", "deep", "core-strategies", "copies12"),
        ),
        make_run(
            name="viability_full_all",
            stage="1-viability",
            intent="Full 2-minute viability sweep across all crossover families using all 28 parent pairs.",
            command="crossover-viability",
            flags=viability_train_stable
            + (
                "--seed",
                "1337",
                "--copies",
                "8",
                "--train-seconds",
                "120",
                "--crossover-strategies",
                "weight_overlap,delta_overlap,sign_consensus,delta_sign_consensus,tensor_swap,layer_swap,delta_importance",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "32",
            ),
            tags=("viability", "core", "all-strategies", "full"),
        ),
        make_run(
            name="viability_full_core_seed2025",
            stage="1-viability",
            intent="Repeat the full viability pass on the most plausible families with a second seed to check stability.",
            command="crossover-viability",
            flags=viability_standard
            + (
                "--seed",
                "2025",
                "--copies",
                "8",
                "--train-seconds",
                "120",
                "--crossover-strategies",
                "delta_overlap,delta_importance,delta_sign_consensus,layer_swap",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "32",
            ),
            tags=("viability", "deep", "seed", "core-strategies", "full"),
        ),
        make_run(
            name="viability_long_core",
            stage="1-viability",
            intent="Longer parent training on the structurally plausible crossover operators to see whether better-trained parents recombine more safely.",
            command="crossover-viability",
            flags=viability_standard
            + (
                "--seed",
                "1337",
                "--copies",
                "8",
                "--train-seconds",
                "300",
                "--crossover-strategies",
                "delta_overlap,delta_importance,delta_sign_consensus,layer_swap",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "32",
            ),
            tags=("viability", "deep", "core-strategies", "long"),
        ),
        make_run(
            name="viability_xsa_core",
            stage="1-viability",
            intent="Run the core viability families on the XSA variant in case recombination behaves differently under a different trunk.",
            command="crossover-viability",
            flags=viability_xsa_train_stable
            + (
                "--seed",
                "1337",
                "--copies",
                "8",
                "--train-seconds",
                "120",
                "--crossover-strategies",
                "delta_overlap,delta_importance,delta_sign_consensus,layer_swap",
                "--percentiles",
                "25,50,75",
                "--eval-batches",
                "32",
            ),
            tags=("viability", "deep", "xsa", "core-strategies"),
        ),
    ]

    for strategy in ("delta_overlap", "delta_importance"):
        for percentile in (25, 50, 75):
            runs.append(
                make_evolution_run(
                    name=f"evo_{strategy}_p{percentile}_pop16",
                    stage="2-evolution-core",
                    intent=f"Core end-to-end evolutionary loop with {strategy} at percentile {percentile} on a moderate population.",
                    strategy=strategy,
                    percentile=float(percentile),
                    tags=("core", "percentile-sweep", "pop16"),
                    shared_flags=train_stable,
                )
            )

    runs.extend(
        [
            make_evolution_run(
                name="evo_parent_copy_pop16",
                stage="2-evolution-core",
                intent="Mutation-only evolutionary baseline: tournament-select one parent, clone it, and rely on mutation to test whether crossover itself is the problem.",
                strategy="parent_copy",
                tags=("core", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_weight_overlap_p50_pop16",
                stage="2-evolution-core",
                intent="Control run using raw weight-overlap crossover at percentile 50 to see how fragile direct weight recombination really is.",
                strategy="weight_overlap",
                percentile=50.0,
                tags=("core", "pop16", "control"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_tensor_swap_pop16",
                stage="2-evolution-core",
                intent="Whole-tensor inheritance control to measure whether coarse module selection is safer than fine-grained mixing.",
                strategy="tensor_swap",
                tags=("core", "pop16", "control"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_layer_swap_pop16",
                stage="2-evolution-core",
                intent="Layer-group crossover to preserve larger circuits while still allowing evolutionary recombination.",
                strategy="layer_swap",
                tags=("core", "pop16", "structural"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_sign_consensus_pop16",
                stage="2-evolution-core",
                intent="Sign-consensus crossover tests whether agreement in direction is a better splice rule than value closeness.",
                strategy="sign_consensus",
                tags=("core", "pop16", "sign"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_sign_consensus_pop16",
                stage="2-evolution-core",
                intent="Delta-space sign consensus keeps the shared base fixed while recombining directional adaptations.",
                strategy="delta_sign_consensus",
                tags=("core", "pop16", "sign"),
                shared_flags=train_stable,
            ),
        ]
    )

    for strategy in ("delta_overlap", "delta_importance"):
        runs.extend(
            [
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_std2e4",
                    stage="3-evolution-sweeps",
                    intent=f"Sweep lower mutation noise for {strategy} at the default operating point.",
                    strategy=strategy,
                    percentile=50.0,
                    mutation_std=2e-4,
                    tags=("sweep", "mutation", "std", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_std1e3",
                    stage="3-evolution-sweeps",
                    intent=f"Sweep higher mutation noise for {strategy} at the default operating point.",
                    strategy=strategy,
                    percentile=50.0,
                    mutation_std=1e-3,
                    tags=("sweep", "mutation", "std", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_frac002",
                    stage="3-evolution-sweeps",
                    intent=f"Use sparser mutations for {strategy} to see whether evolutionary progress prefers gentler perturbations.",
                    strategy=strategy,
                    percentile=50.0,
                    mutation_fraction=0.02,
                    tags=("sweep", "mutation", "fraction", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_frac010",
                    stage="3-evolution-sweeps",
                    intent=f"Use denser mutations for {strategy} to test whether broader exploration helps or destabilizes evolution.",
                    strategy=strategy,
                    percentile=50.0,
                    mutation_fraction=0.10,
                    tags=("sweep", "mutation", "fraction", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_tourn2",
                    stage="3-evolution-sweeps",
                    intent=f"Lower tournament pressure for {strategy} to test whether weaker selection preserves useful diversity.",
                    strategy=strategy,
                    percentile=50.0,
                    tournament_size=2,
                    tags=("sweep", "selection", "tournament", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_tourn8",
                    stage="3-evolution-sweeps",
                    intent=f"Higher tournament pressure for {strategy} to test whether aggressive selection accelerates or collapses progress.",
                    strategy=strategy,
                    percentile=50.0,
                    tournament_size=8,
                    tags=("sweep", "selection", "tournament", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_base60",
                    stage="3-evolution-sweeps",
                    intent=f"Shorter gradient warm start for {strategy} to see whether evolution wants rougher ancestors.",
                    strategy=strategy,
                    percentile=50.0,
                    base_train_seconds=60,
                    tags=("sweep", "pretrain", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop16_base300",
                    stage="3-evolution-sweeps",
                    intent=f"Longer gradient warm start for {strategy} to see whether recombination needs better-trained parents.",
                    strategy=strategy,
                    percentile=50.0,
                    base_train_seconds=300,
                    tags=("sweep", "pretrain", "pop16"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop32",
                    stage="3-evolution-sweeps",
                    intent=f"Scale {strategy} to population 32 once the pop16 dynamics are known.",
                    strategy=strategy,
                    percentile=50.0,
                    population_size=32,
                    tags=("sweep", "population", "pop32"),
                    shared_flags=train_stable,
                ),
                make_evolution_run(
                    name=f"evo_{strategy}_p50_pop32_gen16",
                    stage="3-evolution-sweeps",
                    intent=f"Increase both population and generation depth for {strategy} to see whether more search time compounds.",
                    strategy=strategy,
                    percentile=50.0,
                    population_size=32,
                    generations=16,
                    tags=("sweep", "population", "depth", "pop32"),
                    shared_flags=train_stable,
                ),
            ]
        )

    runs.extend(
        [
            make_evolution_run(
                name="evo_parent_copy_pop16_std2e4",
                stage="3-evolution-sweeps",
                intent="Lower mutation noise on the mutation-only baseline to test a more local hill-climbing regime.",
                strategy="parent_copy",
                mutation_std=2e-4,
                tags=("sweep", "mutation", "std", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_std1e3",
                stage="3-evolution-sweeps",
                intent="Higher mutation noise on the mutation-only baseline to test whether broader exploration helps without crossover.",
                strategy="parent_copy",
                mutation_std=1e-3,
                tags=("sweep", "mutation", "std", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_frac002",
                stage="3-evolution-sweeps",
                intent="Use sparser mutations on the mutation-only baseline to preserve more of the selected parent each generation.",
                strategy="parent_copy",
                mutation_fraction=0.02,
                tags=("sweep", "mutation", "fraction", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_frac010",
                stage="3-evolution-sweeps",
                intent="Use denser mutations on the mutation-only baseline to widen search when crossover is absent.",
                strategy="parent_copy",
                mutation_fraction=0.10,
                tags=("sweep", "mutation", "fraction", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_tourn2",
                stage="3-evolution-sweeps",
                intent="Lower tournament pressure on the mutation-only baseline to preserve diversity longer.",
                strategy="parent_copy",
                tournament_size=2,
                tags=("sweep", "selection", "tournament", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_tourn8",
                stage="3-evolution-sweeps",
                intent="Higher tournament pressure on the mutation-only baseline to test greedier hill-climbing.",
                strategy="parent_copy",
                tournament_size=8,
                tags=("sweep", "selection", "tournament", "pop16", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop32",
                stage="3-evolution-sweeps",
                intent="Scale the mutation-only baseline to population 32 so we can compare pure search breadth against crossover families.",
                strategy="parent_copy",
                population_size=32,
                tags=("sweep", "population", "pop32", "mutation-only"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop32_gen16",
                stage="3-evolution-sweeps",
                intent="Increase both population and generation depth for the mutation-only baseline to see whether it compounds over longer search horizons.",
                strategy="parent_copy",
                population_size=32,
                generations=16,
                tags=("sweep", "population", "depth", "pop32", "mutation-only"),
                shared_flags=train_stable,
            ),
        ]
    )

    runs.extend(
        [
            make_evolution_run(
                name="evo_parent_copy_pop16_seed42",
                stage="4-evolution-seeds",
                intent="Seed repeat for the mutation-only baseline.",
                strategy="parent_copy",
                seed=42,
                tags=("seed", "replicate", "mutation-only", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_seed2025",
                stage="4-evolution-seeds",
                intent="Second seed repeat for the mutation-only baseline.",
                strategy="parent_copy",
                seed=2025,
                tags=("seed", "replicate", "mutation-only", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_parent_copy_pop16_seed4242",
                stage="4-evolution-seeds",
                intent="Third seed repeat for the mutation-only baseline.",
                strategy="parent_copy",
                seed=4242,
                tags=("seed", "replicate", "mutation-only", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_overlap_p50_pop16_seed42",
                stage="4-evolution-seeds",
                intent="Seed repeat for the default delta-overlap operating point.",
                strategy="delta_overlap",
                percentile=50.0,
                seed=42,
                tags=("seed", "replicate", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_overlap_p50_pop16_seed2025",
                stage="4-evolution-seeds",
                intent="Second seed repeat for the default delta-overlap operating point.",
                strategy="delta_overlap",
                percentile=50.0,
                seed=2025,
                tags=("seed", "replicate", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_overlap_p50_pop16_seed4242",
                stage="4-evolution-seeds",
                intent="Third seed repeat for the default delta-overlap operating point.",
                strategy="delta_overlap",
                percentile=50.0,
                seed=4242,
                tags=("seed", "replicate", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_importance_p50_pop16_seed42",
                stage="4-evolution-seeds",
                intent="Seed repeat for the default delta-importance operating point.",
                strategy="delta_importance",
                percentile=50.0,
                seed=42,
                tags=("seed", "replicate", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_importance_p50_pop16_seed2025",
                stage="4-evolution-seeds",
                intent="Second seed repeat for the default delta-importance operating point.",
                strategy="delta_importance",
                percentile=50.0,
                seed=2025,
                tags=("seed", "replicate", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_delta_importance_p50_pop16_seed4242",
                stage="4-evolution-seeds",
                intent="Third seed repeat for the default delta-importance operating point.",
                strategy="delta_importance",
                percentile=50.0,
                seed=4242,
                tags=("seed", "replicate", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_layer_swap_pop16_seed2025",
                stage="4-evolution-seeds",
                intent="Seed repeat for layer-swap to see whether structural crossover behavior is consistent.",
                strategy="layer_swap",
                seed=2025,
                tags=("seed", "replicate", "structural", "pop16"),
                shared_flags=train_stable,
            ),
            make_evolution_run(
                name="evo_layer_swap_pop16_seed4242",
                stage="4-evolution-seeds",
                intent="Second seed repeat for layer-swap to quantify variance on the structural baseline.",
                strategy="layer_swap",
                seed=4242,
                tags=("seed", "replicate", "structural", "pop16"),
                shared_flags=train_stable,
            ),
        ]
    )

    runs.extend(
        [
            make_committee_viability_run(
                name="committee_c4_t120_seed1337",
                stage="5-committee-frontier",
                intent="Best absolute budget-480 depth-first committee baseline on seed 1337.",
                copies=4,
                train_seconds=120.0,
                ensemble_topks="2,4",
                seed=1337,
                tags=("pure", "depth", "budget480", "c4"),
                shared_flags=train_stable,
            ),
            make_committee_viability_run(
                name="committee_c4_t120_seed2025",
                stage="5-committee-frontier",
                intent="Best absolute budget-480 depth-first committee baseline on seed 2025.",
                copies=4,
                train_seconds=120.0,
                ensemble_topks="2,4",
                seed=2025,
                tags=("pure", "depth", "budget480", "c4"),
                shared_flags=train_stable,
            ),
            make_committee_viability_run(
                name="committee_c6_t80_seed1337",
                stage="5-committee-frontier",
                intent="Intermediate depth-width committee point on seed 1337 for the budget-480 frontier.",
                copies=6,
                train_seconds=80.0,
                ensemble_topks="2,4,6",
                seed=1337,
                tags=("pure", "depth", "budget480", "c6"),
                shared_flags=train_stable,
            ),
            make_committee_viability_run(
                name="committee_c6_t80_seed2025",
                stage="5-committee-frontier",
                intent="Intermediate depth-width committee point on seed 2025 for the budget-480 frontier.",
                copies=6,
                train_seconds=80.0,
                ensemble_topks="2,4,6",
                seed=2025,
                tags=("pure", "depth", "budget480", "c6"),
                shared_flags=train_stable,
            ),
            make_committee_viability_run(
                name="committee_c8_t60_seed1337",
                stage="5-committee-frontier",
                intent="Wider pure committee reference point on seed 1337 for the budget-480 frontier.",
                copies=8,
                train_seconds=60.0,
                ensemble_topks="2,4,8",
                seed=1337,
                tags=("pure", "width", "budget480", "c8"),
                shared_flags=train_stable,
            ),
            make_committee_viability_run(
                name="committee_c8_t60_seed2025",
                stage="5-committee-frontier",
                intent="Wider pure committee reference point on seed 2025 for the budget-480 frontier.",
                copies=8,
                train_seconds=60.0,
                ensemble_topks="2,4,8",
                seed=2025,
                tags=("pure", "width", "budget480", "c8"),
                shared_flags=train_stable,
            ),
            make_committee_schedule_run(
                name="committee_sched_4x90_8x15_seed1337",
                stage="5-committee-frontier",
                intent="Depth-heavy staged schedule on seed 1337: train four branches deep, then widen to eight late.",
                stage_copies="4,8",
                stage_train_seconds="90,15",
                ensemble_topks="2,4,8",
                seed=1337,
                tags=("staged", "budget480", "late-width", "c8"),
                shared_flags=train_stable,
            ),
            make_committee_schedule_run(
                name="committee_sched_4x90_8x15_seed2025",
                stage="5-committee-frontier",
                intent="Depth-heavy staged schedule on seed 2025: train four branches deep, then widen to eight late.",
                stage_copies="4,8",
                stage_train_seconds="90,15",
                ensemble_topks="2,4,8",
                seed=2025,
                tags=("staged", "budget480", "late-width", "c8"),
                shared_flags=train_stable,
            ),
            make_committee_schedule_run(
                name="committee_sched_4x60_8x30_seed1337",
                stage="5-committee-frontier",
                intent="Balanced staged schedule on seed 1337 with an equal split between deepening four branches and widening to eight.",
                stage_copies="4,8",
                stage_train_seconds="60,30",
                ensemble_topks="2,4,8",
                seed=1337,
                tags=("staged", "budget480", "balanced", "c8"),
                shared_flags=train_stable,
            ),
            make_committee_schedule_run(
                name="committee_sched_4x60_8x30_seed2025",
                stage="5-committee-frontier",
                intent="Balanced staged schedule on seed 2025 with an equal split between deepening four branches and widening to eight.",
                stage_copies="4,8",
                stage_train_seconds="60,30",
                ensemble_topks="2,4,8",
                seed=2025,
                tags=("staged", "budget480", "balanced", "c8"),
                shared_flags=train_stable,
            ),
            make_committee_schedule_run(
                name="committee_sched_4x60_8x15_16x7p5_seed1337",
                stage="5-committee-frontier",
                intent="Progressive widening schedule on seed 1337 that grows 4 -> 8 -> 16 while keeping total branch-seconds fixed.",
                stage_copies="4,8,16",
                stage_train_seconds="60,15,7.5",
                ensemble_topks="2,4,8,16",
                seed=1337,
                tags=("staged", "budget480", "progressive-width", "c16"),
                shared_flags=train_stable,
            ),
            make_committee_schedule_run(
                name="committee_sched_4x60_8x15_16x7p5_seed2025",
                stage="5-committee-frontier",
                intent="Progressive widening schedule on seed 2025 that grows 4 -> 8 -> 16 while keeping total branch-seconds fixed.",
                stage_copies="4,8,16",
                stage_train_seconds="60,15,7.5",
                ensemble_topks="2,4,8,16",
                seed=2025,
                tags=("staged", "budget480", "progressive-width", "c16"),
                shared_flags=train_stable,
            ),
        ]
    )

    return runs


def select_runs(
    runs: list[EvoMatrixRun],
    *,
    stages: tuple[str, ...] = (),
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
    names: tuple[str, ...] = (),
) -> list[EvoMatrixRun]:
    selected = runs
    if stages:
        stage_set = set(stages)
        selected = [run for run in selected if run.stage in stage_set]
    if include_tags:
        include = set(include_tags)
        selected = [run for run in selected if include.issubset(set(run.tags))]
    if exclude_tags:
        exclude = set(exclude_tags)
        selected = [run for run in selected if exclude.isdisjoint(set(run.tags))]
    if names:
        name_set = set(names)
        selected = [run for run in selected if run.name in name_set]
    return selected


def render_table(runs: list[EvoMatrixRun]) -> str:
    headers = ("name", "stage", "command", "tags", "intent")
    rows = []
    widths = {header: len(header) for header in headers}
    for run in runs:
        row = {
            "name": run.name,
            "stage": run.stage,
            "command": run.command,
            "tags": ",".join(run.tags),
            "intent": run.intent,
        }
        rows.append(row)
        for header, value in row.items():
            widths[header] = max(widths[header], len(value))
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def render_shell(runs: list[EvoMatrixRun], *, python_bin: str, script_path: str, output_dir: str, enwik8_path: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"mkdir -p {shlex.quote(output_dir)}",
        "",
    ]
    for run in runs:
        lines.append(f"# {run.name} [{run.stage}] {' '.join(run.tags)}")
        lines.append(run.shell_command(python_bin, script_path, output_dir, enwik8_path))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate a staged experiment queue for the evolutionary transformer benchmark")
    parser.add_argument("--format", choices=("table", "json", "shell", "names"), default="table")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--script-path", default="tools/evolutionary_benchmark.py")
    parser.add_argument("--output-dir", default="runs/evolutionary")
    parser.add_argument("--enwik8-path", default="/workspace/data/enwik8")
    parser.add_argument("--stages", default=None, help="Comma-separated stage filters")
    parser.add_argument("--include-tags", default=None, help="Comma-separated tags that must all be present")
    parser.add_argument("--exclude-tags", default=None, help="Comma-separated tags to omit")
    parser.add_argument("--names", default=None, help="Comma-separated explicit run names")
    args = parser.parse_args(argv[1:])

    runs = select_runs(
        build_matrix(),
        stages=parse_csv(args.stages),
        include_tags=parse_csv(args.include_tags),
        exclude_tags=parse_csv(args.exclude_tags),
        names=parse_csv(args.names),
    )
    if args.format == "table":
        print(render_table(runs))
        return 0
    if args.format == "json":
        print(json.dumps([asdict(run) for run in runs], indent=2))
        return 0
    if args.format == "names":
        for run in runs:
            print(run.name)
        return 0
    print(
        render_shell(
            runs,
            python_bin=args.python_bin,
            script_path=args.script_path,
            output_dir=args.output_dir,
            enwik8_path=args.enwik8_path,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
