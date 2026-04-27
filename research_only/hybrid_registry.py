from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchOnlyPreset:
    name: str
    description: str
    mixer_kind: str
    env: dict[str, str]
    notes: tuple[str, ...] = ()


RESEARCH_ONLY_PRESETS: dict[str, ResearchOnlyPreset] = {
    "hybrid_hymba_proxy": ResearchOnlyPreset(
        name="hybrid_hymba_proxy",
        description="Proxy research preset for a Hymba-style hybrid attention/SSM mixer.",
        mixer_kind="hybrid_hymba",
        env={
            "VOCAB_SIZE": "8192",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "STATE_DIM": "64",
        },
        notes=(
            "Research-only preset: never treat this as submission-safe.",
        ),
    ),
    "retnet_chunkwise_proxy": ResearchOnlyPreset(
        name="retnet_chunkwise_proxy",
        description="Proxy research preset for chunkwise recurrent retention.",
        mixer_kind="retnet_chunkwise",
        env={
            "VOCAB_SIZE": "8192",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "STATE_DIM": "64",
        },
        notes=(
            "Research-only preset: use only for proxy experiments.",
        ),
    ),
    "delta_hybrid_proxy": ResearchOnlyPreset(
        name="delta_hybrid_proxy",
        description="Proxy research preset for a hybrid delta-rule sequence mixer.",
        mixer_kind="delta_hybrid",
        env={
            "VOCAB_SIZE": "8192",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "STATE_DIM": "64",
        },
        notes=(
            "Research-only preset: use only for proxy experiments.",
        ),
    ),
}

