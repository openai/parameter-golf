from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .codecs import ensure_tokens
from .config import OpenPredictiveCoderConfig
from .datasets import ByteSequenceDataset
from .factories import create_substrate_for_model
from .latents import LatentCommitter, LatentObservation, LatentState
from .metrics import bits_per_byte_from_logits, softmax
from .patching import AdaptiveSegmenter
from .readout import RidgeReadout
from .runtime import FitReport, SequenceReport, SequenceTrace
from .substrates import TokenSubstrate
from .views import ByteLatentFeatureView


from dataclasses import dataclass


@dataclass
class _AdapterState:
    substrate_state: np.ndarray
    latent_state: LatentState


@dataclass(frozen=True)
class _AdapterStep:
    feature: np.ndarray
    observation: LatentObservation


class ByteLatentPredictiveCoder:
    def __init__(
        self,
        config: OpenPredictiveCoderConfig | None = None,
        *,
        substrate: TokenSubstrate | None = None,
        segmenter: AdaptiveSegmenter | None = None,
        committer: LatentCommitter | None = None,
        feature_view: ByteLatentFeatureView | None = None,
        readout: RidgeReadout | None = None,
    ):
        self.config = config or OpenPredictiveCoderConfig()
        self.segmenter = segmenter or AdaptiveSegmenter(self.config.segmenter)
        self.substrate = substrate or create_substrate_for_model(self.config)
        self.committer = committer or LatentCommitter(
            config=self.config.latent,
            substrate_size=self.substrate.state_dim,
            seed=self.config.reservoir.seed + 101,
        )
        self.feature_view = feature_view or ByteLatentFeatureView(
            max_patch_size=self.config.segmenter.max_patch_size,
        )
        self.readout = readout or RidgeReadout(
            input_dim=self.config.feature_dim,
            output_dim=self.config.vocabulary_size,
            alpha=self.config.latent.readout_l2,
        )

    def _initial_state(self) -> _AdapterState:
        return _AdapterState(
            substrate_state=self.substrate.initial_state(),
            latent_state=self.committer.initial_state(),
        )

    def _advance_state(self, state: _AdapterState, token: int) -> _AdapterStep:
        state.substrate_state = self.substrate.step(state.substrate_state, token)
        local_view = self.committer.sample(state.substrate_state)
        observation = self.committer.step(
            state.latent_state,
            local_view,
            self.segmenter,
        )
        return _AdapterStep(
            feature=self.feature_view.encode(observation),
            observation=observation,
        )

    def _coerce_sequences(
        self,
        data: ByteSequenceDataset | str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> tuple[np.ndarray, ...]:
        if isinstance(data, ByteSequenceDataset):
            return data.sequences
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence) and data and all(isinstance(item, int) for item in data):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence):
            return tuple(ensure_tokens(item) for item in data)
        return (ensure_tokens(data),)

    def trace(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> SequenceTrace:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("A sequence must contain at least two tokens.")

        state = self._initial_state()
        features = []
        boundaries = []
        for index in range(tokens.size - 1):
            step = self._advance_state(state, int(tokens[index]))
            features.append(step.feature)
            boundaries.append(step.observation.boundary)
        targets = tokens[1:].astype(np.int64, copy=False)
        final_patches = state.latent_state.patches + (1 if state.latent_state.patch_length > 0 else 0)
        return SequenceTrace(
            features=np.vstack(features),
            targets=targets,
            boundaries=np.asarray(boundaries, dtype=bool),
            tokens=int(tokens.size),
            patches=final_patches,
        )

    def fit(
        self,
        data: ByteSequenceDataset | str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FitReport:
        sequences = self._coerce_sequences(data)
        feature_batches = []
        target_batches = []
        total_tokens = 0
        total_patches = 0

        for sequence in sequences:
            trace = self.trace(sequence)
            feature_batches.append(trace.features)
            target_batches.append(trace.targets)
            total_tokens += trace.tokens
            total_patches += trace.patches

        design = np.concatenate(feature_batches, axis=0)
        labels = np.concatenate(target_batches, axis=0)
        self.readout.fit(design, labels)
        train_logits = self.readout.logits(design)
        train_bpb = bits_per_byte_from_logits(train_logits, labels)
        mean_patch_size = max(total_tokens - len(sequences), 0) / max(total_patches, 1)

        return FitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            patches=total_patches,
            mean_patch_size=mean_patch_size,
            compression_ratio=mean_patch_size,
            train_bits_per_byte=train_bpb,
        )

    def score(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> SequenceReport:
        trace = self.trace(sequence)
        logits = self.readout.logits(trace.features)
        bpb = bits_per_byte_from_logits(logits, trace.targets)
        total_steps = max(trace.tokens - 1, 0)
        mean_patch_size = total_steps / max(trace.patches, 1)
        return SequenceReport(
            tokens=trace.tokens,
            patches=trace.patches,
            mean_patch_size=mean_patch_size,
            compression_ratio=mean_patch_size,
            bits_per_byte=bpb,
        )

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("Prompt must contain at least one token.")
        state = self._initial_state()
        step: _AdapterStep | None = None
        for token in tokens:
            step = self._advance_state(state, int(token))
        assert step is not None
        return self.readout.probabilities(step.feature[None, :])[0]

    def generate(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        steps: int,
        temperature: float = 1.0,
        greedy: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("Prompt must contain at least one token.")

        rng = np.random.default_rng(seed)
        state = self._initial_state()
        step: _AdapterStep | None = None
        output = tokens.astype(np.uint8, copy=True).tolist()

        for token in tokens:
            step = self._advance_state(state, int(token))
        assert step is not None

        for _ in range(steps):
            logits = self.readout.logits(step.feature[None, :])[0]
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                scaled = logits / temperature
                probs = softmax(scaled[None, :], axis=-1)[0]
                next_token = int(rng.choice(self.config.vocabulary_size, p=probs))
            output.append(next_token)
            step = self._advance_state(state, next_token)

        return np.asarray(output, dtype=np.uint8)


__all__ = [
    "ByteLatentPredictiveCoder",
    "FitReport",
    "SequenceReport",
    "SequenceTrace",
]
