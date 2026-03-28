from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config as scale_conker3_config


def _strict_causal_mask(max_seq_len: int) -> np.ndarray:
    return np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.float32), k=-1)


def _lookback_causal_mask(length: int, lookback: int) -> np.ndarray:
    if lookback <= 0:
        return _strict_causal_mask(length)
    time_idx = np.arange(length, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    return ((delta > 0) & (delta <= lookback)).astype(np.float32)


def _recency_kernel(max_seq_len: int, half_life: float) -> np.ndarray:
    if half_life <= 0.0:
        raise ValueError("Conker-4b recency_half_life must be > 0.")
    decay = float(np.exp(np.log(0.5) / half_life))
    time_idx = np.arange(max_seq_len, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    mask = delta > 0
    safe_delta = np.where(mask, delta - 1, 0).astype(np.float32)
    kernel = np.power(decay, safe_delta, dtype=np.float32)
    return np.where(mask, kernel, 0.0).astype(np.float32)


CLASS_OTHER = 0
CLASS_WORD = 1
CLASS_WORD_BOUNDARY = 2
CLASS_DELIM = 3
CLASS_SPACE = 4

DELIM_NONE = 0
DELIM_BRACKET = 1
DELIM_QUOTE = 2
DELIM_SENTENCE = 3
DELIM_PATH = 4
DELIM_OPERATOR = 5

STACK_NONE = 0
STACK_PAREN = 1
STACK_BRACKET = 2
STACK_BRACE = 3


def _decode_vocab_piece(piece: str) -> tuple[str, bool]:
    leading_space = piece.startswith("▁")
    core = piece[1:] if leading_space else piece
    if core.startswith("<0x") and core.endswith(">") and len(core) == 6:
        try:
            core = bytes([int(core[3:5], 16)]).decode("latin-1")
        except ValueError:
            core = ""
    return core, leading_space


def _parse_vocab_line(line: str) -> tuple[str, float | None]:
    parts = line.rstrip("\n").split("\t")
    piece = parts[0]
    score = None
    if len(parts) > 1:
        try:
            score = float(parts[1])
        except ValueError:
            score = None
    return piece, score


def _classify_vocab_piece(piece: str) -> tuple[int, bool]:
    core, leading_space = _decode_vocab_piece(piece)
    if not core:
        return (CLASS_SPACE if leading_space else CLASS_OTHER), False
    if core.isspace():
        return CLASS_SPACE, False
    punct_chars = set(".,;:!?()[]{}<>\"'`/\\\\|-_=+*&^%$#@~")
    if all(ch in punct_chars for ch in core):
        return CLASS_DELIM, True
    if any(ch.isalnum() for ch in core):
        return (CLASS_WORD_BOUNDARY if leading_space else CLASS_WORD), False
    return CLASS_OTHER, False


def _delimiter_subtype(core: str) -> int:
    if not core:
        return DELIM_NONE
    if all(ch in "()[]{}<>" for ch in core):
        return DELIM_BRACKET
    if all(ch in "\"'`" for ch in core):
        return DELIM_QUOTE
    if all(ch in ".,;:!?" for ch in core):
        return DELIM_SENTENCE
    if all(ch in "/\\\\_-" for ch in core):
        return DELIM_PATH
    if all(ch in "=+*&^%$#@~|" for ch in core):
        return DELIM_OPERATOR
    return DELIM_NONE


def _is_number_like(core: str) -> bool:
    if not core or not any(ch.isdigit() for ch in core):
        return False
    return all(ch in "0123456789.,:+-/%" for ch in core)


def _is_identifier_like(core: str, leading_space: bool) -> bool:
    if not core:
        return False
    has_alpha = any(ch.isalpha() for ch in core)
    has_digit = any(ch.isdigit() for ch in core)
    has_ident_punct = any(ch in "_-/.:@#" for ch in core)
    has_inner_upper = (not leading_space) and any(ch.isupper() for ch in core)
    return (has_alpha and has_digit) or has_ident_punct or has_inner_upper


def _is_urlpath_like(core: str) -> bool:
    if not core:
        return False
    lowered = core.lower()
    pathish_chars = "/.:?=&@%#-_~"
    if any(marker in lowered for marker in ("http", "www", ".com", ".org", ".net", "://", "href", "src=")):
        return True
    return any(ch in pathish_chars for ch in core) and any(ch.isalnum() for ch in core)


def _is_markup_like(core: str) -> bool:
    if not core:
        return False
    lowered = core.lower()
    markup_terms = (
        "<",
        ">",
        "</",
        "/>",
        "html",
        "body",
        "div",
        "span",
        "meta",
        "link",
        "script",
        "style",
        "class",
        "href",
        "src",
        "alt",
        "id=",
        "rel=",
        "type=",
        "content=",
        "data-",
        "&lt",
        "&gt",
        "&amp",
    )
    if any(term in lowered for term in markup_terms):
        return True
    return any(ch in "<>=\"'" for ch in core) and any(ch.isalpha() for ch in core)


def _is_attr_like(core: str) -> bool:
    if not core:
        return False
    lowered = core.lower()
    attr_terms = (
        "class",
        "href",
        "src",
        "alt",
        "id",
        "rel",
        "type",
        "name",
        "value",
        "content",
        "style",
        "title",
        "property",
        "charset",
        "data-",
        "aria-",
        "=",
    )
    if any(term in lowered for term in attr_terms):
        return True
    return "=" in core and any(ch.isalpha() for ch in core)


def _is_entity_like(core: str) -> bool:
    if not core:
        return False
    lowered = core.lower()
    if lowered.startswith("&#") or lowered.startswith("&"):
        return True
    entity_terms = (
        "&amp",
        "&lt",
        "&gt",
        "&quot",
        "&nbsp",
        "&copy",
        "&mdash",
        "&ndash",
        "&rsquo",
        "&ldquo",
        "&rdquo",
    )
    return any(term in lowered for term in entity_terms)


def _stack_opener_type(core: str) -> int:
    if core == "(":
        return STACK_PAREN
    if core == "[":
        return STACK_BRACKET
    if core == "{":
        return STACK_BRACE
    return STACK_NONE


def _stack_closer_type(core: str) -> int:
    if not core:
        return STACK_NONE
    if core[0] == ")" and not any(ch.isalnum() for ch in core[1:]):
        return STACK_PAREN
    if core[0] == "]" and not any(ch.isalnum() for ch in core[1:]):
        return STACK_BRACKET
    if core[0] == "}" and not any(ch.isalnum() for ch in core[1:]):
        return STACK_BRACE
    return STACK_NONE


def _build_vocab_class_luts(
    vocab_path: str | None,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    class_ids = np.zeros((vocab_size,), dtype=np.int32)
    delimiter_mask = np.zeros((vocab_size,), dtype=np.float32)
    number_mask = np.zeros((vocab_size,), dtype=np.float32)
    special_mask = np.zeros((vocab_size,), dtype=np.float32)
    urlpath_mask = np.zeros((vocab_size,), dtype=np.float32)
    markup_mask = np.zeros((vocab_size,), dtype=np.float32)
    attr_mask = np.zeros((vocab_size,), dtype=np.float32)
    entity_mask = np.zeros((vocab_size,), dtype=np.float32)
    stack_open_ids = np.zeros((vocab_size,), dtype=np.int32)
    stack_close_ids = np.zeros((vocab_size,), dtype=np.int32)
    delim_subtype_ids = np.zeros((vocab_size,), dtype=np.int32)
    if vocab_path is None:
        return class_ids, delimiter_mask, number_mask, special_mask, urlpath_mask, markup_mask, attr_mask, entity_mask, stack_open_ids, stack_close_ids, delim_subtype_ids
    path = Path(vocab_path)
    if not path.exists():
        return class_ids, delimiter_mask, number_mask, special_mask, urlpath_mask, markup_mask, attr_mask, entity_mask, stack_open_ids, stack_close_ids, delim_subtype_ids
    scores = np.full((vocab_size,), np.nan, dtype=np.float32)
    with path.open("r", encoding="utf-8") as handle:
        for token_id, line in enumerate(handle):
            if token_id >= vocab_size:
                break
            piece, score = _parse_vocab_line(line)
            cls, is_delim = _classify_vocab_piece(piece)
            class_ids[token_id] = cls
            delimiter_mask[token_id] = 1.0 if is_delim else 0.0
            if score is not None:
                scores[token_id] = score
            core, leading_space = _decode_vocab_piece(piece)
            if _is_number_like(core):
                number_mask[token_id] = 1.0
            if _is_identifier_like(core, leading_space):
                special_mask[token_id] = 1.0
            if _is_urlpath_like(core):
                urlpath_mask[token_id] = 1.0
            if _is_markup_like(core):
                markup_mask[token_id] = 1.0
            if _is_attr_like(core):
                attr_mask[token_id] = 1.0
            if _is_entity_like(core):
                entity_mask[token_id] = 1.0
            stack_open_ids[token_id] = _stack_opener_type(core)
            stack_close_ids[token_id] = _stack_closer_type(core)
            delim_subtype_ids[token_id] = _delimiter_subtype(core)
    finite_scores = scores[np.isfinite(scores)]
    rare_threshold = float(np.quantile(finite_scores, 0.2)) if finite_scores.size else None
    if rare_threshold is not None:
        for token_id in range(vocab_size):
            if not np.isfinite(scores[token_id]):
                continue
            if (
                delimiter_mask[token_id] > 0.0
                or number_mask[token_id] > 0.0
                or special_mask[token_id] > 0.0
                or urlpath_mask[token_id] > 0.0
                or markup_mask[token_id] > 0.0
                or attr_mask[token_id] > 0.0
                or entity_mask[token_id] > 0.0
            ):
                continue
            if class_ids[token_id] not in (CLASS_WORD, CLASS_WORD_BOUNDARY):
                continue
            if scores[token_id] <= rare_threshold:
                special_mask[token_id] = 1.0
    return class_ids, delimiter_mask, number_mask, special_mask, urlpath_mask, markup_mask, attr_mask, entity_mask, stack_open_ids, stack_close_ids, delim_subtype_ids


@dataclass(frozen=True)
class ConkerFourBConfig:
    base_config: ConkerThreeConfig = field(default_factory=ConkerThreeConfig)
    freeze_base: bool = True
    enable_exact1: bool = True
    enable_exact2: bool = True
    enable_exact3: bool = False
    enable_special2: bool = False
    enable_number2: bool = False
    enable_urlpath2: bool = False
    enable_markup2: bool = False
    enable_attr2: bool = False
    enable_entity2: bool = False
    enable_stack2: bool = False
    enable_wordclass2: bool = False
    enable_delim2: bool = False
    enable_delimsub2: bool = False
    enable_recency: bool = True
    tokenizer_vocab_path: str | None = None
    recency_half_life: float = 8.0
    exact_context_span: int = 0
    residual_cap: float = 4.0
    base_feature_scale: float = 1.0
    dynamic_support_gates: bool = False
    dynamic_gate_span: float = 0.5
    gate_only_mode: bool = False
    support_gate_mode: str = "independent"
    support_gate_topk: int = 0
    support_gate_temperature: float = 1.0
    support_overlap_penalty: float = 0.0
    exact1_opens_mask: bool = True
    special2_opens_mask: bool = False
    number2_opens_mask: bool = False
    urlpath2_opens_mask: bool = False
    markup2_opens_mask: bool = False
    attr2_opens_mask: bool = False
    entity2_opens_mask: bool = False
    stack2_opens_mask: bool = False
    wordclass2_opens_mask: bool = False
    delim2_opens_mask: bool = True
    delimsub2_opens_mask: bool = False


class ConkerFourBModel(nn.Module):
    """Frozen Conker-3 base plus sparse exact-context residual calibration."""

    def __init__(self, vocab_size: int, config: ConkerFourBConfig = ConkerFourBConfig()):
        super().__init__()
        if not any(
            (
                config.enable_exact1,
                config.enable_exact2,
                config.enable_exact3,
                config.enable_special2,
                config.enable_number2,
                config.enable_urlpath2,
                config.enable_markup2,
                config.enable_attr2,
                config.enable_entity2,
                config.enable_stack2,
                config.enable_wordclass2,
                config.enable_delim2,
                config.enable_delimsub2,
                config.enable_recency,
            )
        ):
            raise ValueError("Conker-4b must enable at least one residual source.")
        self.vocab_size = vocab_size
        self.config = config
        self.base = ConkerThreeModel(vocab_size=vocab_size, config=config.base_config)
        if config.freeze_base:
            self.base.freeze(recurse=True)

        self.vocab_axis = mx.arange(vocab_size, dtype=mx.int32)
        self.causal_mask = mx.array(_strict_causal_mask(config.base_config.max_seq_len))
        self.recency_kernel = mx.array(_recency_kernel(config.base_config.max_seq_len, config.recency_half_life))
        class_ids, delimiter_mask, number_mask, special_mask, urlpath_mask, markup_mask, attr_mask, entity_mask, stack_open_ids, stack_close_ids, delim_subtype_ids = _build_vocab_class_luts(
            config.tokenizer_vocab_path, vocab_size
        )
        class_vocab_size = max(int(class_ids.max(initial=0)) + 1, 1)
        delim_subtype_vocab_size = max(int(delim_subtype_ids.max(initial=0)) + 1, 1)
        class_to_token_mask = np.zeros((class_vocab_size, vocab_size), dtype=np.float32)
        class_to_token_mask[class_ids, np.arange(vocab_size)] = 1.0
        wordclass_token_mask = np.zeros((class_vocab_size, vocab_size), dtype=np.float32)
        wordclass_rows = [CLASS_WORD, CLASS_WORD_BOUNDARY]
        for row in wordclass_rows:
            if row < class_vocab_size:
                wordclass_token_mask[row] = class_to_token_mask[row]
        self.token_class_ids = mx.array(class_ids)
        self.class_axis = mx.arange(class_vocab_size, dtype=mx.int32)
        self.class_to_token_mask = mx.array(class_to_token_mask)
        self.wordclass_token_mask = mx.array(wordclass_token_mask)
        self.delimiter_mask = mx.array(delimiter_mask)
        self.number_mask = mx.array(number_mask)
        self.special_mask = mx.array(special_mask)
        self.urlpath_mask = mx.array(urlpath_mask)
        self.markup_mask = mx.array(markup_mask)
        self.attr_mask = mx.array(attr_mask)
        self.entity_mask = mx.array(entity_mask)
        self.stack_open_ids = mx.array(stack_open_ids)
        self.stack_close_ids = mx.array(stack_close_ids)
        stack_vocab_size = max(int(max(stack_open_ids.max(initial=0), stack_close_ids.max(initial=0))) + 1, 1)
        stack_closer_token_mask = np.zeros((stack_vocab_size, vocab_size), dtype=np.float32)
        stack_closer_token_mask[stack_close_ids, np.arange(vocab_size)] = 1.0
        if stack_vocab_size > 0:
            stack_closer_token_mask[STACK_NONE] = 0.0
        self.stack_closer_token_mask = mx.array(stack_closer_token_mask)
        delim_subtype_to_token_mask = np.zeros((delim_subtype_vocab_size, vocab_size), dtype=np.float32)
        delim_subtype_to_token_mask[delim_subtype_ids, np.arange(vocab_size)] = 1.0
        if delim_subtype_vocab_size > 0:
            delim_subtype_to_token_mask[DELIM_NONE] = 0.0
        self.delim_subtype_ids = mx.array(delim_subtype_ids)
        self.delim_subtype_axis = mx.arange(delim_subtype_vocab_size, dtype=mx.int32)
        self.delim_subtype_token_mask = mx.array(delim_subtype_to_token_mask)

        self.w_exact1 = mx.array(0.0, dtype=mx.float32)
        self.w_exact2 = mx.array(0.0, dtype=mx.float32)
        self.w_exact3 = mx.array(0.0, dtype=mx.float32)
        self.w_special2 = mx.array(0.0, dtype=mx.float32)
        self.w_number2 = mx.array(0.0, dtype=mx.float32)
        self.w_urlpath2 = mx.array(0.0, dtype=mx.float32)
        self.w_markup2 = mx.array(0.0, dtype=mx.float32)
        self.w_attr2 = mx.array(0.0, dtype=mx.float32)
        self.w_entity2 = mx.array(0.0, dtype=mx.float32)
        self.w_stack2 = mx.array(0.0, dtype=mx.float32)
        self.w_wordclass2 = mx.array(0.0, dtype=mx.float32)
        self.w_delim2 = mx.array(0.0, dtype=mx.float32)
        self.w_delimsub2 = mx.array(0.0, dtype=mx.float32)
        self.w_recency = mx.array(0.0, dtype=mx.float32)
        self.w_exact1_flag = mx.array(0.0, dtype=mx.float32)
        self.w_exact2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_exact3_flag = mx.array(0.0, dtype=mx.float32)
        self.w_special2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_number2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_urlpath2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_markup2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_attr2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_entity2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_stack2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_wordclass2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_delim2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_delimsub2_flag = mx.array(0.0, dtype=mx.float32)
        self.w_recency_flag = mx.array(0.0, dtype=mx.float32)
        self.w_base = mx.array(0.0, dtype=mx.float32)
        self.bias = mx.array(0.0, dtype=mx.float32)
        self.support_gate_sources = (
            "exact1",
            "special2",
            "number2",
            "urlpath2",
            "markup2",
            "attr2",
            "entity2",
            "stack2",
            "wordclass2",
            "delim2",
            "delimsub2",
        )
        self.support_gate_source_to_idx = {name: idx for idx, name in enumerate(self.support_gate_sources)}
        gate_feature_dim = 6
        self.support_gate_feature_weights = mx.zeros((len(self.support_gate_sources), gate_feature_dim), dtype=mx.float32)
        self.support_gate_mass_weights = mx.zeros((len(self.support_gate_sources),), dtype=mx.float32)
        self.support_gate_flag_weights = mx.zeros((len(self.support_gate_sources),), dtype=mx.float32)
        self.support_gate_bias = mx.zeros((len(self.support_gate_sources),), dtype=mx.float32)
        self.support_gate_abstain_feature_weights = mx.zeros((gate_feature_dim,), dtype=mx.float32)
        self.support_gate_abstain_bias = mx.array(0.0, dtype=mx.float32)

        self.freeze(
            keys=(
                "vocab_axis",
                "causal_mask",
                "recency_kernel",
                "token_class_ids",
                "class_axis",
                "class_to_token_mask",
                "wordclass_token_mask",
                "delimiter_mask",
                "number_mask",
                "special_mask",
                "urlpath_mask",
                "markup_mask",
                "attr_mask",
                "entity_mask",
                "stack_open_ids",
                "stack_close_ids",
                "stack_closer_token_mask",
                "delim_subtype_ids",
                "delim_subtype_axis",
                "delim_subtype_token_mask",
            ),
            strict=False,
        )

    def _one_hot(self, chars: mx.array) -> mx.array:
        return mx.where(chars[..., None] == self.vocab_axis[None, None, :], 1.0, 0.0)

    def _support_gate_features(self, chars: mx.array, base_logits: mx.array) -> mx.array:
        entropy, max_logit, variance = self.base._logit_features(base_logits)
        zeros = mx.zeros_like(entropy)
        if self.base.config.enable_linear:
            states, _ = self.base._linear_states(chars)
            abs_states = mx.abs(states)
            total_energy = mx.mean(abs_states, axis=-1)
            if self.base.non_osc_modes > 0:
                non_osc_energy = mx.mean(abs_states[:, :, : self.base.non_osc_modes], axis=-1)
            else:
                non_osc_energy = zeros
            if self.base.osc_mode_count > 0:
                osc_energy = mx.mean(abs_states[:, :, self.base.non_osc_modes :], axis=-1)
            else:
                osc_energy = zeros
        else:
            total_energy = zeros
            non_osc_energy = zeros
            osc_energy = zeros
        return mx.stack(
            [entropy, max_logit, variance, total_energy, non_osc_energy, osc_energy],
            axis=-1,
        )

    def _source_gate_logit(
        self,
        source_name: str,
        source: mx.array | None,
        base_features: mx.array | None,
        dtype: mx.Dtype,
    ) -> mx.array | None:
        if (
            source is None
            or base_features is None
            or (not self.config.dynamic_support_gates and not self.config.gate_only_mode)
        ):
            return None
        source_idx = self.support_gate_source_to_idx.get(source_name)
        if source_idx is None:
            return None
        feature_weights = self.support_gate_feature_weights[source_idx]
        mass_weight = self.support_gate_mass_weights[source_idx]
        flag_weight = self.support_gate_flag_weights[source_idx]
        bias = self.support_gate_bias[source_idx]
        source_mass = mx.log1p(mx.sum(source, axis=-1))
        source_flag = (mx.sum(source, axis=-1) > 0).astype(dtype)
        gate_pre = (
            mx.sum(base_features * feature_weights[None, None, :], axis=-1)
            + mass_weight * source_mass
            + flag_weight * source_flag
            + bias
        )
        return gate_pre.astype(dtype)

    def _source_gate(
        self,
        source_name: str,
        source: mx.array | None,
        base_features: mx.array | None,
        opens_mask: bool,
        dtype: mx.Dtype,
    ) -> mx.array:
        if self.config.support_gate_mode == "softmax" and not opens_mask and source_name in self.support_gate_source_to_idx:
            raise RuntimeError("Softmax support gates must be resolved through _support_ownership_gates().")
        if (
            source is None
            or base_features is None
            or (not self.config.dynamic_support_gates and not self.config.gate_only_mode)
            or (opens_mask and not self.config.gate_only_mode)
        ):
            return mx.array(1.0, dtype=dtype)
        gate_logit = self._source_gate_logit(source_name, source, base_features, dtype)
        if gate_logit is None:
            return mx.array(1.0, dtype=dtype)
        return 1.0 + self.config.dynamic_gate_span * mx.tanh(gate_logit)[..., None]

    def _support_ownership_gates(
        self,
        support_sources: dict[str, mx.array],
        base_features: mx.array | None,
        dtype: mx.Dtype,
    ) -> dict[str, mx.array]:
        if self.config.support_gate_mode != "softmax" or base_features is None:
            return {}
        active_items = [(name, source) for name, source in support_sources.items() if source is not None]
        if not active_items:
            return {}
        logits = []
        names = []
        for name, source in active_items:
            gate_logit = self._source_gate_logit(name, source, base_features, dtype)
            if gate_logit is None:
                gate_logit = mx.zeros(base_features.shape[:2], dtype=dtype)
            logits.append(gate_logit)
            names.append(name)
        abstain_logit = (
            mx.sum(base_features * self.support_gate_abstain_feature_weights[None, None, :], axis=-1)
            + self.support_gate_abstain_bias
        ).astype(dtype)
        gate_logits = mx.stack(logits + [abstain_logit], axis=-1)
        gate_probs = mx.softmax(gate_logits, axis=-1)
        support_probs = gate_probs[..., :-1]
        support_mass = 1.0 - gate_probs[..., -1:]
        if self.config.support_gate_topk > 0:
            k = min(self.config.support_gate_topk, support_probs.shape[-1])
            thresh = mx.sort(support_probs, axis=-1)[..., -k:]
            cutoff = thresh[..., :1]
            top_mask = (support_probs >= cutoff).astype(dtype)
            support_probs = support_probs * top_mask
            denom = mx.maximum(mx.sum(support_probs, axis=-1, keepdims=True), mx.array(1e-6, dtype=dtype))
            support_probs = support_mass * (support_probs / denom)
        else:
            support_probs = support_mass * support_probs
        return {
            name: support_probs[..., idx : idx + 1]
            for idx, name in enumerate(names)
        }

    def _independent_support_gate(
        self,
        source_name: str,
        source: mx.array | None,
        base_features: mx.array | None,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array] | tuple[None, None]:
        gate_logit = self._source_gate_logit(source_name, source, base_features, dtype)
        if gate_logit is None or source is None:
            return None, None
        temperature = max(self.config.support_gate_temperature, 1e-4)
        activation = mx.sigmoid(gate_logit / temperature) * (mx.sum(source, axis=-1) > 0).astype(dtype)
        gate = 1.0 + self.config.dynamic_gate_span * (2.0 * activation[..., None] - 1.0)
        return gate, activation

    def _source_term(
        self,
        source_name: str,
        source: mx.array | None,
        base_features: mx.array | None,
        opens_mask: bool,
        dtype: mx.Dtype,
        weight: mx.array,
        flag_weight: mx.array,
        fixed_gate: mx.array | None = None,
    ) -> mx.array | None:
        if source is None:
            return None
        gate = fixed_gate if fixed_gate is not None else self._source_gate(source_name, source, base_features, opens_mask, dtype)
        source_flag = (source > 0).astype(dtype)
        if self.config.gate_only_mode:
            return gate * (mx.log1p(source) + source_flag)
        return gate * (weight * mx.log1p(source) + flag_weight * source_flag)

    def _count_features_core(
        self,
        chars: mx.array,
        mask: mx.array,
    ) -> tuple[
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
    ]:
        batch, timesteps = chars.shape
        one_hot = self._one_hot(chars)
        zeros = mx.zeros((batch, 1, self.vocab_size), dtype=one_hot.dtype)
        next_one_hot = mx.concatenate([one_hot[:, 1:, :], zeros], axis=1)
        prev_one_hot = mx.concatenate([zeros, one_hot[:, :-1, :]], axis=1)

        exact1 = exact2 = exact3 = special2 = number2 = urlpath2 = markup2 = attr2 = entity2 = stack2 = wordclass2 = delim2 = delimsub2 = recency = None

        if any(
            (
                self.config.enable_exact1,
                self.config.enable_exact2,
                self.config.enable_exact3,
                self.config.enable_special2,
                self.config.enable_number2,
                self.config.enable_urlpath2,
                self.config.enable_markup2,
                self.config.enable_attr2,
                self.config.enable_entity2,
            )
        ):
            cur_match = mx.matmul(one_hot, mx.transpose(one_hot, (0, 2, 1))) * mask[None, :, :]
            if self.config.enable_exact1:
                exact1 = mx.matmul(cur_match, next_one_hot)
            if any((self.config.enable_exact2, self.config.enable_exact3, self.config.enable_special2, self.config.enable_number2, self.config.enable_urlpath2, self.config.enable_markup2, self.config.enable_attr2, self.config.enable_entity2)):
                prev_match = mx.matmul(prev_one_hot, mx.transpose(prev_one_hot, (0, 2, 1)))
                pair_match = cur_match * prev_match
                if self.config.enable_exact2:
                    exact2 = mx.matmul(pair_match, next_one_hot)
                if self.config.enable_exact3:
                    prev2_one_hot = mx.concatenate([mx.zeros((batch, 2, self.vocab_size), dtype=one_hot.dtype), one_hot[:, :-2, :]], axis=1)
                    prev2_match = mx.matmul(prev2_one_hot, mx.transpose(prev2_one_hot, (0, 2, 1)))
                    exact3 = mx.matmul(pair_match * prev2_match, next_one_hot)
                if self.config.enable_special2:
                    special2 = mx.matmul(pair_match, next_one_hot * self.special_mask[None, None, :])
                if self.config.enable_number2:
                    number2 = mx.matmul(pair_match, next_one_hot * self.number_mask[None, None, :])
                if self.config.enable_urlpath2:
                    urlpath2 = mx.matmul(pair_match, next_one_hot * self.urlpath_mask[None, None, :])
                if self.config.enable_markup2:
                    markup2 = mx.matmul(pair_match, next_one_hot * self.markup_mask[None, None, :])
                if self.config.enable_attr2:
                    attr2 = mx.matmul(pair_match, next_one_hot * self.attr_mask[None, None, :])
                if self.config.enable_entity2:
                    entity2 = mx.matmul(pair_match, next_one_hot * self.entity_mask[None, None, :])

        class_ids = self.token_class_ids[chars]
        prev_class_ids = mx.concatenate([mx.zeros((batch, 1), dtype=class_ids.dtype), class_ids[:, :-1]], axis=1)
        next_class_ids = mx.concatenate([class_ids[:, 1:], mx.zeros((batch, 1), dtype=class_ids.dtype)], axis=1)

        if self.config.enable_wordclass2:
            next_class_one_hot = (next_class_ids[..., None] == self.class_axis[None, None, :]).astype(one_hot.dtype)
            cur_class_match = (class_ids[:, :, None] == class_ids[:, None, :]).astype(one_hot.dtype) * mask[None, :, :]
            prev_class_match = (prev_class_ids[:, :, None] == prev_class_ids[:, None, :]).astype(one_hot.dtype)
            class2_counts = mx.matmul(cur_class_match * prev_class_match, next_class_one_hot)
            wordclass2 = mx.matmul(class2_counts, self.wordclass_token_mask)

        if self.config.enable_delim2:
            next_delim = next_one_hot * self.delimiter_mask[None, None, :]
            cur_class_match = (class_ids[:, :, None] == class_ids[:, None, :]).astype(one_hot.dtype) * mask[None, :, :]
            prev_class_match = (prev_class_ids[:, :, None] == prev_class_ids[:, None, :]).astype(one_hot.dtype)
            delim2 = mx.matmul(cur_class_match * prev_class_match, next_delim)

        if self.config.enable_delimsub2:
            delim_subtype_ids = self.delim_subtype_ids[chars]
            prev_delim_subtype_ids = mx.concatenate([mx.zeros((batch, 1), dtype=delim_subtype_ids.dtype), delim_subtype_ids[:, :-1]], axis=1)
            next_delim_subtype_ids = mx.concatenate([delim_subtype_ids[:, 1:], mx.zeros((batch, 1), dtype=delim_subtype_ids.dtype)], axis=1)
            next_delim_subtype_one_hot = (next_delim_subtype_ids[..., None] == self.delim_subtype_axis[None, None, :]).astype(one_hot.dtype)
            cur_delim_match = (delim_subtype_ids[:, :, None] == delim_subtype_ids[:, None, :]).astype(one_hot.dtype) * mask[None, :, :]
            prev_delim_match = (prev_delim_subtype_ids[:, :, None] == prev_delim_subtype_ids[:, None, :]).astype(one_hot.dtype)
            delimsub_counts = mx.matmul(cur_delim_match * prev_delim_match, next_delim_subtype_one_hot)
            delimsub2 = mx.matmul(delimsub_counts, self.delim_subtype_token_mask)

        if self.config.enable_recency:
            recency_kernel = mx.array(_recency_kernel(timesteps, self.config.recency_half_life))
            if self.config.exact_context_span > 0:
                recency_kernel = recency_kernel * mask
            recency = mx.matmul(mx.broadcast_to(recency_kernel[None, :, :], (batch, timesteps, timesteps)), one_hot)

        if self.config.enable_stack2:
            char_np = np.array(chars, dtype=np.int32, copy=False)
            open_np = np.array(self.stack_open_ids, dtype=np.int32, copy=False)
            close_np = np.array(self.stack_close_ids, dtype=np.int32, copy=False)
            closer_mask_np = np.array(self.stack_closer_token_mask, dtype=np.float32, copy=False)
            stack_counts = np.zeros((batch, timesteps, self.vocab_size), dtype=np.float32)
            for b in range(batch):
                stack: list[int] = []
                for t in range(timesteps):
                    token_id = int(char_np[b, t])
                    close_type = int(close_np[token_id])
                    if close_type != STACK_NONE and stack and stack[-1] == close_type:
                        stack.pop()
                    open_type = int(open_np[token_id])
                    if open_type != STACK_NONE:
                        stack.append(open_type)
                    if stack:
                        top = stack[-1]
                        stack_counts[b, t, :] = closer_mask_np[top] * float(len(stack))
            stack2 = mx.array(stack_counts)

        return exact1, exact2, exact3, special2, number2, urlpath2, markup2, attr2, entity2, stack2, wordclass2, delim2, delimsub2, recency

    def _count_features(
        self,
        chars: mx.array,
    ) -> tuple[
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
    ]:
        batch, timesteps = chars.shape
        if timesteps > self.config.base_config.max_seq_len:
            raise ValueError(
                f"Conker-4b max_seq_len={self.config.base_config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.config.exact_context_span <= 0:
            mask = self.causal_mask[:timesteps, :timesteps]
            return self._count_features_core(chars, mask)

        total_timesteps = batch * timesteps
        flat_chars = chars.reshape(1, total_timesteps)
        flat_mask = mx.array(_lookback_causal_mask(total_timesteps, self.config.exact_context_span))
        flat_features = self._count_features_core(flat_chars, flat_mask)

        def _reshape(feature: mx.array | None) -> mx.array | None:
            if feature is None:
                return None
            return feature.reshape(batch, timesteps, self.vocab_size)

        return tuple(_reshape(feature) for feature in flat_features)

    def _forward_impl(
        self,
        chars: mx.array,
        return_support_activations: bool = False,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        base_logits = self.base(chars)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)

        exact1, exact2, exact3, special2, number2, urlpath2, markup2, attr2, entity2, stack2, wordclass2, delim2, delimsub2, recency = self._count_features(chars)

        base_log_probs = base_logits - mx.logsumexp(base_logits, axis=-1, keepdims=True)
        base_centered = self.config.base_feature_scale * (
            base_log_probs - mx.mean(base_log_probs, axis=-1, keepdims=True)
        )
        support_gate_features = self._support_gate_features(chars, base_logits) if self.config.dynamic_support_gates else None
        support_ownership_gates = self._support_ownership_gates(
            {
                "exact1": exact1,
                "special2": special2,
                "number2": number2,
                "urlpath2": urlpath2,
                "markup2": markup2,
                "attr2": attr2,
                "entity2": entity2,
                "stack2": stack2,
                "wordclass2": wordclass2,
                "delim2": delim2,
                "delimsub2": delimsub2,
            },
            support_gate_features,
            base_logits.dtype,
        )
        support_activations: dict[str, mx.array] = {}
        support_independent_gates: dict[str, mx.array] = {}
        if self.config.support_gate_mode == "independent":
            for source_name, source in (
                ("exact1", exact1),
                ("special2", special2),
                ("number2", number2),
                ("urlpath2", urlpath2),
                ("markup2", markup2),
                ("attr2", attr2),
                ("entity2", entity2),
                ("stack2", stack2),
                ("wordclass2", wordclass2),
                ("delim2", delim2),
                ("delimsub2", delimsub2),
            ):
                gate, activation = self._independent_support_gate(
                    source_name,
                    source,
                    support_gate_features,
                    base_logits.dtype,
                )
                if gate is not None and activation is not None:
                    support_independent_gates[source_name] = gate
                    support_activations[source_name] = activation

        pre = mx.zeros_like(base_logits) if self.config.gate_only_mode else mx.broadcast_to(self.bias, base_logits.shape)
        candidate_mask = mx.zeros(base_logits.shape, dtype=base_logits.dtype)

        if exact1 is not None:
            exact1_term = self._source_term(
                "exact1", exact1, support_gate_features, self.config.exact1_opens_mask, base_logits.dtype, self.w_exact1, self.w_exact1_flag,
                fixed_gate=support_ownership_gates.get("exact1") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("exact1"),
            )
            pre = pre + exact1_term
            if self.config.exact1_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (exact1 > 0).astype(base_logits.dtype))

        if exact2 is not None:
            exact2_term = self._source_term(
                "exact2", exact2, support_gate_features, True, base_logits.dtype, self.w_exact2, self.w_exact2_flag
            )
            pre = pre + exact2_term
            candidate_mask = mx.maximum(candidate_mask, (exact2 > 0).astype(base_logits.dtype))

        if exact3 is not None:
            exact3_term = self._source_term(
                "exact3", exact3, support_gate_features, True, base_logits.dtype, self.w_exact3, self.w_exact3_flag
            )
            pre = pre + exact3_term
            candidate_mask = mx.maximum(candidate_mask, (exact3 > 0).astype(base_logits.dtype))

        if special2 is not None:
            special2_term = self._source_term(
                "special2", special2, support_gate_features, self.config.special2_opens_mask, base_logits.dtype, self.w_special2, self.w_special2_flag,
                fixed_gate=support_ownership_gates.get("special2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("special2"),
            )
            pre = pre + special2_term
            if self.config.special2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (special2 > 0).astype(base_logits.dtype))

        if number2 is not None:
            number2_term = self._source_term(
                "number2", number2, support_gate_features, self.config.number2_opens_mask, base_logits.dtype, self.w_number2, self.w_number2_flag,
                fixed_gate=support_ownership_gates.get("number2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("number2"),
            )
            pre = pre + number2_term
            if self.config.number2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (number2 > 0).astype(base_logits.dtype))

        if urlpath2 is not None:
            urlpath2_term = self._source_term(
                "urlpath2", urlpath2, support_gate_features, self.config.urlpath2_opens_mask, base_logits.dtype, self.w_urlpath2, self.w_urlpath2_flag,
                fixed_gate=support_ownership_gates.get("urlpath2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("urlpath2"),
            )
            pre = pre + urlpath2_term
            if self.config.urlpath2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (urlpath2 > 0).astype(base_logits.dtype))

        if markup2 is not None:
            markup2_term = self._source_term(
                "markup2", markup2, support_gate_features, self.config.markup2_opens_mask, base_logits.dtype, self.w_markup2, self.w_markup2_flag,
                fixed_gate=support_ownership_gates.get("markup2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("markup2"),
            )
            pre = pre + markup2_term
            if self.config.markup2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (markup2 > 0).astype(base_logits.dtype))

        if attr2 is not None:
            attr2_term = self._source_term(
                "attr2", attr2, support_gate_features, self.config.attr2_opens_mask, base_logits.dtype, self.w_attr2, self.w_attr2_flag,
                fixed_gate=support_ownership_gates.get("attr2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("attr2"),
            )
            pre = pre + attr2_term
            if self.config.attr2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (attr2 > 0).astype(base_logits.dtype))

        if entity2 is not None:
            entity2_term = self._source_term(
                "entity2", entity2, support_gate_features, self.config.entity2_opens_mask, base_logits.dtype, self.w_entity2, self.w_entity2_flag,
                fixed_gate=support_ownership_gates.get("entity2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("entity2"),
            )
            pre = pre + entity2_term
            if self.config.entity2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (entity2 > 0).astype(base_logits.dtype))

        if stack2 is not None:
            stack2_term = self._source_term(
                "stack2", stack2, support_gate_features, self.config.stack2_opens_mask, base_logits.dtype, self.w_stack2, self.w_stack2_flag,
                fixed_gate=support_ownership_gates.get("stack2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("stack2"),
            )
            pre = pre + stack2_term
            if self.config.stack2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (stack2 > 0).astype(base_logits.dtype))

        if wordclass2 is not None:
            wordclass2_term = self._source_term(
                "wordclass2", wordclass2, support_gate_features, self.config.wordclass2_opens_mask, base_logits.dtype, self.w_wordclass2, self.w_wordclass2_flag,
                fixed_gate=support_ownership_gates.get("wordclass2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("wordclass2"),
            )
            pre = pre + wordclass2_term
            if self.config.wordclass2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (wordclass2 > 0).astype(base_logits.dtype))

        if delim2 is not None:
            delim2_term = self._source_term(
                "delim2", delim2, support_gate_features, self.config.delim2_opens_mask, base_logits.dtype, self.w_delim2, self.w_delim2_flag,
                fixed_gate=support_ownership_gates.get("delim2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("delim2"),
            )
            pre = pre + delim2_term
            if self.config.delim2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (delim2 > 0).astype(base_logits.dtype))

        if delimsub2 is not None:
            delimsub2_term = self._source_term(
                "delimsub2", delimsub2, support_gate_features, self.config.delimsub2_opens_mask, base_logits.dtype, self.w_delimsub2, self.w_delimsub2_flag,
                fixed_gate=support_ownership_gates.get("delimsub2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("delimsub2"),
            )
            pre = pre + delimsub2_term
            if self.config.delimsub2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (delimsub2 > 0).astype(base_logits.dtype))

        if recency is not None:
            recency_term = self._source_term(
                "recency", recency, support_gate_features, True, base_logits.dtype, self.w_recency, self.w_recency_flag
            )
            pre = pre + recency_term
            candidate_mask = mx.maximum(candidate_mask, (recency > 0).astype(base_logits.dtype))

        if not self.config.gate_only_mode:
            pre = pre + self.w_base * base_centered
        residual = candidate_mask * (self.config.residual_cap * mx.tanh(pre / self.config.residual_cap))
        return base_logits + residual, (support_activations if return_support_activations else None)

    def __call__(self, chars: mx.array) -> mx.array:
        logits, _ = self._forward_impl(chars, return_support_activations=False)
        return logits

    def supervised_loss(self, x: mx.array, y: mx.array) -> mx.array:
        logits, support_activations = self._forward_impl(x, return_support_activations=True)
        batch_size, timesteps, vocab_size = logits.shape
        loss = mx.mean(
            nn.losses.cross_entropy(
                logits.reshape(batch_size * timesteps, vocab_size),
                y.reshape(batch_size * timesteps),
            )
        )
        if (
            self.config.support_overlap_penalty > 0.0
            and support_activations is not None
            and len(support_activations) >= 2
        ):
            acts = mx.stack(list(support_activations.values()), axis=-1)
            sum_acts = mx.sum(acts, axis=-1)
            pairwise = 0.5 * (sum_acts * sum_acts - mx.sum(acts * acts, axis=-1))
            denom = max((acts.shape[-1] * (acts.shape[-1] - 1)) / 2.0, 1.0)
            loss = loss + self.config.support_overlap_penalty * mx.mean(pairwise / denom)
        return loss


def scale_config(config: ConkerFourBConfig, scale: float) -> ConkerFourBConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        base_config=scale_conker3_config(config.base_config, scale),
    )
