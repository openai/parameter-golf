"""Tap-In V4/V6: document-local phrase matching for eval-time logit nudging.

This ports the weighting logic from abaybektursun PR #1518:
- V4 within-window alignment rule
- V6 cross-window rule
- top-k gating on hinted tokens
- per-fire scaling by match length and vote count
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# C++ matcher
# -----------------------------------------------------------------------------
CPP_SOURCE = r"""
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

struct Scratch {
    std::vector<int32_t> head;
    std::vector<int32_t> tail;
    std::vector<int32_t> fwd;
    std::vector<int32_t> head_lost;
    std::vector<int32_t> tail_lost;
    std::vector<int32_t> fwd_lost;
    std::vector<int32_t> cv_tok;
    std::vector<int32_t> cv_ml;
    std::vector<int32_t> cv_cnt;
    std::vector<int32_t> out_pos;
    std::vector<int32_t> out_tok;
    std::vector<int32_t> out_cross;
    std::vector<int32_t> out_ml;
    std::vector<int32_t> out_vc;
};
static thread_local Scratch S;

// Returns int32[5, N]: (positions, tokens, is_cross, match_lens, vote_counts).
py::array_t<int32_t> tapin_match_v6(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> ids_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> lost_arr,
    py::array_t<float,   py::array::c_style | py::array::forcecast> ent_arr,
    int32_t s_start,
    int32_t min_match,
    int32_t max_match,
    float ent_min,
    bool cross_enabled,
    int32_t cross_base,
    int32_t vocab_size)
{
    const int32_t wlen = (int32_t)ids_arr.size();
    const int32_t lost_len = (int32_t)lost_arr.size();
    const int32_t* ids = ids_arr.data();
    const int32_t* lost = lost_arr.data();
    const float* ent = ent_arr.data();

    if ((int32_t)S.head.size() < vocab_size) S.head.resize(vocab_size, -1);
    if ((int32_t)S.tail.size() < vocab_size) S.tail.resize(vocab_size, -1);
    if ((int32_t)S.fwd.size() < wlen) S.fwd.resize(wlen, -1);
    if (cross_enabled) {
        if ((int32_t)S.head_lost.size() < vocab_size) S.head_lost.resize(vocab_size, -1);
        if ((int32_t)S.tail_lost.size() < vocab_size) S.tail_lost.resize(vocab_size, -1);
        if ((int32_t)S.fwd_lost.size() < std::max(1, lost_len)) {
            S.fwd_lost.resize(std::max(1, lost_len), -1);
        }
    }
    if (S.cv_tok.empty()) {
        S.cv_tok.resize(64);
        S.cv_ml.resize(64);
        S.cv_cnt.resize(64);
    }

    int32_t* head = S.head.data();
    int32_t* tail = S.tail.data();
    int32_t* fwd = S.fwd.data();
    int32_t* hlost = cross_enabled ? S.head_lost.data() : nullptr;
    int32_t* tlost = cross_enabled ? S.tail_lost.data() : nullptr;
    int32_t* flost = cross_enabled ? S.fwd_lost.data() : nullptr;

    int32_t n_fires = 0;
    {
        py::gil_scoped_release release;

        for (int32_t v = 0; v < vocab_size; v++) { head[v] = -1; tail[v] = -1; }
        for (int32_t t = 0; t < wlen; t++) fwd[t] = -1;

        if (cross_enabled && lost_len > 0) {
            for (int32_t v = 0; v < vocab_size; v++) { hlost[v] = -1; tlost[v] = -1; }
            for (int32_t i = 0; i < lost_len; i++) flost[i] = -1;
            for (int32_t i = 0; i < lost_len; i++) {
                int32_t tk = lost[i];
                if ((uint32_t)tk >= (uint32_t)vocab_size) continue;
                if (hlost[tk] < 0) {
                    hlost[tk] = i;
                    tlost[tk] = i;
                } else {
                    flost[tlost[tk]] = i;
                    tlost[tk] = i;
                }
            }
        }

        S.out_pos.clear();
        S.out_tok.clear();
        S.out_cross.clear();
        S.out_ml.clear();
        S.out_vc.clear();

        int32_t* cv_tok_p = S.cv_tok.data();
        int32_t* cv_ml_p = S.cv_ml.data();
        int32_t* cv_cnt_p = S.cv_cnt.data();
        int32_t cv_cap = (int32_t)S.cv_tok.size();

        for (int32_t t = 0; t < wlen; t++) {
            const int32_t tok = ids[t];

            if (t >= s_start && ent[t] >= ent_min && (uint32_t)tok < (uint32_t)vocab_size) {
                int32_t best_len = 0;
                int32_t best_cont = -1;
                int32_t is_cross = 0;
                int32_t cv_n = 0;

                for (int32_t p = head[tok]; p >= 0; p = fwd[p]) {
                    if (p + 1 >= t) continue;
                    int32_t kmax = max_match;
                    if (p < kmax) kmax = p;
                    if (t < kmax) kmax = t;
                    int32_t ml = 1;
                    for (int32_t k = 1; k < kmax; k++) {
                        if (ids[p - k] != ids[t - k]) break;
                        ml++;
                    }
                    if (ml >= min_match) {
                        int32_t cont = ids[p + 1];
                        int32_t found = -1;
                        for (int32_t ci = 0; ci < cv_n; ci++) {
                            if (cv_tok_p[ci] == cont) { found = ci; break; }
                        }
                        if (found < 0) {
                            if (cv_n >= cv_cap) {
                                S.cv_tok.resize(cv_cap * 2);
                                S.cv_ml.resize(cv_cap * 2);
                                S.cv_cnt.resize(cv_cap * 2);
                                cv_tok_p = S.cv_tok.data();
                                cv_ml_p = S.cv_ml.data();
                                cv_cnt_p = S.cv_cnt.data();
                                cv_cap *= 2;
                            }
                            cv_tok_p[cv_n] = cont;
                            cv_ml_p[cv_n] = ml;
                            cv_cnt_p[cv_n] = 1;
                            cv_n++;
                        } else {
                            if (ml > cv_ml_p[found]) cv_ml_p[found] = ml;
                            cv_cnt_p[found]++;
                        }
                        if (ml > best_len) {
                            best_len = ml;
                            best_cont = cont;
                        }
                    }
                }

                if (cross_enabled && lost_len > 0) {
                    int32_t lost_len_at_t = t + cross_base;
                    if (lost_len_at_t > lost_len) lost_len_at_t = lost_len;
                    if (lost_len_at_t > min_match) {
                        for (int32_t p = hlost[tok]; p >= 0; p = flost[p]) {
                            if (p >= lost_len_at_t) break;
                            if (p + 1 >= lost_len_at_t) continue;
                            int32_t kmax = max_match;
                            if (p < kmax) kmax = p;
                            if (t < kmax) kmax = t;
                            int32_t ml = 1;
                            for (int32_t k = 1; k < kmax; k++) {
                                if (lost[p - k] != ids[t - k]) break;
                                ml++;
                            }
                            if (ml >= min_match) {
                                int32_t cont = lost[p + 1];
                                int32_t found = -1;
                                for (int32_t ci = 0; ci < cv_n; ci++) {
                                    if (cv_tok_p[ci] == cont) { found = ci; break; }
                                }
                                if (found < 0) {
                                    if (cv_n >= cv_cap) {
                                        S.cv_tok.resize(cv_cap * 2);
                                        S.cv_ml.resize(cv_cap * 2);
                                        S.cv_cnt.resize(cv_cap * 2);
                                        cv_tok_p = S.cv_tok.data();
                                        cv_ml_p = S.cv_ml.data();
                                        cv_cnt_p = S.cv_cnt.data();
                                        cv_cap *= 2;
                                    }
                                    cv_tok_p[cv_n] = cont;
                                    cv_ml_p[cv_n] = ml;
                                    cv_cnt_p[cv_n] = 1;
                                    cv_n++;
                                } else {
                                    if (ml > cv_ml_p[found]) cv_ml_p[found] = ml;
                                    cv_cnt_p[found]++;
                                }
                                if (ml > best_len) {
                                    best_len = ml;
                                    best_cont = cont;
                                    is_cross = 1;
                                }
                            }
                        }
                    }
                }

                if (best_len >= min_match && best_cont >= 0) {
                    int32_t n_votes = 0;
                    for (int32_t ci = 0; ci < cv_n; ci++) {
                        if (cv_tok_p[ci] == best_cont) { n_votes = cv_cnt_p[ci]; break; }
                    }
                    S.out_pos.push_back(t);
                    S.out_tok.push_back(best_cont);
                    S.out_cross.push_back(is_cross);
                    S.out_ml.push_back(best_len);
                    S.out_vc.push_back(n_votes);
                    n_fires++;
                }
            }

            // Score-before-update: emit hints using the strict prefix, then append.
            if ((uint32_t)tok < (uint32_t)vocab_size) {
                if (head[tok] < 0) {
                    head[tok] = t;
                    tail[tok] = t;
                } else {
                    fwd[tail[tok]] = t;
                    tail[tok] = t;
                }
            }
        }
    }

    py::array_t<int32_t> result({(py::ssize_t)5, (py::ssize_t)n_fires});
    if (n_fires > 0) {
        auto r = result.mutable_unchecked<2>();
        for (int32_t i = 0; i < n_fires; i++) {
            r(0, i) = S.out_pos[i];
            r(1, i) = S.out_tok[i];
            r(2, i) = S.out_cross[i];
            r(3, i) = S.out_ml[i];
            r(4, i) = S.out_vc[i];
        }
    }
    return result;
}
"""

_EXT_BUILD_DIR = Path(__file__).resolve().with_name(".torch_extensions")
_EXT_BUILD_DIR.mkdir(exist_ok=True)

_ext = load_inline(
    name="tapin_v6_cpp_v2",
    cpp_sources=[CPP_SOURCE],
    functions=["tapin_match_v6"],
    extra_cflags=["-O3", "-march=native", "-ffast-math", "-funroll-loops", "-Wall"],
    build_directory=str(_EXT_BUILD_DIR),
    verbose=False,
)

_EMPTY_INT32 = np.empty(0, dtype=np.int32)


def apply_tapin_v4_v6(
    logits: torch.Tensor,
    x_batch: torch.Tensor,
    ws_list: list[int],
    wlens: list[int],
    all_tokens_np,
    tapin_state,
    h,
    bos_token_id: int = 1,
) -> torch.Tensor:
    """Apply the #1518 V4/V6 Tap-In stack to logits in-place."""
    if logits.numel() == 0:
        return logits

    bsz, _, vocab_size = logits.shape
    device = logits.device

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1)

    top_k = min(int(h.tapin_v4_top_k), vocab_size)
    if top_k <= 0:
        return logits
    topk_ids = logits.topk(top_k, dim=-1).indices

    ent_min = float(h.tapin_v4_entropy_min)
    min_match = int(h.tapin_v4_min_match)
    max_match = int(h.tapin_v4_max_match)
    mix_w = float(h.tapin_v4_boost_per_match)
    cross_enabled = bool(getattr(h, "tapin_v6_cross_window", False))
    cross_w = float(h.tapin_v6_cross_w) if cross_enabled else mix_w
    window_size = int(h.eval_seq_len)
    stride = int(h.eval_stride)

    bos_positions = None
    all_tokens_i32 = None
    if cross_enabled and all_tokens_np is not None and tapin_state is not None:
        if "_bos_positions" not in tapin_state:
            all_tokens_arr = np.ascontiguousarray(all_tokens_np, dtype=np.int32)
            tapin_state["_bos_positions"] = np.where(all_tokens_arr == bos_token_id)[0].astype(np.int64)
            tapin_state["_all_tokens_i32"] = all_tokens_arr
        bos_positions = tapin_state["_bos_positions"]
        all_tokens_i32 = tapin_state["_all_tokens_i32"]

    all_bi = []
    all_pos = []
    all_tok = []
    all_cross = []
    all_ml = []
    all_vc = []

    for bi in range(bsz):
        ws = int(ws_list[bi])
        wlen = int(wlens[bi])
        s_start = 0 if ws == 0 else (window_size - stride)

        ids_i32 = x_batch[bi, :wlen].to(torch.int32).cpu().numpy()
        ent_f = entropies[bi, :wlen].float().cpu().numpy()

        cross_base = 0
        lost_np = _EMPTY_INT32
        if cross_enabled and bos_positions is not None and bos_positions.size > 0:
            idx = int(np.searchsorted(bos_positions, ws, side="right")) - 1
            if idx >= 0:
                doc_start_global = int(bos_positions[idx])
                lost_end_max = ws + wlen - window_size
                if lost_end_max > doc_start_global:
                    lost_np = all_tokens_i32[doc_start_global:lost_end_max]
                cross_base = ws - window_size + 1 - doc_start_global

        fires = _ext.tapin_match_v6(
            ids_i32,
            lost_np,
            ent_f,
            s_start,
            min_match,
            max_match,
            ent_min,
            cross_enabled,
            cross_base,
            vocab_size,
        )
        n_fires = int(fires.shape[1])
        if n_fires == 0:
            continue
        all_bi.append(np.full(n_fires, bi, dtype=np.int64))
        all_pos.append(fires[0].astype(np.int64))
        all_tok.append(fires[1].astype(np.int64))
        all_cross.append(fires[2].astype(np.bool_))
        all_ml.append(fires[3].astype(np.float32))
        all_vc.append(fires[4].astype(np.float32))

    if not all_bi:
        return logits

    bi_t = torch.from_numpy(np.concatenate(all_bi)).to(device)
    pos_t = torch.from_numpy(np.concatenate(all_pos)).to(device)
    tok_t = torch.from_numpy(np.concatenate(all_tok)).to(device)
    cross_mask_all = torch.from_numpy(np.concatenate(all_cross)).to(device)
    ml_all = torch.from_numpy(np.concatenate(all_ml)).to(device)
    vote_all = torch.from_numpy(np.concatenate(all_vc)).to(device)

    fire_topk = topk_ids[bi_t, pos_t]
    in_topk = (fire_topk == tok_t.unsqueeze(1)).any(dim=1)

    bi_t = bi_t[in_topk]
    pos_t = pos_t[in_topk]
    if pos_t.numel() == 0:
        return logits

    tok_t = tok_t[in_topk]
    cross_mask = cross_mask_all[in_topk]
    ml_t = ml_all[in_topk]
    vote_t = vote_all[in_topk]

    if tapin_state is not None:
        stats = tapin_state.setdefault("_stats", {"fires": 0, "cross_fires": 0, "within_fires": 0})
        n_fires_total = int(pos_t.shape[0])
        n_cross_total = int(cross_mask.sum().item())
        stats["fires"] += n_fires_total
        stats["cross_fires"] += n_cross_total
        stats["within_fires"] += n_fires_total - n_cross_total

    cur_log_p = log_probs[bi_t, pos_t]
    sym_log_p = cur_log_p.gather(1, tok_t.unsqueeze(1)).squeeze(1)
    sym_p = sym_log_p.exp()

    base_w = torch.where(
        cross_mask,
        torch.full_like(cross_mask, cross_w, dtype=torch.float32),
        torch.full_like(cross_mask, mix_w, dtype=torch.float32),
    )
    length_mult = 1.0 + 0.15 * (ml_t - min_match)
    vote_mult = 1.0 + 0.20 * torch.log(vote_t.clamp(min=1.0))
    w_per_fire = torch.clamp(base_w * length_mult * vote_mult, max=0.50)

    log_one_minus_w = torch.log1p(-w_per_fire + 1e-10)
    new_log_p = cur_log_p + log_one_minus_w.unsqueeze(1)
    new_sym_log_p = torch.log((1.0 - w_per_fire) * sym_p + w_per_fire + 1e-30)
    new_log_p.scatter_(1, tok_t.unsqueeze(1), new_sym_log_p.unsqueeze(1))
    logits[bi_t, pos_t] = new_log_p.to(logits.dtype)
    return logits
