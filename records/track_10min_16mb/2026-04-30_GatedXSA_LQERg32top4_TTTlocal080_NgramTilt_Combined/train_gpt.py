import base64, collections, copy, ctypes, fcntl, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
from flash_attn_interface import (
    flash_attn_func as flash_attn_3_func,
    flash_attn_varlen_func,
)
from concurrent.futures import ThreadPoolExecutor
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor




# ===== Inlined online n-gram tilt helpers =====
# Keep the eval-time n-gram scorer and native state machine in this script so
# the submitted training/eval logic is self-contained and fully counted here.
_ONLINE_NGRAM_STATE_C = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COEFF_COUNT 32

static const uint64_t ROLLING_COEFFS[COEFF_COUNT] = {
    36313ULL,   27191ULL,   51647ULL,   81929ULL,   131071ULL,  196613ULL,
    262147ULL,  393241ULL,  524309ULL,  655373ULL,  786433ULL,  917521ULL,
    1048583ULL, 1179653ULL, 1310729ULL, 1441801ULL, 1572869ULL, 1703941ULL,
    1835017ULL, 1966087ULL, 2097169ULL, 2228243ULL, 2359319ULL, 2490389ULL,
    2621471ULL, 2752549ULL, 2883617ULL, 3014687ULL, 3145757ULL, 3276833ULL,
    3407903ULL, 3538973ULL,
};

static const uint64_t PAIR_MIX = 1000003ULL;
static const uint64_t PREFIX_BASE = 1099511628211ULL;
static const uint64_t LEN_MIX = 0x9E3779B185EBCA87ULL;
static const uint64_t TABLE_MIX = 0x9e3779b97f4a7c15ULL;

typedef struct {
    uint64_t key;
    uint32_t total;
    uint32_t top_count;
    uint16_t top_tok;
    uint16_t _pad;
} CtxBucket;

typedef struct {
    uint64_t key;
    uint32_t count;
    uint32_t _pad;
} PairBucket;

typedef struct {
    int token_ctx_len;
    int token_prefix_len;
    int token_head;
    uint16_t *token_ring;

    CtxBucket *token_ctx_tbl;
    uint8_t *token_ctx_used;
    size_t token_ctx_mask;

    PairBucket *token_pair_tbl;
    uint8_t *token_pair_used;
    size_t token_pair_mask;

    uint64_t within_hash;
    uint32_t within_len;

    CtxBucket *within_ctx_tbl;
    uint8_t *within_ctx_used;
    size_t within_ctx_mask;

    PairBucket *within_pair_tbl;
    uint8_t *within_pair_used;
    size_t within_pair_mask;
} OnlineNgramState;

static inline size_t mix_index(uint64_t key, size_t mask) {
    return (size_t)((key * TABLE_MIX) & mask);
}

static inline size_t find_ctx_slot(
    CtxBucket *tbl,
    uint8_t *used,
    size_t mask,
    uint64_t key,
    int *found
) {
    size_t idx = mix_index(key, mask);
    for (size_t probe = 0; probe <= mask; ++probe) {
        if (!used[idx]) {
            *found = 0;
            return idx;
        }
        if (tbl[idx].key == key) {
            *found = 1;
            return idx;
        }
        idx = (idx + 1U) & mask;
    }
    *found = -1;
    return 0;
}

static inline size_t find_pair_slot(
    PairBucket *tbl,
    uint8_t *used,
    size_t mask,
    uint64_t key,
    int *found
) {
    size_t idx = mix_index(key, mask);
    for (size_t probe = 0; probe <= mask; ++probe) {
        if (!used[idx]) {
            *found = 0;
            return idx;
        }
        if (tbl[idx].key == key) {
            *found = 1;
            return idx;
        }
        idx = (idx + 1U) & mask;
    }
    *found = -1;
    return 0;
}

static inline uint64_t token_pair_key(uint64_t ctx_key, uint16_t tok, int ctx_len) {
    return (ctx_key * PAIR_MIX) ^ (((uint64_t)tok) * ROLLING_COEFFS[(size_t)ctx_len % COEFF_COUNT]);
}

static inline uint64_t within_pair_key(uint64_t ctx_key, uint16_t tok) {
    return (ctx_key * PAIR_MIX) ^ (((uint64_t)tok) * ROLLING_COEFFS[0]);
}

static inline uint64_t extend_prefix_hash(uint64_t current_hash, uint16_t tok, uint32_t pos) {
    return (current_hash * PREFIX_BASE) ^ (((uint64_t)tok + 1ULL) * ROLLING_COEFFS[(size_t)pos % COEFF_COUNT]);
}

static inline uint32_t pair_increment(
    PairBucket *tbl,
    uint8_t *used,
    size_t mask,
    uint64_t key
) {
    int found = 0;
    size_t idx = find_pair_slot(tbl, used, mask, key, &found);
    if (found < 0) {
        return 0U;
    }
    if (!found) {
        used[idx] = 1U;
        tbl[idx].key = key;
        tbl[idx].count = 1U;
        return 1U;
    }
    tbl[idx].count += 1U;
    return tbl[idx].count;
}

static inline int ctx_increment(
    CtxBucket *tbl,
    uint8_t *used,
    size_t mask,
    uint64_t key,
    uint16_t tok,
    uint32_t pair_count
) {
    int found = 0;
    size_t idx = find_ctx_slot(tbl, used, mask, key, &found);
    if (found < 0) {
        return -1;
    }
    if (!found) {
        used[idx] = 1U;
        tbl[idx].key = key;
        tbl[idx].total = 1U;
        tbl[idx].top_count = pair_count;
        tbl[idx].top_tok = tok;
        return 0;
    }
    tbl[idx].total += 1U;
    if (pair_count > tbl[idx].top_count) {
        tbl[idx].top_count = pair_count;
        tbl[idx].top_tok = tok;
    }
    return 0;
}

static inline uint64_t token_context_hash(const OnlineNgramState *st) {
    uint64_t h = 0ULL;
    if (st->token_ctx_len <= 0) {
        return h;
    }
    for (int j = 0; j < st->token_ctx_len; ++j) {
        const int ring_idx = (st->token_head + j) % st->token_ctx_len;
        h ^= ((uint64_t)st->token_ring[ring_idx]) * ROLLING_COEFFS[(size_t)j];
    }
    return h;
}

static inline void token_push(OnlineNgramState *st, uint16_t tok) {
    if (st->token_ctx_len <= 0) {
        return;
    }
    if (st->token_prefix_len < st->token_ctx_len) {
        st->token_ring[st->token_prefix_len] = tok;
        st->token_prefix_len += 1;
        return;
    }
    st->token_ring[st->token_head] = tok;
    st->token_head = (st->token_head + 1) % st->token_ctx_len;
}

static void *xcalloc(size_t count, size_t size) {
    if (count == 0 || size == 0) {
        return NULL;
    }
    return calloc(count, size);
}

static int alloc_tables(
    size_t table_bits,
    CtxBucket **ctx_tbl,
    uint8_t **ctx_used,
    size_t *ctx_mask,
    PairBucket **pair_tbl,
    uint8_t **pair_used,
    size_t *pair_mask
) {
    const size_t size = 1ULL << table_bits;
    *ctx_tbl = (CtxBucket *)xcalloc(size, sizeof(CtxBucket));
    *ctx_used = (uint8_t *)xcalloc(size, sizeof(uint8_t));
    *pair_tbl = (PairBucket *)xcalloc(size, sizeof(PairBucket));
    *pair_used = (uint8_t *)xcalloc(size, sizeof(uint8_t));
    if (!*ctx_tbl || !*ctx_used || !*pair_tbl || !*pair_used) {
        return -1;
    }
    *ctx_mask = size - 1U;
    *pair_mask = size - 1U;
    return 0;
}

void *online_ngram_state_create(
    int token_ctx_len,
    int token_table_bits,
    int within_table_bits
) {
    if (token_ctx_len < 0 || token_table_bits <= 0 || within_table_bits <= 0) {
        return NULL;
    }
    OnlineNgramState *st = (OnlineNgramState *)calloc(1, sizeof(OnlineNgramState));
    if (!st) {
        return NULL;
    }
    st->token_ctx_len = token_ctx_len;
    if (token_ctx_len > 0) {
        st->token_ring = (uint16_t *)xcalloc((size_t)token_ctx_len, sizeof(uint16_t));
        if (!st->token_ring) {
            free(st);
            return NULL;
        }
    }
    if (alloc_tables(
            (size_t)token_table_bits,
            &st->token_ctx_tbl,
            &st->token_ctx_used,
            &st->token_ctx_mask,
            &st->token_pair_tbl,
            &st->token_pair_used,
            &st->token_pair_mask
        ) != 0) {
        free(st->token_ring);
        free(st);
        return NULL;
    }
    if (alloc_tables(
            (size_t)within_table_bits,
            &st->within_ctx_tbl,
            &st->within_ctx_used,
            &st->within_ctx_mask,
            &st->within_pair_tbl,
            &st->within_pair_used,
            &st->within_pair_mask
        ) != 0) {
        free(st->token_pair_used);
        free(st->token_pair_tbl);
        free(st->token_ctx_used);
        free(st->token_ctx_tbl);
        free(st->token_ring);
        free(st);
        return NULL;
    }
    return (void *)st;
}

void online_ngram_state_destroy(void *ptr) {
    OnlineNgramState *st = (OnlineNgramState *)ptr;
    if (!st) {
        return;
    }
    free(st->within_pair_used);
    free(st->within_pair_tbl);
    free(st->within_ctx_used);
    free(st->within_ctx_tbl);
    free(st->token_pair_used);
    free(st->token_pair_tbl);
    free(st->token_ctx_used);
    free(st->token_ctx_tbl);
    free(st->token_ring);
    free(st);
}

void online_ngram_state_seed_prefix_token(void *ptr, uint16_t tok) {
    OnlineNgramState *st = (OnlineNgramState *)ptr;
    if (!st) {
        return;
    }
    token_push(st, tok);
}

int online_ngram_state_process_chunk(
    void *ptr,
    const uint16_t *tokens,
    int64_t n_tokens,
    const uint8_t *starts_new_word_lut,
    const uint8_t *boundary_lut,
    uint16_t *token_top_token,
    float *token_top_prob,
    uint16_t *within_top_token,
    float *within_top_prob,
    uint8_t *within_valid
) {
    OnlineNgramState *st = (OnlineNgramState *)ptr;
    if (!st || !tokens || n_tokens < 0) {
        return -1;
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        const uint16_t tok = tokens[i];
        const uint8_t is_boundary = boundary_lut[tok];
        const uint8_t is_new_word = starts_new_word_lut[tok];

        uint64_t token_ctx_key = 0ULL;
        if (st->token_ctx_len == 0 || st->token_prefix_len >= st->token_ctx_len) {
            token_ctx_key = token_context_hash(st);
            int found = 0;
            size_t idx = find_ctx_slot(
                st->token_ctx_tbl,
                st->token_ctx_used,
                st->token_ctx_mask,
                token_ctx_key,
                &found
            );
            if (found > 0) {
                token_top_token[i] = st->token_ctx_tbl[idx].top_tok;
                token_top_prob[i] =
                    (float)st->token_ctx_tbl[idx].top_count / (float)st->token_ctx_tbl[idx].total;
            } else {
                token_top_token[i] = 0U;
                token_top_prob[i] = 0.0f;
            }
        } else {
            token_top_token[i] = 0U;
            token_top_prob[i] = 0.0f;
        }

        uint64_t within_ctx_key = 0ULL;
        if (!is_boundary && !is_new_word && st->within_len > 0U) {
            within_ctx_key = st->within_hash ^ ((uint64_t)st->within_len * LEN_MIX);
            int found = 0;
            size_t idx = find_ctx_slot(
                st->within_ctx_tbl,
                st->within_ctx_used,
                st->within_ctx_mask,
                within_ctx_key,
                &found
            );
            within_valid[i] = 1U;
            if (found > 0) {
                within_top_token[i] = st->within_ctx_tbl[idx].top_tok;
                within_top_prob[i] =
                    (float)st->within_ctx_tbl[idx].top_count / (float)st->within_ctx_tbl[idx].total;
            } else {
                within_top_token[i] = 0U;
                within_top_prob[i] = 0.0f;
            }
        } else {
            within_valid[i] = 0U;
            within_top_token[i] = 0U;
            within_top_prob[i] = 0.0f;
        }

        if (st->token_ctx_len == 0 || st->token_prefix_len >= st->token_ctx_len) {
            const uint64_t pair_key = token_pair_key(token_ctx_key, tok, st->token_ctx_len);
            const uint32_t pair_count = pair_increment(
                st->token_pair_tbl,
                st->token_pair_used,
                st->token_pair_mask,
                pair_key
            );
            if (pair_count == 0U) {
                return -2;
            }
            if (ctx_increment(
                    st->token_ctx_tbl,
                    st->token_ctx_used,
                    st->token_ctx_mask,
                    token_ctx_key,
                    tok,
                    pair_count
                ) != 0) {
                return -3;
            }
        }
        token_push(st, tok);

        if (is_boundary) {
            st->within_hash = 0ULL;
            st->within_len = 0U;
            continue;
        }
        if (is_new_word || st->within_len == 0U) {
            st->within_hash = extend_prefix_hash(0ULL, tok, 0U);
            st->within_len = 1U;
            continue;
        }
        const uint32_t within_pair_count = pair_increment(
            st->within_pair_tbl,
            st->within_pair_used,
            st->within_pair_mask,
            within_pair_key(within_ctx_key, tok)
        );
        if (within_pair_count == 0U) {
            return -4;
        }
        if (ctx_increment(
                st->within_ctx_tbl,
                st->within_ctx_used,
                st->within_ctx_mask,
                within_ctx_key,
                tok,
                within_pair_count
            ) != 0) {
            return -5;
        }
        st->within_hash = extend_prefix_hash(st->within_hash, tok, st->within_len);
        st->within_len += 1U;
    }
    return 0;
}

int online_ngram_state_process_chunk_token_only(
    void *ptr,
    const uint16_t *tokens,
    int64_t n_tokens,
    uint16_t *token_top_token,
    float *token_top_prob
) {
    OnlineNgramState *st = (OnlineNgramState *)ptr;
    if (!st || !tokens || n_tokens < 0) {
        return -1;
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        const uint16_t tok = tokens[i];

        uint64_t token_ctx_key = 0ULL;
        if (st->token_ctx_len == 0 || st->token_prefix_len >= st->token_ctx_len) {
            token_ctx_key = token_context_hash(st);
            int found = 0;
            size_t idx = find_ctx_slot(
                st->token_ctx_tbl,
                st->token_ctx_used,
                st->token_ctx_mask,
                token_ctx_key,
                &found
            );
            if (found > 0) {
                token_top_token[i] = st->token_ctx_tbl[idx].top_tok;
                token_top_prob[i] =
                    (float)st->token_ctx_tbl[idx].top_count / (float)st->token_ctx_tbl[idx].total;
            } else {
                token_top_token[i] = 0U;
                token_top_prob[i] = 0.0f;
            }

            const uint64_t pair_key = token_pair_key(token_ctx_key, tok, st->token_ctx_len);
            const uint32_t pair_count = pair_increment(
                st->token_pair_tbl,
                st->token_pair_used,
                st->token_pair_mask,
                pair_key
            );
            if (pair_count == 0U) {
                return -2;
            }
            if (ctx_increment(
                    st->token_ctx_tbl,
                    st->token_ctx_used,
                    st->token_ctx_mask,
                    token_ctx_key,
                    tok,
                    pair_count
                ) != 0) {
                return -3;
            }
        } else {
            token_top_token[i] = 0U;
            token_top_prob[i] = 0.0f;
        }
        token_push(st, tok);
    }
    return 0;
}

"""
_ONLINE_NGRAM_LIB_CACHE = None

WHITESPACE_BYTE_IDS = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36}
EDGE_PUNCT = ".,:;!?()[]{}<>\"'`"


def normalize_word(text: str, mode: str) -> str:
    text = text.strip()
    if mode == "lower":
        return text.lower()
    if mode == "identity":
        return text
    if mode == "strip_punct_lower":
        return text.strip(EDGE_PUNCT).lower()
    raise ValueError(f"Unknown word normalization mode: {mode}")


def suggest_table_bits(expected_entries: int, load_factor: float) -> int:
    if expected_entries <= 0:
        return 16
    target = max(int(expected_entries / max(load_factor, 1e-6)), 1)
    bits = max(int(math.ceil(math.log2(target))), 12)
    return min(bits, 28)


def ensure_online_ngram_lib(log0=print) -> ctypes.CDLL:
    global _ONLINE_NGRAM_LIB_CACHE
    if _ONLINE_NGRAM_LIB_CACHE is not None:
        return _ONLINE_NGRAM_LIB_CACHE
    build_dir = Path(os.environ.get("PG_NGRAM_BUILD_DIR", "/tmp"))
    build_dir.mkdir(parents=True, exist_ok=True)
    lib_path = build_dir / f"pg_online_ngram_state_{os.getpid()}.so"
    if not lib_path.exists():
        tmp_lib = lib_path.with_suffix(f".tmp.{os.getpid()}.so")
        log0("ngram_tilt:building_embedded_native_helper")
        subprocess.run(
            [
                "gcc", "-O3", "-march=native", "-std=c11", "-shared", "-fPIC",
                "-x", "c", "-", "-o", str(tmp_lib),
            ],
            input=_ONLINE_NGRAM_STATE_C.encode("utf-8"),
            check=True,
        )
        os.replace(tmp_lib, lib_path)
    lib = ctypes.CDLL(str(lib_path))
    lib.online_ngram_state_create.restype = ctypes.c_void_p
    lib.online_ngram_state_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.online_ngram_state_destroy.restype = None
    lib.online_ngram_state_destroy.argtypes = [ctypes.c_void_p]
    lib.online_ngram_state_seed_prefix_token.restype = None
    lib.online_ngram_state_seed_prefix_token.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    lib.online_ngram_state_process_chunk.restype = ctypes.c_int
    lib.online_ngram_state_process_chunk.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.online_ngram_state_process_chunk_token_only.restype = ctypes.c_int
    lib.online_ngram_state_process_chunk_token_only.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
    ]
    _ONLINE_NGRAM_LIB_CACHE = lib
    return lib

class OnlineNgramState:
    def __init__(
        self, *, lib, token_ctx_len, token_table_bits, within_table_bits,
        starts_new_word_lut, boundary_lut, seed_prefix_token,
    ):
        self.lib = lib
        self.state = lib.online_ngram_state_create(token_ctx_len, token_table_bits, within_table_bits)
        if not self.state:
            raise RuntimeError(
                f"Native ngram state alloc failed token_table_bits={token_table_bits} within_table_bits={within_table_bits}"
            )
        self.starts_new_word_lut = np.ascontiguousarray(starts_new_word_lut.astype(np.uint8, copy=False))
        self.boundary_lut = np.ascontiguousarray(boundary_lut.astype(np.uint8, copy=False))
        self.lib.online_ngram_state_seed_prefix_token(self.state, ctypes.c_uint16(int(seed_prefix_token)))

    def close(self):
        if self.state:
            self.lib.online_ngram_state_destroy(self.state)
            self.state = None

    def __del__(self):
        self.close()

    def process_chunk(self, chunk_tokens):
        chunk_tokens = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(chunk_tokens.size)
        token_top_token = np.zeros(n, dtype=np.uint16)
        token_top_prob = np.zeros(n, dtype=np.float32)
        within_top_token = np.zeros(n, dtype=np.uint16)
        within_top_prob = np.zeros(n, dtype=np.float32)
        within_valid = np.zeros(n, dtype=np.uint8)
        rc = self.lib.online_ngram_state_process_chunk(
            self.state,
            chunk_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            self.starts_new_word_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.boundary_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            token_top_token.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            token_top_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            within_top_token.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            within_top_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            within_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if rc != 0:
            raise RuntimeError(f"Native ngram process_chunk failed rc={rc}")
        return token_top_token, token_top_prob, within_top_token, within_top_prob, within_valid.astype(bool)

    def process_chunk_token_only(self, chunk_tokens):
        chunk_tokens = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(chunk_tokens.size)
        token_top_token = np.zeros(n, dtype=np.uint16)
        token_top_prob = np.zeros(n, dtype=np.float32)
        rc = self.lib.online_ngram_state_process_chunk_token_only(
            self.state,
            chunk_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            token_top_token.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            token_top_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"Native ngram token-only process_chunk failed rc={rc}")
        return token_top_token, token_top_prob


class WordStartState:
    def __init__(self, *, sp, order, normalize_mode):
        self.sp = sp
        self.ctx_w = max(order - 1, 0)
        self.normalize_mode = normalize_mode
        self.prev_word_ids = collections.deque(maxlen=self.ctx_w)
        self.current_word_tokens: list = []
        self.word_to_id: dict = {}
        self.next_word_id = 1
        self.ctx_total: dict = {}
        self.pair_count: dict = {}
        self.ctx_best_token: dict = {}
        self.ctx_best_count: dict = {}

    def _flush_current_word(self):
        if not self.current_word_tokens:
            return
        text = normalize_word(self.sp.decode(self.current_word_tokens), self.normalize_mode)
        if text:
            wid = self.word_to_id.get(text)
            if wid is None:
                wid = self.next_word_id
                self.word_to_id[text] = wid
                self.next_word_id += 1
            if self.ctx_w > 0:
                self.prev_word_ids.append(wid)
        self.current_word_tokens = []

    def process_chunk(self, chunk_tokens, *, starts_new_word_lut, boundary_lut):
        chunk_tokens = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        top_token = np.zeros(chunk_tokens.size, dtype=np.uint16)
        top_prob = np.zeros(chunk_tokens.size, dtype=np.float32)
        for i, tok_u16 in enumerate(chunk_tokens):
            tok = int(tok_u16)
            is_boundary = bool(boundary_lut[tok])
            is_word_start = bool(starts_new_word_lut[tok]) or not self.current_word_tokens
            if is_boundary:
                self._flush_current_word()
                continue
            if bool(starts_new_word_lut[tok]):
                self._flush_current_word()
            ctx_key = None
            if is_word_start and len(self.prev_word_ids) >= self.ctx_w:
                ctx_key = tuple(self.prev_word_ids) if self.ctx_w > 0 else ()
                total = self.ctx_total.get(ctx_key, 0)
                if total > 0:
                    top_token[i] = np.uint16(self.ctx_best_token[ctx_key])
                    top_prob[i] = np.float32(self.ctx_best_count[ctx_key] / total)
            if is_word_start:
                if ctx_key is not None:
                    pair_key = (ctx_key, tok)
                    pair = self.pair_count.get(pair_key, 0) + 1
                    self.pair_count[pair_key] = pair
                    total = self.ctx_total.get(ctx_key, 0) + 1
                    self.ctx_total[ctx_key] = total
                    best_count = self.ctx_best_count.get(ctx_key, 0)
                    if pair > best_count:
                        self.ctx_best_count[ctx_key] = pair
                        self.ctx_best_token[ctx_key] = tok
                self.current_word_tokens = [tok]
            else:
                self.current_word_tokens.append(tok)
        return top_token, top_prob


def build_piece_luts(*, tokenizer_path, vocab_size):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    pieces = [sp.id_to_piece(i) for i in range(sp.vocab_size())]
    starts_new_word_lut = np.zeros(vocab_size, dtype=np.uint8)
    for i, piece in enumerate(pieces):
        starts_new_word_lut[i] = 1 if piece.startswith("▁") else 0
    boundary_lut = np.zeros(vocab_size, dtype=np.uint8)
    bos_id = sp.bos_id()
    if bos_id >= 0 and bos_id < vocab_size:
        boundary_lut[bos_id] = 1
    for tok in range(min(sp.vocab_size(), vocab_size)):
        if sp.is_byte(tok) and tok in WHITESPACE_BYTE_IDS:
            boundary_lut[tok] = 1
    return sp, starts_new_word_lut, boundary_lut


def build_hints_for_targets(
    *, target_token_ids_np, tokenizer_path, vocab_size, log0=print,
    token_order=16, token_threshold=0.800, token_boost=2.625,
    within_tau=999.0, within_boost=0.0,
    word_order=4, word_normalize="strip_punct_lower",
    word_tau=999.0, word_boost=0.0,
    agree_add_boost=0.0,
    seed_prefix_token=None,
):
    """Single L->R pass. Returns dict with hint_ids, gate_mask, boost_per_pos.

    target_token_ids_np: np.uint16 array of realized targets (length = total_targets).
    Output arrays are aligned to target_token_ids_np indexing.

    For each scored position t we pick at most one hint h_t:
      - prefer the expert with highest expected gain = p_top * boost - log1p(p_top * (exp(boost)-1))
      - if multiple experts agree on the same h_t, additive boost agree_add_boost
      - gate (don't tilt) when no expert clears its threshold

    The realized loss formula used by the caller:
      ptl' = ptl - beta * 1[y == h_t] + log1p(q_t * (exp(beta) - 1))   when gate_mask == True
      ptl' = ptl                                                        when gate_mask == False
    """
    sp, starts_new_word_lut, boundary_lut = build_piece_luts(
        tokenizer_path=tokenizer_path, vocab_size=vocab_size
    )
    total = int(target_token_ids_np.size)
    if total == 0:
        return {
            "hint_ids":   np.zeros(0, dtype=np.int64),
            "gate_mask":  np.zeros(0, dtype=bool),
            "boost":      np.zeros(0, dtype=np.float32),
            "sp":         sp,
            "starts_new_word_lut": starts_new_word_lut,
            "boundary_lut": boundary_lut,
        }

    token_table_bits = suggest_table_bits(total, load_factor=0.55)
    within_table_bits = suggest_table_bits(max(total // 2, 1), load_factor=0.60)
    token_only = (
        float(within_boost) == 0.0
        and float(word_boost) == 0.0
    )
    online_lib = ensure_online_ngram_lib(log0)
    ngram_state = OnlineNgramState(
        lib=online_lib,
        token_ctx_len=max(token_order - 1, 0),
        token_table_bits=token_table_bits,
        within_table_bits=1 if token_only else within_table_bits,
        starts_new_word_lut=starts_new_word_lut,
        boundary_lut=boundary_lut,
        seed_prefix_token=int(target_token_ids_np[0] if seed_prefix_token is None else seed_prefix_token),
    )
    if token_only:
        token_top_tok, token_top_prob = ngram_state.process_chunk_token_only(target_token_ids_np)
        token_gate = token_top_prob >= np.float32(token_threshold)
        hint_ids = np.where(token_gate, token_top_tok.astype(np.int64), 0).astype(np.int64)
        boost = np.where(token_gate, np.float32(token_boost), np.float32(0.0)).astype(np.float32)
        log0(
            f"ngram_tilt:hints total={total} gated={int(token_gate.sum())} "
            f"token_gate={int(token_gate.sum())} within_gate=0 word_gate=0 agree2plus=0"
        )
        return {
            "hint_ids":   hint_ids,
            "gate_mask":  token_gate,
            "boost":      boost,
            "sp":         sp,
            "starts_new_word_lut": starts_new_word_lut,
            "boundary_lut": boundary_lut,
        }
    word_state = WordStartState(sp=sp, order=word_order, normalize_mode=word_normalize)

    token_top_tok, token_top_prob, within_top_tok, within_top_prob, within_valid = (
        ngram_state.process_chunk(target_token_ids_np)
    )
    word_top_tok, word_top_prob = word_state.process_chunk(
        target_token_ids_np,
        starts_new_word_lut=starts_new_word_lut,
        boundary_lut=boundary_lut,
    )

    def _expected_gain(p_top, boost):
        # E[ -log p'(y) under -log p(y)] when y ~ p
        # = p_top * boost - log1p(p_top * (exp(boost) - 1))
        # Maximizing this over experts => pick the most informative hint.
        log_norm = np.log1p(p_top * (math.exp(boost) - 1.0))
        return p_top * boost - log_norm

    token_gate = token_top_prob >= np.float32(token_threshold)
    within_gate = within_valid & (within_top_prob >= np.float32(within_tau))
    word_gate = word_top_prob >= np.float32(word_tau)

    token_gain = np.where(token_gate, _expected_gain(token_top_prob.astype(np.float64), token_boost), -np.inf)
    within_gain = np.where(within_gate, _expected_gain(within_top_prob.astype(np.float64), within_boost), -np.inf)
    word_gain = np.where(word_gate, _expected_gain(word_top_prob.astype(np.float64), word_boost), -np.inf)

    stack = np.stack([token_gain, within_gain, word_gain], axis=1)
    best_idx = np.argmax(stack, axis=1)
    best_gain = np.max(stack, axis=1)
    any_gate = best_gain > -np.inf

    hint_ids = np.zeros(total, dtype=np.int64)
    boost = np.zeros(total, dtype=np.float32)
    base_boost_per_expert = np.array([token_boost, within_boost, word_boost], dtype=np.float32)
    hint_per_expert = np.stack([
        token_top_tok.astype(np.int64),
        within_top_tok.astype(np.int64),
        word_top_tok.astype(np.int64),
    ], axis=1)

    rows = np.arange(total)
    hint_ids[any_gate] = hint_per_expert[rows[any_gate], best_idx[any_gate]]
    boost[any_gate] = base_boost_per_expert[best_idx[any_gate]]

    # Agreement bonus: if 2+ experts agree on the same hint as best, add agree_add_boost
    gate_mask_each = np.stack([token_gate, within_gate, word_gate], axis=1)
    expert_hints = hint_per_expert.copy()
    expert_hints[~gate_mask_each] = -1
    agreements = (expert_hints == hint_ids[:, None]).sum(axis=1)
    agreement_extra = np.where(agreements >= 2, np.float32(agree_add_boost), np.float32(0.0))
    boost = (boost + agreement_extra).astype(np.float32)

    log0(
        f"ngram_tilt:hints total={total} gated={int(any_gate.sum())} "
        f"token_gate={int(token_gate.sum())} within_gate={int(within_gate.sum())} word_gate={int(word_gate.sum())} "
        f"agree2plus={int((agreements >= 2).sum())}"
    )

    return {
        "hint_ids":   hint_ids,
        "gate_mask":  any_gate,
        "boost":      boost,
        "sp":         sp,
        "starts_new_word_lut": starts_new_word_lut,
        "boundary_lut": boundary_lut,
    }


def apply_tilt_to_ptl_torch(
    ptl: torch.Tensor,
    log_q_hint: torch.Tensor,
    target_ids: torch.Tensor,
    hint_ids: torch.Tensor,
    gate_mask: torch.Tensor,
    boost: torch.Tensor,
):
    """Closed-form tilt applied to per-token NLL.

    All tensors same shape [..., L].
        ptl_tilted = ptl - beta * 1[y == h] + log1p(q * (exp(beta) - 1))   if gate else ptl
    """
    boost64 = boost.to(torch.float64)
    q = log_q_hint.to(torch.float64).clamp_(max=0.0).exp()
    is_hit = (target_ids == hint_ids).to(torch.float64)
    log_Z = torch.log1p(q * (torch.expm1(boost64)))
    ptl_tilted = ptl.to(torch.float64) - boost64 * is_hit + log_Z
    return torch.where(gate_mask, ptl_tilted, ptl.to(torch.float64)).to(ptl.dtype)


def apply_tilt_to_ptl_torch_fast(
    ptl: torch.Tensor,
    log_q_hint: torch.Tensor,
    target_ids: torch.Tensor,
    hint_ids: torch.Tensor,
    gate_mask: torch.Tensor,
    boost: torch.Tensor,
):
    """fp32 variant of apply_tilt — cast removed where safe.

    BPB downstream accumulator is fp64, so per-token tilt computation in
    fp32 has no impact on final precision. Saves ~10-15s per eval pass on
    H100 (avoids fp64 ALU + double memory traffic).
    """
    boost32 = boost.to(torch.float32)
    q = log_q_hint.to(torch.float32).clamp_(max=0.0).exp()
    is_hit = (target_ids == hint_ids).to(torch.float32)
    log_Z = torch.log1p(q * (torch.expm1(boost32)))
    ptl_f32 = ptl.to(torch.float32)
    ptl_tilted = ptl_f32 - boost32 * is_hit + log_Z
    return torch.where(gate_mask, ptl_tilted, ptl_f32).to(ptl.dtype)

# ===== End inlined online n-gram tilt helpers =====


# ===== Fused softcapped cross-entropy (Triton) — training-only path =====
# Replaces the eager
#     logits_softcap = softcap * tanh(logits / softcap)
#     F.cross_entropy(logits_softcap.float(), targets, reduction="mean")
# sequence with a single fused kernel that reads logits_proj once, applies
# softcap in-register, and computes (LSE, loss) in one streaming pass. The
# backward kernel mirrors the forward so there's no stored softcapped logits.
# Numerically identical to the eager path up to fp32 accumulation differences.
_FUSED_CE_LIBRARY = "pgsubmission1draft7fusedce"
_FUSED_CE_BLOCK_SIZE = 1024
_FUSED_CE_NUM_WARPS = 4


@triton.jit
def _softcapped_ce_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, softcap,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    max_val = -float("inf")
    sum_exp = 0.0
    A = 2.0 * softcap
    inv_C = 2.0 / softcap
    for off in range(0, n_cols, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n_cols
        val = tl.load(
            logits_row_ptr + cols * stride_logits_v,
            mask=mask, other=-float("inf"),
        ).to(tl.float32)
        z = A * tl.sigmoid(val * inv_C)
        z = tl.where(mask, z, -float("inf"))
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max
    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row_idx, lse)
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    target_val = tl.load(logits_row_ptr + target * stride_logits_v).to(tl.float32)
    target_z = A * tl.sigmoid(target_val * inv_C)
    tl.store(losses_ptr + row_idx, lse - target_z)


@triton.jit
def _softcapped_ce_bwd_kernel(
    grad_logits_ptr, grad_losses_ptr, lse_ptr, logits_ptr, targets_ptr,
    stride_logits_n, stride_logits_v,
    stride_grad_n, stride_grad_v,
    n_rows, n_cols, softcap,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_logits_ptr + row_idx * stride_grad_n
    lse = tl.load(lse_ptr + row_idx)
    grad_loss = tl.load(grad_losses_ptr + row_idx).to(tl.float32)
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    A = 2.0 * softcap
    inv_C = 2.0 / softcap
    dz_dx_scale = A * inv_C
    for off in range(0, n_cols, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n_cols
        val = tl.load(
            logits_row_ptr + cols * stride_logits_v,
            mask=mask, other=0.0,
        ).to(tl.float32)
        sigmoid_u = tl.sigmoid(val * inv_C)
        z = A * sigmoid_u
        probs = tl.exp(z - lse)
        grad_z = grad_loss * (probs - tl.where(cols == target, 1.0, 0.0))
        grad_x = grad_z * (dz_dx_scale * sigmoid_u * (1.0 - sigmoid_u))
        tl.store(grad_row_ptr + cols * stride_grad_v, grad_x, mask=mask)


def _validate_softcapped_ce_inputs(
    logits: Tensor, targets: Tensor, softcap: float,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits.ndim=2, got {logits.ndim}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets.ndim=1, got {targets.ndim}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Expected matching rows, got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    if not logits.is_cuda or not targets.is_cuda:
        raise ValueError("softcapped_cross_entropy requires CUDA tensors")
    if softcap <= 0.0:
        raise ValueError(f"softcap must be positive, got {softcap}")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    logits = logits.contiguous()
    targets = targets.contiguous()
    if targets.dtype != torch.int64:
        targets = targets.to(dtype=torch.int64)
    return logits, targets


@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce", mutates_args=())
def softcapped_ce_op(logits: Tensor, targets: Tensor, softcap: float) -> tuple[Tensor, Tensor]:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    n_rows, n_cols = logits.shape
    losses = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    lse = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    _softcapped_ce_fwd_kernel[(n_rows,)](
        logits, losses, lse, targets,
        logits.stride(0), logits.stride(1),
        n_rows, n_cols, float(softcap),
        block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
    )
    return losses, lse


@softcapped_ce_op.register_fake
def _(logits: Tensor, targets: Tensor, softcap: float):
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("softcapped_ce fake impl expects 2D logits and 1D targets")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Expected matching rows, got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    n_rows = logits.shape[0]
    return (
        logits.new_empty((n_rows,), dtype=torch.float32),
        logits.new_empty((n_rows,), dtype=torch.float32),
    )


@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce_backward", mutates_args=())
def softcapped_ce_backward_op(
    logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float,
) -> Tensor:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    lse = lse.contiguous()
    grad_losses = grad_losses.contiguous().to(dtype=torch.float32)
    if lse.ndim != 1 or grad_losses.ndim != 1:
        raise ValueError("Expected 1D lse and grad_losses")
    if lse.shape[0] != logits.shape[0] or grad_losses.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Expected row-aligned lse/grad_losses, got logits={tuple(logits.shape)} "
            f"lse={tuple(lse.shape)} grad_losses={tuple(grad_losses.shape)}"
        )
    grad_logits = torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    _softcapped_ce_bwd_kernel[(n_rows,)](
        grad_logits, grad_losses, lse, logits, targets,
        logits.stride(0), logits.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        n_rows, n_cols, float(softcap),
        block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
    )
    return grad_logits


@softcapped_ce_backward_op.register_fake
def _(logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float):
    if logits.ndim != 2 or targets.ndim != 1 or lse.ndim != 1 or grad_losses.ndim != 1:
        raise ValueError("softcapped_ce_backward fake impl expects 2D logits and 1D row tensors")
    if (
        logits.shape[0] != targets.shape[0]
        or logits.shape[0] != lse.shape[0]
        or logits.shape[0] != grad_losses.shape[0]
    ):
        raise ValueError("softcapped_ce_backward fake impl expects row-aligned tensors")
    return logits.new_empty(logits.shape)


def _softcapped_ce_setup_context(
    ctx: torch.autograd.function.FunctionCtx, inputs, output,
) -> None:
    logits, targets, softcap = inputs
    _losses, lse = output
    ctx.save_for_backward(logits, targets, lse)
    ctx.softcap = float(softcap)


def _softcapped_ce_backward(
    ctx: torch.autograd.function.FunctionCtx, grad_losses: Tensor, grad_lse: "Tensor | None",
):
    del grad_lse
    logits, targets, lse = ctx.saved_tensors
    grad_logits = torch.ops.pgsubmission1draft7fusedce.softcapped_ce_backward(
        logits, targets, lse, grad_losses, ctx.softcap
    )
    return grad_logits, None, None


softcapped_ce_op.register_autograd(
    _softcapped_ce_backward, setup_context=_softcapped_ce_setup_context,
)


def softcapped_cross_entropy(
    logits: Tensor, targets: Tensor, softcap: float, reduction: str = "mean",
) -> Tensor:
    losses, _lse = torch.ops.pgsubmission1draft7fusedce.softcapped_ce(
        logits, targets, float(softcap)
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction={reduction!r}")


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.75))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    # Fused softcapped CE (Triton). Training-only — forward_logits eval path still uses
    # eager softcap+F.cross_entropy. Default ON since validated as at-worst neutral.
    fused_ce_enabled = bool(int(os.environ.get("FUSED_CE_ENABLED", "1")))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 40000))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 3e1))
    rope_base = float(os.environ.get("ROPE_BASE", 1e4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    rope_yarn = bool(int(os.environ.get("ROPE_YARN", "0")))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", 8))
    parallel_final_lane = os.environ.get("PARALLEL_FINAL_LANE", "mean")
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.026))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.97))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 96))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.0001))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 48))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 2048))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))
    ttt_grad_steps = int(os.environ.get("TTT_GRAD_STEPS", 1))
    # V19: PR #1886 (renqianluo) + sunnypatneedi research log 2026-04-28 found that
    # the Triton fused-CE kernel's fp32-accumulation interacts with warm-start LoRA-A
    # to destabilize seeds 314/1337 at TTT_WEIGHT_DECAY=1.0. Raising the default to
    # 2.0 prevents seed collapse without measurably moving stable seeds.
    ttt_weight_decay = float(os.environ.get("TTT_WEIGHT_DECAY", 2.0))
    ttt_beta1 = float(os.environ.get("TTT_BETA1", 0))
    ttt_beta2 = float(os.environ.get("TTT_BETA2", 0.999))
    ttt_k_lora = bool(int(os.environ.get("TTT_K_LORA", "1")))
    ttt_mlp_lora = bool(int(os.environ.get("TTT_MLP_LORA", "1")))
    ttt_o_lora = bool(int(os.environ.get("TTT_O_LORA", "1")))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adam")
    ttt_eval_batches = os.environ.get("TTT_EVAL_BATCHES", "")
    val_doc_fraction = float(os.environ.get("VAL_DOC_FRACTION", 1.0))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 16))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 4.0))
    phased_ttt_prefix_docs = int(os.environ.get("PHASED_TTT_PREFIX_DOCS", 2000))
    phased_ttt_num_phases = int(os.environ.get("PHASED_TTT_NUM_PHASES", 1))
    global_ttt_lr = float(os.environ.get("GLOBAL_TTT_LR", 0.001))
    global_ttt_momentum = float(os.environ.get("GLOBAL_TTT_MOMENTUM", 0.9))
    global_ttt_epochs = int(os.environ.get("GLOBAL_TTT_EPOCHS", 1))
    global_ttt_chunk_tokens = int(os.environ.get("GLOBAL_TTT_CHUNK_TOKENS", 32768))
    global_ttt_batch_seqs = int(os.environ.get("GLOBAL_TTT_BATCH_SEQS", 32))
    global_ttt_warmup_start_lr = float(os.environ.get("GLOBAL_TTT_WARMUP_START_LR", 0.0))
    global_ttt_warmup_chunks = int(os.environ.get("GLOBAL_TTT_WARMUP_CHUNKS", 0))
    global_ttt_grad_clip = float(os.environ.get("GLOBAL_TTT_GRAD_CLIP", 1.0))
    global_ttt_respect_doc_boundaries = bool(int(os.environ.get("GLOBAL_TTT_RESPECT_DOC_BOUNDARIES", "1")))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 2e1))
    mlp_clip_sigmas = float(os.environ.get("MLP_CLIP_SIGMAS", 10.0))
    attn_clip_sigmas = float(os.environ.get("ATTN_CLIP_SIGMAS", 13.0))
    # AttnOutGate (per-head multiplicative output gate, PR #1667 MarioPaerle).
    # Zero-init weight: 2*sigmoid(0)=1 -> transparent at start. Source defaults to
    # block input x ('proj'); 'q' uses raw Q projection output.
    attn_out_gate_enabled = bool(int(os.environ.get("ATTN_OUT_GATE_ENABLED", "0")))
    attn_out_gate_src = os.environ.get("ATTN_OUT_GATE_SRC", "proj")
    # SmearGate (input-dependent forward-1 token smear, modded-nanogpt @classiclarryd
    # via PR #1667). x_t <- x_t + lam * sigmoid(W*x_t[:gate_window]) * x_{t-1}.
    # lam=0 + W=0 -> transparent at init.
    smear_gate_enabled = bool(int(os.environ.get("SMEAR_GATE_ENABLED", "0")))
    # Window: first GATE_WINDOW dims of the source feed the gate projection.
    gate_window = int(os.environ.get("GATE_WINDOW", 12))
    # Gated Attention (Qwen, NeurIPS 2025 Best Paper, arXiv:2505.06708;
    # qiuzh20/gated_attention). Per-head sigmoid gate on SDPA output, BEFORE
    # out_proj. Gate input = full block input x (paper's headwise G1 variant
    # driven from hidden_states). W_g shape (num_heads, dim), plain sigmoid.
    # Near-zero init gives g~0.5 at step 0 (half attention output); per-block
    # attn_scale (init 1.0) compensates during training. Name contains
    # "attn_gate" so CONTROL_TENSOR_NAME_PATTERNS routes it to scalar AdamW.
    gated_attn_enabled = bool(int(os.environ.get("GATED_ATTN_ENABLED", "0")))
    gated_attn_init_std = float(os.environ.get("GATED_ATTN_INIT_STD", 0.01))
    # Dedicated int8-per-row quantization for `attn_gate_w` tensors. These are
    # small ((num_heads, dim) = (8, 512) = 4096 params) and bypass GPTQ via the
    # numel<=65536 passthrough branch -> stored as fp16 (8 KB/layer, ~65 KB total
    # compressed). int8-per-row cuts the raw tensor in half with negligible BPB
    # impact: scales per head (8 values), symmetric quant over [-127, 127].
    # No Hessian needed (gate weights not in collect_hessians()).
    gated_attn_quant_gate = bool(int(os.environ.get("GATED_ATTN_QUANT_GATE", "0")))
    # Sparse Attention Gate (modded-nanogpt-style). Keeps dense SDPA and only
    # swaps the output-gate input to the first GATE_WINDOW residual dims.
    # W_g: (num_heads, gate_window) = (8, 12) = 96 params/layer (~44K total),
    # vs dense GatedAttn's (8, 512) = 4K/layer (~44K diff). Name "attn_gate_w"
    # is shared so quant routing and int8 gate passthrough Just Work. Gate
    # passthrough int8 still applies via GATED_ATTN_QUANT_GATE=1.
    # Mutually exclusive with ATTN_OUT_GATE_ENABLED and GATED_ATTN_ENABLED.
    sparse_attn_gate_enabled = bool(int(os.environ.get("SPARSE_ATTN_GATE_ENABLED", "0")))
    sparse_attn_gate_init_std = float(os.environ.get("SPARSE_ATTN_GATE_INIT_STD", 0.0))
    sparse_attn_gate_scale = float(os.environ.get("SPARSE_ATTN_GATE_SCALE", 1.0))
    # LQER asymmetric rank-k correction on top-K quant-error tensors (PR #1530 v2 port).
    # Computes SVD of E = W_fp - W_quant, packs top-r A,B as INT2/INT4 (asym) or INTk (sym).
    lqer_enabled = bool(int(os.environ.get("LQER_ENABLED", "1")))
    lqer_rank = int(os.environ.get("LQER_RANK", 4))
    # Default 1 (was 3): PR #1948 sweep showed top_k=1 is -0.00044 BPB
    # and saves ~50 KB which we need to keep the seed-max artifact under
    # 16,000,000 with the new XSA gate weights and Skylight optimizer state
    # (state is not exported, but Skylight slightly increases pre-quant BPB
    # variance across seeds, so we want artifact headroom).
    lqer_top_k = int(os.environ.get("LQER_TOP_K", 1))
    lqer_factor_bits = int(os.environ.get("LQER_FACTOR_BITS", 4))
    lqer_asym_enabled = bool(int(os.environ.get("LQER_ASYM_ENABLED", "1")))
    lqer_asym_group = int(os.environ.get("LQER_ASYM_GROUP", "64"))
    lqer_scope = os.environ.get("LQER_SCOPE", "all")
    lqer_gain_select = bool(int(os.environ.get("LQER_GAIN_SELECT", "0")))
    awq_lite_enabled = bool(int(os.environ.get("AWQ_LITE_ENABLED", "0")))
    awq_lite_bits = int(os.environ.get("AWQ_LITE_BITS", "8"))
    awq_lite_group_top_k = int(os.environ.get("AWQ_LITE_GROUP_TOP_K", "1"))
    awq_lite_group_size = int(os.environ.get("AWQ_LITE_GROUP_SIZE", "64"))
    # PR #1145 online n-gram tilt (AnirudhRahul, valerio-endorsed). Causal,
    # normalized, prefix-only token expert; closed-form multiplicative-boost-with-renorm
    # applied to per-token NLL. See inlined n-gram helpers above for math + compliance.
    ngram_tilt_enabled = bool(int(os.environ.get("NGRAM_TILT_ENABLED", "0")))
    token_order = int(os.environ.get("TOKEN_ORDER", "16"))
    token_threshold = float(os.environ.get("TOKEN_THRESHOLD", "0.800"))
    token_boost = float(os.environ.get("TOKEN_BOOST", "2.625"))
    within_tau = float(os.environ.get("WITHIN_TAU", "999.0"))
    within_boost = float(os.environ.get("WITHIN_BOOST", "0.0"))
    word_order = int(os.environ.get("WORD_ORDER", "4"))
    word_normalize = os.environ.get("WORD_NORMALIZE", "strip_punct_lower")
    word_tau = float(os.environ.get("WORD_TAU", "999.0"))
    word_boost = float(os.environ.get("WORD_BOOST", "0.0"))
    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.0"))
    # Optional legacy switch: move ngram hint precompute outside the measured eval
    # timer. The submitted configuration keeps this disabled so precompute is timed.
    ngram_hint_precompute_outside = bool(int(os.environ.get("NGRAM_HINT_PRECOMPUTE_OUTSIDE", "0")))
    # === Modded-NanoGPT transplants (PR-#1967 base + miracle stack) ===
    # GATED_XSA: per-(layer, head) tanh(alpha) gate around the existing XSA
    # subtraction. zero-init alpha -> tanh(0)=0 -> step-0 model is bit-identical
    # to the un-gated PR-#1967 baseline. Modded-nanogpt PR #264, p=0.0014.
    # alpha is 1-D (num_heads,) so the existing optimizer routing puts it in the
    # scalar-AdamW group automatically; per-tensor numel is small enough to land
    # in the fp16 passthrough quant bucket -> ~16 bytes/layer artifact cost.
    gated_xsa_enabled = bool(int(os.environ.get("GATED_XSA", "1")))
    # SKYLIGHT_MUON: per-row variance EMA + Frobenius re-normalization + u/w
    # floor on the post-NS Muon update. Modded-nanogpt PR #269 (Skylight-001),
    # 6/6 seeds, -250 steps to target. Pure optimizer change -> zero artifact
    # bytes. State buffer (row_var_ema, skylight_step) lives only in optimizer
    # state, never serialized.
    skylight_muon_enabled = bool(int(os.environ.get("SKYLIGHT_MUON", "1")))
    skylight_beta2 = float(os.environ.get("BETA2_NORMUON", "0.95"))
    skylight_uw_floor = float(os.environ.get("UW_FLOOR", "0.35"))
    # 2C: Temperature scaling on logits before softcap. Σ P=1 preserved.
    #     Default 1.0 = no-op. Tune on train holdout, apply at eval.
    temperature_scale = float(os.environ.get("TEMPERATURE_SCALE", "1.0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    # CaseOps integration: optional override of dataset root + tokenizer path.
    # When CASEOPS_ENABLED=1, the wrapper loads a per-token byte sidecar
    # (fineweb_val_bytes_*.bin, identical shard layout to val_*.bin) and uses
    # it as the canonical raw-byte budget for BPB accounting. The sidecar
    # REPLACES the build_sentencepiece_luts byte-counting path entirely.
    caseops_enabled = bool(int(os.environ.get("CASEOPS_ENABLED", "0")))
    _default_caseops_data = os.path.join(
        data_dir,
        "datasets",
        "fineweb10B_sp8192_caseops",
        "datasets",
        "datasets",
        "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved",
    )
    _default_caseops_tok = os.path.join(
        data_dir,
        "datasets",
        "fineweb10B_sp8192_caseops",
        "datasets",
        "tokenizers",
        "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    )
    if caseops_enabled:
        datasets_dir = os.environ.get("DATA_PATH", _default_caseops_data)
        tokenizer_path = os.environ.get("TOKENIZER_PATH", _default_caseops_tok)
    else:
        datasets_dir = os.environ.get(
            "DATA_PATH",
            os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}"),
        )
        tokenizer_path = os.environ.get(
            "TOKENIZER_PATH",
            os.path.join(data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model"),
        )
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    val_bytes_files = os.path.join(datasets_dir, "fineweb_val_bytes_*.bin")
    artifact_dir = os.environ.get("ARTIFACT_DIR", "")
    logfile = (
        os.path.join(artifact_dir, f"{run_id}.txt")
        if artifact_dir
        else f"logs/{run_id}.txt"
    )
    model_path = (
        os.path.join(artifact_dir, "final_model.pt")
        if artifact_dir
        else "final_model.pt"
    )
    quantized_model_path = (
        os.path.join(artifact_dir, "final_model.int6.ptz")
        if artifact_dir
        else "final_model.int6.ptz"
    )


_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.caseops_enabled = bool(getattr(h, "caseops_enabled", False))
        if self.caseops_enabled:
            self.base_bytes_lut = None
            self.has_leading_space_lut = None
            self.is_boundary_token_lut = None
        else:
            (
                self.base_bytes_lut,
                self.has_leading_space_lut,
                self.is_boundary_token_lut,
            ) = build_sentencepiece_luts(self.sp, h.vocab_size, device)
        self.val_bytes = None
        if self.caseops_enabled:
            self.val_bytes = load_validation_byte_sidecar(
                h.val_bytes_files, h.eval_seq_len, self.val_tokens.numel()
            )


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert (
        sp.piece_to_id("▁") != sp.unk_id()
    ), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
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


def load_validation_tokens(pattern, seq_len):
    # Filter out CaseOps byte sidecar shards which share the val_*.bin glob.
    files = [
        Path(p)
        for p in sorted(glob.glob(pattern))
        if "_bytes_" not in Path(p).name
    ]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_validation_byte_sidecar(pattern, seq_len, expected_len):
    """Load CaseOps per-token byte sidecar(s). Same shard layout as token shards
    (256 int32 header + uint16 array). Each entry = canonical raw-text byte
    budget for that token in the corresponding val shard. Returns a CPU
    int16 tensor sliced to match expected_len (i.e. val_tokens length)."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No byte sidecar files for pattern: {pattern}")
    shards = [load_data_shard(file) for file in files]
    # load_data_shard returns uint16 — that's exactly what the sidecar stores.
    bytes_full = torch.cat(shards).contiguous()
    if bytes_full.numel() < expected_len:
        raise ValueError(
            f"Byte sidecar too short: {bytes_full.numel()} < val_tokens {expected_len}"
        )
    return bytes_full[:expected_len].to(torch.int32)


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


BOS_ID = None


def get_next_multiple_of_n(v, n):
    return ((v + n - 1) // n) * n


def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start = starts_with_end[i]
        end = starts_with_end[i + 1]
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    boundaries = seg_starts + [total_len]
    padded_len = get_next_multiple_of_n(len(boundaries), bucket_size)
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max(end - start for start, end in zip(seg_starts, seg_ends))
    return cu, max_seqlen


class DocumentPackingLoader:
    _shard_pool = ThreadPoolExecutor(1)

    def __init__(self, h, device, cu_bucket_size=64):
        self.rank = h.rank
        self.world_size = h.world_size
        self.device = device
        self.cu_bucket_size = cu_bucket_size
        self.max_seq_len = h.train_seq_len
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files
        self.file_iter = iter(self.files)
        self._init_shard(load_data_shard(next(self.file_iter)))
        self._next_shard = self._submit_next_shard()
        self._batch_pool = ThreadPoolExecutor(1)
        self._prefetch_queue = []

    def _init_shard(self, tokens):
        global BOS_ID
        self.tokens = tokens
        self.shard_size = tokens.numel()
        if BOS_ID is None:
            BOS_ID = 1
        self.bos_idx = (
            (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        )
        self.cursor = int(self.bos_idx[0])

    def _submit_next_shard(self):
        try:
            path = next(self.file_iter)
            return self._shard_pool.submit(load_data_shard, path)
        except StopIteration:
            return None

    def _advance_shard(self):
        if self._next_shard is None:
            self.file_iter = iter(self.files)
            self._next_shard = self._shard_pool.submit(
                load_data_shard, next(self.file_iter)
            )
        self._init_shard(self._next_shard.result())
        self._next_shard = self._submit_next_shard()

    def _local_doc_starts(self, local_start, total_len):
        lo = np.searchsorted(self.bos_idx, local_start, side="left")
        hi = np.searchsorted(self.bos_idx, local_start + total_len, side="left")
        return (self.bos_idx[lo:hi] - local_start).tolist()

    def _prepare_batch(self, num_tokens_local, max_seq_len):
        per_rank_span = num_tokens_local + 1
        global_span = per_rank_span * self.world_size
        while self.cursor + global_span > self.shard_size:
            self._advance_shard()
        local_start = self.cursor + self.rank * per_rank_span
        buf = self.tokens[local_start : local_start + per_rank_span]
        inputs = torch.empty(per_rank_span - 1, dtype=torch.int64, pin_memory=True)
        targets = torch.empty(per_rank_span - 1, dtype=torch.int64, pin_memory=True)
        inputs.copy_(buf[:-1])
        targets.copy_(buf[1:])
        starts = self._local_doc_starts(local_start, inputs.numel())
        cu_seqlens, max_seqlen = _build_cu_seqlens(
            starts, inputs.numel(), inputs.device, max_seq_len, self.cu_bucket_size
        )
        cu_seqlens = cu_seqlens.pin_memory()
        self.cursor += global_span
        return inputs, targets, cu_seqlens, max_seqlen

    def next_batch(self, global_tokens, grad_accum_steps):
        num_tokens_local = global_tokens // (self.world_size * grad_accum_steps)
        while len(self._prefetch_queue) < 2:
            self._prefetch_queue.append(
                self._batch_pool.submit(self._prepare_batch, num_tokens_local, self.max_seq_len))
        inputs, targets, cu_seqlens, max_seqlen = self._prefetch_queue.pop(0).result()
        self._prefetch_queue.append(
            self._batch_pool.submit(self._prepare_batch, num_tokens_local, self.max_seq_len))
        return (
            inputs[None].to(self.device, non_blocking=True),
            targets[None].to(self.device, non_blocking=True),
            cu_seqlens.to(self.device, non_blocking=True),
            max_seqlen,
        )


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank :: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(
            self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1)
        )
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array(
                    [len(s) for s in self.start_inds], dtype=np.float64
                )
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(
                np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64)
            )
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


@triton.jit
def fused_log_softmax_dual_gather_kernel(
    logits_ptr,
    target_ids_ptr,
    hint_ids_ptr,
    log_p_y_out_ptr,
    log_q_h_out_ptr,
    BT,
    V,
    BLOCK_V: tl.constexpr,
):
    """Fused log_softmax + dual gather. Single pass over [BT, V] logits per row,
    extracts log p(target_id) and log p(hint_id) via online logsumexp.
    Replaces F.log_softmax (which materializes [BT, V] fp32) + 2 gather ops.
    """
    pid = tl.program_id(0)
    if pid >= BT:
        return

    target = tl.load(target_ids_ptr + pid)
    hint = tl.load(hint_ids_ptr + pid)
    row_offset = pid * V

    target_logit = tl.load(logits_ptr + row_offset + target).to(tl.float32)
    hint_logit = tl.load(logits_ptr + row_offset + hint).to(tl.float32)

    NEG_INF = float("-inf")
    max_val = NEG_INF
    for v_start in tl.range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        chunk = tl.load(
            logits_ptr + row_offset + v_offsets, mask=mask, other=NEG_INF
        ).to(tl.float32)
        block_max = tl.max(chunk, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for v_start in tl.range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        chunk = tl.load(
            logits_ptr + row_offset + v_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        chunk_centered = chunk - max_val
        exp_chunk = tl.where(mask, tl.exp(chunk_centered), 0.0)
        sum_exp += tl.sum(exp_chunk, axis=0)

    log_sum_exp = max_val + tl.log(sum_exp)
    log_p_y = target_logit - log_sum_exp
    log_p_h = hint_logit - log_sum_exp

    tl.store(log_p_y_out_ptr + pid, log_p_y)
    tl.store(log_q_h_out_ptr + pid, log_p_h)


def fused_log_softmax_dual_gather(logits, target_ids, hint_ids):
    """Triton wrapper — replaces F.log_softmax + 2 gather pattern.
    Returns (log_p_y, log_q_h) where p = softmax(logits).
    """
    bsz, sl, V = logits.shape
    BT = bsz * sl
    logits_flat = logits.reshape(BT, V).contiguous()
    target_flat = target_ids.reshape(BT).contiguous()
    hint_flat = hint_ids.reshape(BT).contiguous()

    log_p_y_out = torch.empty(BT, dtype=torch.float32, device=logits.device)
    log_q_h_out = torch.empty(BT, dtype=torch.float32, device=logits.device)

    BLOCK_V = 1024
    grid = (BT,)
    fused_log_softmax_dual_gather_kernel[grid](
        logits_flat,
        target_flat,
        hint_flat,
        log_p_y_out,
        log_q_h_out,
        BT,
        V,
        BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return log_p_y_out.reshape(bsz, sl), log_q_h_out.reshape(bsz, sl)


@triton.jit
def linear_leaky_relu_square_kernel(
    a_desc,
    b_desc,
    c_desc,
    aux_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FORWARD: tl.constexpr,
):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        offs_am_c = offs_am
        offs_bn_c = offs_bn
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        c1 = acc1.to(dtype)
        if not FORWARD:
            pre0 = aux_desc.load([offs_am_c, offs_bn_c])
            pre1 = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c0 = c0 * tl.where(pre0 > 0, 2.0 * pre0, 0.18 * pre0)
            c1 = c1 * tl.where(pre1 > 0, 2.0 * pre1, 0.18 * pre1)
        c_desc.store([offs_am_c, offs_bn_c], c0)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            aux0 = tl.where(c0 > 0, c0, 0.3 * c0)
            aux1 = tl.where(c1 > 0, c1, 0.3 * c1)
            aux_desc.store([offs_am_c, offs_bn_c], aux0 * aux0)
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], aux1 * aux1)


def linear_leaky_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    forward = aux is None
    if aux is None:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64
    num_stages = 4 if forward else 3
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    grid = lambda _meta: (
        min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),
    )
    linear_leaky_relu_square_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        aux_desc,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=num_sms,
        FORWARD=forward,
        num_stages=num_stages,
        num_warps=8,
    )
    if forward:
        return c, aux
    return c


class FusedLinearLeakyReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        x_flat = x.reshape(-1, x.shape[-1])
        pre, post = linear_leaky_relu_square(x_flat, w1)
        out = F.linear(post, w2)
        ctx.save_for_backward(x, w1, w2, pre, post)
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        dw2 = grad_output_flat.T @ post
        dpre = linear_leaky_relu_square(grad_output_flat, w2.T.contiguous(), aux=pre)
        dw1 = dpre.T @ x_flat
        dx = dpre @ w1
        return dx.view_as(x), dw1, dw2


FusedLeakyReLUSquareMLP = FusedLinearLeakyReLUSquareFunction.apply


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0, yarn=True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.yarn = yarn
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if self.yarn and seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
                )
            else:
                inv_freq = self.inv_freq.float().to(device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :seq_len].to(dtype=dtype), self._sin_cached[:, :seq_len].to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, yarn=True,
        attn_out_gate=False, attn_out_gate_src="proj", gate_window=12,
        gated_attn=False, gated_attn_init_std=0.01,
        sparse_attn_gate=False, sparse_attn_gate_init_std=0.0, sparse_attn_gate_scale=1.0,
        gated_xsa=False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if int(attn_out_gate) + int(gated_attn) + int(sparse_attn_gate) > 1:
            raise ValueError(
                "attn_out_gate, gated_attn, and sparse_attn_gate are mutually exclusive"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len, yarn=yarn)
        self.use_xsa = False
        # AttnOutGate (PR #1667 MarioPaerle): per-head multiplicative gate on attention
        # output. CastedLinear so restore_fp32_params casts back to fp32 for GPTQ.
        # _zero_init -> 2*sigmoid(0)=1 -> transparent at init.
        self.attn_out_gate = attn_out_gate
        self.attn_out_gate_src = attn_out_gate_src
        self.gate_window = gate_window
        if attn_out_gate:
            self.attn_gate_proj = CastedLinear(gate_window, num_heads, bias=False)
            self.attn_gate_proj._zero_init = True
        # Gated Attention (arXiv:2505.06708, Qwen, NeurIPS 2025). Per-head sigmoid
        # gate on SDPA output, BEFORE out_proj. Gate projection W_g: (num_heads, dim).
        # Name "attn_gate_w" contains "attn_gate" substring so it matches
        # CONTROL_TENSOR_NAME_PATTERNS and routes to the scalar AdamW group.
        # fp32 Parameter -> restore_fp32_params path covers it via the ndim<2 OR
        # name-pattern check (name matches "attn_gate"). Cast to x.dtype on use.
        self.gated_attn = gated_attn
        if gated_attn:
            W = torch.empty(num_heads, dim, dtype=torch.float32)
            nn.init.normal_(W, mean=0.0, std=gated_attn_init_std)
            self.attn_gate_w = nn.Parameter(W)
        # Sparse attention head-output gate (modded-nanogpt style). Keeps dense SDPA
        # and only narrows the gate input to the first gate_window residual dims.
        # W_g: (num_heads, gate_window). y_{t,h} <- sigmoid(scale * W_g_h @ x_t[:gate_window]) * y_{t,h}.
        # Shares attn_gate_w name with dense GatedAttn so the quant routing
        # (CONTROL_TENSOR_NAME_PATTERNS / attn_gate_w int8 passthrough) is unchanged.
        self.sparse_attn_gate = sparse_attn_gate
        self.sparse_attn_gate_scale = sparse_attn_gate_scale
        if sparse_attn_gate:
            W = torch.empty(num_heads, gate_window, dtype=torch.float32)
            if sparse_attn_gate_init_std > 0:
                nn.init.normal_(W, mean=0.0, std=sparse_attn_gate_init_std)
            else:
                nn.init.zeros_(W)
            self.attn_gate_w = nn.Parameter(W)
        # Gated XSA (modded-nanogpt PR #264). zero-init -> tanh(0)=0 -> the
        # original `_xsa_efficient` is unchanged at step 0. The per-head alpha
        # can grow toward +1 (full XSA), shrink toward 0 (disable XSA per head),
        # or go slightly negative (mild self-amplification). Routed to the
        # scalar AdamW group via the ndim<2 check in Optimizers.__init__.
        self.gated_xsa_enabled = gated_xsa
        if gated_xsa:
            self.xsa_alpha = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        coef = (y_g * vn).sum(dim=-1, keepdim=True)
        if getattr(self, "gated_xsa_enabled", False):
            # alpha shape (H,) -> reshape to broadcast over (B, T, Hkv, group, 1)
            a = torch.tanh(self.xsa_alpha).view(1, 1, Hkv, group, 1).to(y.dtype)
            coef = coef * a
        return (y_g - coef * vn).reshape(B, T, H, D)

    def forward(self, x, q_w, k_w, v_w, out_w, cu_seqlens=None, max_seqlen=0):
        bsz, seqlen, dim = x.shape
        # q_raw kept around as a tap point for attn_out_gate_src='q' (post-projection,
        # pre-reshape, pre-RoPE).
        q_raw = F.linear(x, q_w.to(x.dtype))
        q = q_raw.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if cu_seqlens is not None:
            y = flash_attn_varlen_func(
                q[0],
                k[0],
                v[0],
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1),
            )[None]
        else:
            y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        # AttnOutGate inlined (PR #1667). Inline + .contiguous() barrier so torch.compile
        # fullgraph=True is happy (this avoids the @torch.compiler.disable trap that
        # crashed gates v3). Per-head gate on (B,T,H,D) tensor: g shape [B,T,H], broadcast
        # over D via [..., None]. zero-init weight -> 2*sigmoid(0)=1 -> transparent.
        if self.attn_out_gate:
            gate_src = q_raw if self.attn_out_gate_src == "q" else x
            gate_in = gate_src[..., : self.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(self.attn_gate_proj(gate_in))
            y = y * g[..., None]
        # Gated Attention (arXiv:2505.06708 G1). Inline + .contiguous() barrier so
        # torch.compile fullgraph=True is happy. Per-head gate on (B,T,H,D): g shape
        # [B,T,H], broadcast over D via [..., None]. Paper: g = sigmoid(x @ W_g.T)
        # where W_g: (H, dim). .to(x.dtype) on fp32 param before broadcast with bf16.
        if self.gated_attn:
            x_c = x.contiguous()
            g = torch.sigmoid(F.linear(x_c, self.attn_gate_w.to(x.dtype)))
            y = y * g[..., None]
        # Sparse head-output gate: narrower (gate_window) input, same shape g as GatedAttn.
        if self.sparse_attn_gate:
            gate_in = x[..., : self.gate_window].contiguous()
            g = torch.sigmoid(
                self.sparse_attn_gate_scale
                * F.linear(gate_in, self.attn_gate_w.to(x.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        self._last_proj_input = y.detach() if getattr(self, "_calib", False) else None
        return F.linear(y, out_w.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.use_fused = True

    def forward(self, x, up_w, down_w):
        if self.training and self.use_fused:
            return FusedLeakyReLUSquareMLP(x, up_w.to(x.dtype), down_w.to(x.dtype))
        hidden = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.3).square()
        self._last_down_input = hidden.detach() if getattr(self, "_calib", False) else None
        return F.linear(hidden, down_w.to(x.dtype))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
        yarn=True,
        attn_out_gate=False,
        attn_out_gate_src="proj",
        gate_window=12,
        gated_attn=False,
        gated_attn_init_std=0.01,
        sparse_attn_gate=False,
        sparse_attn_gate_init_std=0.0,
        sparse_attn_gate_scale=1.0,
        gated_xsa=False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, yarn=yarn,
            attn_out_gate=attn_out_gate, attn_out_gate_src=attn_out_gate_src, gate_window=gate_window,
            gated_attn=gated_attn, gated_attn_init_std=gated_attn_init_std,
            sparse_attn_gate=sparse_attn_gate,
            sparse_attn_gate_init_std=sparse_attn_gate_init_std,
            sparse_attn_gate_scale=sparse_attn_gate_scale,
            gated_xsa=gated_xsa,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=None, max_seqlen=0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w, k_w, v_w, out_w,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[
            None, None, :
        ] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.fused_ce_enabled = bool(h.fused_ce_enabled)
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.num_layers = h.num_layers
        head_dim = h.model_dim // h.num_heads
        kv_dim = h.num_kv_heads * head_dim
        hidden_dim = int(h.mlp_mult * h.model_dim)
        self.qo_bank = nn.Parameter(torch.empty(2 * h.num_layers, h.model_dim, h.model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * h.num_layers, kv_dim, h.model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(h.num_layers, hidden_dim, h.model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, hidden_dim))
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    yarn=h.rope_yarn,
                    attn_out_gate=h.attn_out_gate_enabled,
                    attn_out_gate_src=h.attn_out_gate_src,
                    gate_window=h.gate_window,
                    gated_attn=h.gated_attn_enabled,
                    gated_attn_init_std=h.gated_attn_init_std,
                    sparse_attn_gate=h.sparse_attn_gate_enabled,
                    sparse_attn_gate_init_std=h.sparse_attn_gate_init_std,
                    sparse_attn_gate_scale=h.sparse_attn_gate_scale,
                    gated_xsa=h.gated_xsa_enabled,
                )
                for i in range(h.num_layers)
            ]
        )
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.train_seq_len,
                    rope_dims=h.rope_dims,
                    yarn=h.rope_yarn,
                )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.model_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        self.parallel_start_layer = h.parallel_start_layer
        self.parallel_final_lane = h.parallel_final_lane.lower()
        self.parallel_post_lambdas = nn.Parameter(
            torch.ones(h.num_layers, 2, 2, dtype=torch.float32)
        )
        self.parallel_resid_lambdas = nn.Parameter(
            torch.full((h.num_layers, 2), 1.1, dtype=torch.float32)
        )
        # SmearGate (PR #1667 / modded-nanogpt @classiclarryd):
        #   x_t <- x_t + lam * sigmoid(W * x_t[:gate_window]) * x_{t-1}.
        # Per-token forward-1 smear of the embedding lane. W zero-init + lam=0 ->
        # transparent at init. Uses CastedLinear so restore_fp32_params handles dtype.
        self.smear_gate_enabled = h.smear_gate_enabled
        if self.smear_gate_enabled:
            self.smear_window = h.gate_window
            self.smear_gate = CastedLinear(self.smear_window, 1, bias=False)
            self.smear_gate._zero_init = True
            self.smear_lambda = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # V19: Asymmetric Logit Rescale (PR #1923 jorge-asenjo).
        # Two learnable softcap scales applied on the EVAL path (forward_logits +
        # forward_ttt). Init to logit_softcap so the layer is identity at step 0.
        # Train path keeps the single fused softcap to preserve PR #1855 numerics.
        self.asym_logit_enabled = bool(int(os.environ.get("ASYM_LOGIT_RESCALE", "0")))
        if self.asym_logit_enabled:
            self.softcap_pos = nn.Parameter(torch.tensor(float(h.logit_softcap), dtype=torch.float32))
            self.softcap_neg = nn.Parameter(torch.tensor(float(h.logit_softcap), dtype=torch.float32))
        # v5 Stage 2C: temperature scaling on logits before softcap (eval-only TTT path).
        self.temperature_scale = float(getattr(h, "temperature_scale", 1.0))
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
        for i in range(n):
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _bank_weights(self, i):
        n = self.num_layers
        return (
            self.qo_bank[i],
            self.kv_bank[i],
            self.kv_bank[n + i],
            self.qo_bank[n + i],
            self.mlp_up_bank[i],
            self.mlp_down_bank[i],
        )

    def _parallel_block(
        self, block_idx, lane0, lane1, x0,
        q_w, k_w, v_w, out_w, up_w, down_w,
        cu_seqlens=None, max_seqlen=0,
    ):
        block = self.blocks[block_idx]
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        attn_out = block.attn(
            block.attn_norm(attn_read) * block.ln_scale_factor,
            q_w, k_w, v_w, out_w,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        mlp_read = lane1
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * block.mlp(
            block.mlp_norm(mlp_read) * block.ln_scale_factor, up_w, down_w
        )
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1

    def _final_parallel_hidden(self, lane0, lane1):
        if self.parallel_final_lane == "mlp":
            return lane1
        if self.parallel_final_lane == "attn":
            return lane0
        return 0.5 * (lane0 + lane1)

    def _forward_hidden(self, input_ids, cu_seqlens=None, max_seqlen=0):
        """Run the encoder/decoder stack to the final RMSNorm; returns pre-projection hidden.
        Shared by eval (softcap+projection via forward_logits) and train (fused CE path)."""
        x = self.tok_emb(input_ids)
        # SmearGate (PR #1667). lam=0 + W=0 -> identity at init.
        # Cross-doc leak fix: zero the prev-token smear at any position whose current token
        # is BOS, so the BOS embedding starting doc N+1 in a packed stream is not
        # contaminated by doc N's last token (audited issue on PR#1797 base).
        if self.smear_gate_enabled:
            sl = self.smear_lambda.to(dtype=x.dtype)
            gate_in = x[:, 1:, : self.smear_window].contiguous()
            g = sl * torch.sigmoid(self.smear_gate(gate_in))
            not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else range(self.num_encoder_layers)
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(
                self.num_encoder_layers,
                self.num_encoder_layers + self.num_decoder_layers,
            )
        )
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1 = self._parallel_block(
                    i, lane0, lane1, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        return x

    def _project_logits(self, hidden):
        if self.tie_embeddings:
            return F.linear(hidden, self.tok_emb.weight)
        return self.lm_head(hidden)

    def _apply_asym_softcap(self, logits):
        # V19: Asymmetric softcap (PR #1923). Splits the logit_softcap scalar into
        # learnable positive/negative branches. Score-first preserved: still a
        # bounded, normalized post-projection nonlinearity feeding a standard
        # softmax over the full vocab.
        sp = self.softcap_pos.to(logits.dtype)
        sn = self.softcap_neg.to(logits.dtype)
        return torch.where(logits > 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))

    def forward_logits(self, input_ids, cu_seqlens=None, max_seqlen=0):
        hidden = self._forward_hidden(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        logits_proj = self._project_logits(hidden)
        if self.asym_logit_enabled:
            return self._apply_asym_softcap(logits_proj)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids, cu_seqlens=None, max_seqlen=0):
        hidden = self._forward_hidden(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        logits_proj = self._project_logits(hidden)
        flat_targets = target_ids.reshape(-1)
        # Fused softcapped-CE kernel (training path only). Applies softcap inside the
        # Triton kernel; takes pre-softcap logits_proj. Non-fused path matches stock
        # PR-1736 numerics exactly (softcap in fp32, then F.cross_entropy on fp32).
        if self.fused_ce_enabled:
            return softcapped_cross_entropy(
                logits_proj.reshape(-1, logits_proj.size(-1)),
                flat_targets,
                self.logit_softcap,
                reduction="mean",
            )
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            flat_targets,
            reduction="mean",
        )

    def forward_ttt(self, input_ids, target_ids, lora, hint_ids=None):
        x = self.tok_emb(input_ids)
        # SmearGate on the TTT path — same inline compute as forward_logits.
        # Cross-doc leak fix: see _forward_hidden comment.
        if self.smear_gate_enabled:
            sl = self.smear_lambda.to(dtype=x.dtype)
            gate_in = x[:, 1:, : self.smear_window].contiguous()
            g = sl * torch.sigmoid(self.smear_gate(gate_in))
            not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else list(range(self.num_encoder_layers))
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else list(
                range(
                    self.num_encoder_layers,
                    self.num_encoder_layers + self.num_decoder_layers,
                )
            )
        )
        slot = 0
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w)
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1 = self._parallel_block_with_lora(
                    i, lane0, lane1, x0, lora, slot,
                    q_w, k_w, v_w, out_w, up_w, down_w,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w)
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + lora.lm_head_lora(x)
        # v5 Stage 2C: temperature scaling. T=1.0 (default) -> no-op.
        # Applied BEFORE softcap so cap acts on calibrated logits.
        if getattr(self, "temperature_scale", 1.0) != 1.0:
            logits = logits / self.temperature_scale
        # V19: same asymmetric softcap on the TTT eval path.
        if self.asym_logit_enabled:
            logits = self._apply_asym_softcap(logits)
        else:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        bsz, sl, V = logits.shape
        if hint_ids is None:
            return F.cross_entropy(
                logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
            ).reshape(bsz, sl)
        # PR #1145 tilt branch (v4): Triton fused kernel for eval scoring (no_grad).
        # TTT learning path needs autograd, so fall back to vanilla F.log_softmax
        # when logits require grad. Triton kernel is forward-only (no backward).
        if logits.requires_grad:
            ls = F.log_softmax(logits.float(), dim=-1)
            log_p_y = ls.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            log_q_h = ls.gather(-1, hint_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return -log_p_y, log_q_h
        log_p_y, log_q_h = fused_log_softmax_dual_gather(
            logits, target_ids, hint_ids.clamp(min=0)
        )
        return -log_p_y, log_q_h

    def _block_with_lora(self, block, x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w):
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        # Keep raw Q for AttnOutGate src='q' (matches forward path semantics).
        q_raw = F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)
        q = q_raw.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        k = F.linear(n, k_w.to(n.dtype))
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # AttnOutGate (TTT path) — inline + .contiguous() barrier, same as the eval path.
        if attn.attn_out_gate:
            gate_src = q_raw if attn.attn_out_gate_src == "q" else n
            gate_in = gate_src[..., : attn.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(attn.attn_gate_proj(gate_in))
            y = y * g[..., None]
        # Gated Attention (TTT path). Gate input is n (post-norm block input), same
        # as eval path. .to(n.dtype) on fp32 param before bf16 broadcast.
        if attn.gated_attn:
            n_c = n.contiguous()
            g = torch.sigmoid(F.linear(n_c, attn.attn_gate_w.to(n.dtype)))
            y = y * g[..., None]
        # Sparse attention head-output gate (TTT path) — must match the eval path in
        # forward() exactly, else training (which applied the gate) and TTT eval (which
        # skipped it) produce mismatched representations and catastrophic BPB regression.
        if attn.sparse_attn_gate:
            gate_in = n[..., : attn.gate_window].contiguous()
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale
                * F.linear(gate_in, attn.attn_gate_w.to(n.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        return x_out

    def _parallel_block_with_lora(
        self, block_idx, lane0, lane1, x0, lora, slot,
        q_w, k_w, v_w, out_w, up_w, down_w,
    ):
        block = self.blocks[block_idx]
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        n = block.attn_norm(attn_read) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        q_raw = F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)
        q = q_raw.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        k = F.linear(n, k_w.to(n.dtype))
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # AttnOutGate (TTT parallel path) — inline + .contiguous() barrier.
        if attn.attn_out_gate:
            gate_src = q_raw if attn.attn_out_gate_src == "q" else n
            gate_in = gate_src[..., : attn.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(attn.attn_gate_proj(gate_in))
            y = y * g[..., None]
        # Gated Attention (TTT parallel path). Gate input is n (post-norm block input).
        if attn.gated_attn:
            n_c = n.contiguous()
            g = torch.sigmoid(F.linear(n_c, attn.attn_gate_w.to(n.dtype)))
            y = y * g[..., None]
        # Sparse attention head-output gate (TTT parallel path) — must match the
        # eval path in forward() to keep train/eval semantics in sync.
        if attn.sparse_attn_gate:
            gate_in = n[..., : attn.gate_window].contiguous()
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale
                * F.linear(gate_in, attn.attn_gate_w.to(n.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        mlp_read = lane1
        mlp_n = block.mlp_norm(mlp_read) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1


class BatchedLinearLoRA(nn.Module):
    # PR-1767: rank-scaled output (alpha/rank), like standard LoRA. Decouples
    # effective magnitude from rank so changing rank does not change LR scale.
    _ALPHA = float(os.environ.get("TTT_LORA_ALPHA", "144"))
    # PR-1767: optionally keep A warm across per-doc resets (only B is zeroed).
    # Accumulates useful feature directions across documents within a TTT phase.
    _WARM_START_A = bool(int(os.environ.get("TTT_WARM_START_A", "1")))

    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self._scale = self._ALPHA / rank
        self.A = nn.Parameter(
            torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound)
        )
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            if not self._WARM_START_A:
                self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return ((x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)) * self._scale


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        self.bsz = bsz
        dim = model.qo_bank.shape[-1]
        vocab = model.tok_emb.num_embeddings
        if getattr(model, "looping_active", False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        kv_dim = model.blocks[0].attn.num_kv_heads * (
            dim // model.blocks[0].attn.num_heads
        )
        embed_dim = model.tok_emb.embedding_dim
        self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
        self.q_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
        )
        self.v_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
        )
        self.k_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
            )
            if k_lora
            else None
        )
        self.mlp_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if mlp_lora
            else None
        )
        self.o_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if o_lora
            else None
        )

    def reset(self):
        with torch.no_grad():
            self.lm_head_lora.reset()
            for loras in [self.q_loras, self.v_loras, self.k_loras,
                          self.mlp_loras, self.o_loras]:
                if loras is not None:
                    for lora in loras:
                        lora.reset()


# Polar Express per-iteration minimax Newton-Schulz coefficients (PR #1344).
# Replaces the fixed (3.4445, -4.775, 2.0315) coefficients of stock Muon.
# Applied at backend_steps=5 — taking more than 5 iterations from this list
# falls back to the final (converged) tuple via the slice guard below.
_PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    coeffs = _PE_COEFFS[:steps] if steps <= len(_PE_COEFFS) else _PE_COEFFS
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
        skylight_enabled=False,
        skylight_beta2=0.95,
        skylight_uw_floor=0.35,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )
        self._built = False
        # Skylight (modded-nanogpt PR #269 = NorMuon row-var EMA + u/w floor).
        # Applied to the full-rank update AFTER all_gather completes, so every
        # rank computes the same scale on identical inputs and keeps p replicas
        # bit-identical without extra collectives. State (row_var_ema, step)
        # lives only in self.state[p] and is never serialized.
        self._skylight_enabled = bool(skylight_enabled)
        self._skylight_beta2 = float(skylight_beta2)
        self._skylight_uw_floor = float(skylight_uw_floor)

    def _apply_skylight(self, p, update):
        """Skylight: per-row variance EMA -> rsqrt scaling -> Frobenius restore -> u/w floor.

        `update` shape: (rows..., in_features). We treat the last dim as the
        per-row feature axis (matches the existing row_normalize convention at
        the same call site, which also uses dim=-1).

        Returns a new tensor in the same dtype as `update`.
        """
        if not self._skylight_enabled:
            return update
        state = self.state[p]
        upd_f = update.float()
        # Per-row variance: mean over the last dim.
        row_var = upd_f.pow(2).mean(dim=-1, keepdim=True)
        ema = state.get("row_var_ema")
        if ema is None or ema.shape != row_var.shape or ema.device != row_var.device:
            ema = torch.zeros_like(row_var)
            state["row_var_ema"] = ema
            state["skylight_step"] = 0
        state["skylight_step"] = int(state.get("skylight_step", 0)) + 1
        step = state["skylight_step"]
        beta2 = self._skylight_beta2
        ema.mul_(beta2).add_(row_var, alpha=1.0 - beta2)
        ema_hat = ema / (1.0 - beta2 ** step)
        scale = torch.rsqrt(ema_hat + 1e-12)
        pre_fro = upd_f.norm()
        upd_f = upd_f * scale
        post_fro = upd_f.norm() + 1e-12
        upd_f = upd_f * (pre_fro / post_fro)
        # u/w floor: enforce ||U||_F / ||W||_F >= UW_FLOOR.
        w_fro = p.data.float().norm() + 1e-12
        u_fro = upd_f.norm() + 1e-12
        ratio = u_fro / w_fro
        if ratio < self._skylight_uw_floor:
            upd_f = upd_f * (self._skylight_uw_floor / ratio)
        return upd_f.to(update.dtype)

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    "p": p,
                    "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "scale": max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            pg[: m["B"]].copy_(p.grad)
            fut = dist.reduce_scatter_tensor(
                m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True
            )
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            row_normalize = group.get("row_normalize", False)
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for idx, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    upd = prev_m["full_update"][: prev_m["B"]]
                    upd = self._apply_skylight(pp, upd)
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd, alpha=-lr * prev_m["scale"])
                if sharded and self._rs_futures[idx] is not None:
                    self._rs_futures[idx].wait()
                    g = m["shard"]
                    buf = m["shard_mom"]
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                if row_normalize:
                    rn = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                    update = update / rn.to(update.dtype)
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m["full_update"], update, async_op=True
                    )
                    prev_m = m
                else:
                    update = self._apply_skylight(p, update)
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr * m["scale"])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][: prev_m["B"]]
                upd = self._apply_skylight(pp, upd)
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd, alpha=-lr * prev_m["scale"])
            if hasattr(self, "_rs_futures"):
                del self._rs_futures
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,parallel_post_lambdas,parallel_resid_lambdas,attn_gate_proj,attn_gate_w,smear_gate,smear_lambda",
    ).split(",")
    if pattern
)


PACKED_REPLICATED_GRAD_MAX_NUMEL = 1 << 15


class Optimizers:
    def __init__(self, h, base_model):
        matrix_params = [
            base_model.qo_bank,
            base_model.kv_bank,
            base_model.mlp_up_bank,
            base_model.mlp_down_bank,
        ]
        block_named_params = list(base_model.blocks.named_parameters())
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if base_model.parallel_post_lambdas is not None:
            scalar_params.append(base_model.parallel_post_lambdas)
        if base_model.parallel_resid_lambdas is not None:
            scalar_params.append(base_model.parallel_resid_lambdas)
        # SmearGate params live on GPT root (not in .blocks), so add them by hand.
        # Both are tiny (gate_window scalars + 1 lambda). Optimized via scalar Adam.
        if getattr(base_model, "smear_gate_enabled", False):
            scalar_params.append(base_model.smear_gate.weight)
            scalar_params.append(base_model.smear_lambda)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
        ]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
            skylight_enabled=h.skylight_muon_enabled,
            skylight_beta2=h.skylight_beta2,
            skylight_uw_floor=h.skylight_uw_floor,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        self.replicated_params = list(tok_params[0]["params"])
        self.replicated_params.extend(scalar_params)
        self.replicated_large_params = []
        self.replicated_packed_params = []
        for p in self.replicated_params:
            if p.numel() <= PACKED_REPLICATED_GRAD_MAX_NUMEL:
                self.replicated_packed_params.append(p)
            else:
                self.replicated_large_params.append(p)
        self._aux_stream = torch.cuda.Stream()

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def _all_reduce_packed_grads(self):
        grads_by_key = collections.defaultdict(list)
        for p in self.replicated_packed_params:
            if p.grad is not None:
                grads_by_key[(p.grad.device, p.grad.dtype)].append(p.grad)
        for grads in grads_by_key.values():
            flat = torch.empty(
                sum(g.numel() for g in grads),
                device=grads[0].device,
                dtype=grads[0].dtype,
            )
            offset = 0
            for g in grads:
                n = g.numel()
                flat[offset : offset + n].copy_(g.contiguous().view(-1))
                offset += n
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n

    def step(self, distributed=False):
        self.optimizer_muon.launch_reduce_scatters()
        if distributed:
            reduce_handles = [
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
                for p in self.replicated_large_params
                if p.grad is not None
            ]
            self._all_reduce_packed_grads()
            for handle in reduce_handles:
                handle.wait()
        self._aux_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._aux_stream):
            self.optimizer_tok.step()
            self.optimizer_scalar.step()
        self.optimizer_muon.step()
        torch.cuda.current_stream().wait_stream(self._aux_stream)
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (
            param.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ) and param.dtype != torch.float32:
            param.data = param.data.float()
    if hasattr(model, "qo_bank") and model.qo_bank is not None:
        model.qo_bank.data = model.qo_bank.data.float()
        model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    act_sumsq = {}
    act_counts = {}
    hooks = []
    for i, block in enumerate(model.blocks):
        block.attn._calib = True
        block.mlp._calib = True
        block.mlp.use_fused = False

    def make_attn_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            x_sq = x.square().sum(dim=0)
            x_count = x.shape[0]
            for suffix in ["c_q", "c_k", "c_v"]:
                name = f"blocks.{layer_idx}.attn.{suffix}.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
                if name not in act_sumsq:
                    act_sumsq[name] = torch.zeros(
                        x.shape[1], dtype=torch.float32, device=device
                    )
                    act_counts[name] = 0
                act_sumsq[name] += x_sq
                act_counts[name] += x_count
            y = module._last_proj_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.attn.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        y.shape[1], y.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(y.T, y)
                if name not in act_sumsq:
                    act_sumsq[name] = torch.zeros(
                        y.shape[1], dtype=torch.float32, device=device
                    )
                    act_counts[name] = 0
                act_sumsq[name] += y.square().sum(dim=0)
                act_counts[name] += y.shape[0]
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            name = f"blocks.{layer_idx}.mlp.fc.weight"
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)
            if name not in act_sumsq:
                act_sumsq[name] = torch.zeros(
                    x.shape[1], dtype=torch.float32, device=device
                )
                act_counts[name] = 0
            act_sumsq[name] += x.square().sum(dim=0)
            act_counts[name] += x.shape[0]
            h_act = module._last_down_input
            if h_act is not None:
                h_act = h_act.float()
                if h_act.ndim == 3:
                    h_act = h_act.reshape(-1, h_act.shape[-1])
                name = f"blocks.{layer_idx}.mlp.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        h_act.shape[1], h_act.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(h_act.T, h_act)
                if name not in act_sumsq:
                    act_sumsq[name] = torch.zeros(
                        h_act.shape[1], dtype=torch.float32, device=device
                    )
                    act_counts[name] = 0
                act_sumsq[name] += h_act.square().sum(dim=0)
                act_counts[name] += h_act.shape[0]
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

    # Hessian hooks for embedding factorization projection layers
    def make_linear_input_hook(weight_name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if weight_name not in hessians:
                hessians[weight_name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[weight_name].addmm_(x.T, x)
        return hook_fn

    if model.tie_embeddings:
        hook_module = model.final_norm

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
                if name not in act_sumsq:
                    act_sumsq[name] = torch.zeros(
                        x.shape[1], dtype=torch.float32, device=device
                    )
                    act_counts[name] = 0
                act_sumsq[name] += x.square().sum(dim=0)
                act_counts[name] += x.shape[0]
            return hook_fn

        hooks.append(
            hook_module.register_forward_hook(make_output_hook("tok_emb.weight"))
        )
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
        block.mlp.use_fused = True
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    act_stats = {}
    for name, sumsq in act_sumsq.items():
        count = max(act_counts.get(name, 0), 1)
        act_stats[name] = (sumsq / count).sqrt().cpu()
    return hessians, act_stats


def gptq_quantize_weight(
    w,
    H,
    clip_sigmas=3.0,
    clip_range=63,
    block_size=128,
    protect_groups=None,
    group_size=None,
    protect_clip_range=None,
):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    protect_meta = None
    protect_mask_perm = None
    s_hi = None
    sf_hi = None
    if (
        protect_groups
        and group_size is not None
        and protect_clip_range is not None
        and protect_clip_range > clip_range
    ):
        protect_mask = torch.zeros(cols, dtype=torch.bool)
        starts = []
        for (start, end) in protect_groups:
            if start < 0 or end > cols or end <= start:
                continue
            protect_mask[start:end] = True
            starts.append(start)
        if starts:
            protect_mask_perm = protect_mask[perm]
            s_hi = (clip_sigmas * row_std / protect_clip_range).clamp_min(1e-10).to(
                torch.float16
            )
            sf_hi = s_hi.float()
            protect_meta = {
                "starts": torch.tensor(starts, dtype=torch.int16),
                "size": int(group_size),
                "s_hi": s_hi,
            }
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            if protect_mask_perm is not None and bool(protect_mask_perm[i1 + j]):
                q_col = torch.clamp(
                    torch.round(w_col / sf_hi),
                    -protect_clip_range,
                    protect_clip_range,
                )
                w_recon = q_col.float() * sf_hi
            else:
                q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
                w_recon = q_col.float() * sf
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - w_recon) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s, protect_meta


def _quantize_gate_int8_row(w):
    # Symmetric int8-per-row quantization for small gate tensors. w shape
    # (R, C) -> (R,) scales in fp16, int8 values in [-127, 127]. Single scale
    # per row keeps accuracy high while halving storage vs fp16.
    W = w.float().contiguous()
    row_max = W.abs().amax(dim=1).clamp_min(1e-10)
    s = (row_max / 127.0).to(torch.float16)
    sf = s.float().view(-1, 1)
    q = torch.clamp(torch.round(W / sf), -127, 127).to(torch.int8)
    return q, s


def _lqer_pack(A, B, bits):
    rng = 2 ** (bits - 1) - 1
    sA = (A.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    sB = (B.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float().view(-1, 1)), -rng, rng).to(torch.int8)
    qB = torch.clamp(torch.round(B / sB.float().view(-1, 1)), -rng, rng).to(torch.int8)
    return qA, sA, qB, sB


def _lqer_pack_asym(A, B, g=64):
    # A: INT2 per-matrix scalar (signed [-2,1], scale = |A|max/1.5).
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    # B: INT4 groupwise g over flattened B (signed [-8,7], per-group scale).
    Bf = B.reshape(-1, g)
    Bmax = Bf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sB = (Bmax / 7.5).to(torch.float16).reshape(-1)
    qB = torch.clamp(torch.round(Bf / sB.float().reshape(-1, 1)), -8, 7).to(
        torch.int8
    ).reshape(B.shape)
    return qA, sA, qB, sB


def _lqer_fit_quantized(E, h):
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    r = min(h.lqer_rank, S.numel())
    if r <= 0:
        return None
    A = (U[:, :r] * S[:r]).contiguous()
    B = Vh[:r, :].contiguous()
    asym_on = bool(getattr(h, "lqer_asym_enabled", False))
    asym_g = int(getattr(h, "lqer_asym_group", 64))
    if asym_on and B.numel() % asym_g == 0:
        qA, sA, qB, sB = _lqer_pack_asym(A, B, asym_g)
        A_hat = qA.float() * float(sA)
        g_sz = qB.numel() // sB.numel()
        B_hat = (qB.reshape(-1, g_sz).float() * sB.float().view(-1, 1)).reshape(
            qB.shape
        )
        return {
            "kind": "asym",
            "qA": qA,
            "sA": sA,
            "qB": qB,
            "sB": sB,
            "delta": A_hat @ B_hat,
        }
    qA, sA, qB, sB = _lqer_pack(A, B, h.lqer_factor_bits)
    A_hat = qA.float() * sA.float().view(-1, 1)
    B_hat = qB.float() * sB.float().view(-1, 1)
    return {
        "kind": "sym",
        "qA": qA,
        "sA": sA,
        "qB": qB,
        "sB": sB,
        "delta": A_hat @ B_hat,
    }


def _awq_lite_group_candidates(w, act_rms, group_size):
    cols = w.shape[1]
    n_groups = cols // group_size
    if n_groups <= 0:
        return []
    weight_score = w.float().abs().mean(dim=0)
    saliency = act_rms.float() * weight_score
    cands = []
    for gi in range(n_groups):
        start = gi * group_size
        end = start + group_size
        score = float(saliency[start:end].sum())
        cands.append((score, start, end))
    return cands


def gptq_mixed_quantize(state_dict, hessians, act_stats, h):
    result = {}
    meta = {}
    quant_gate = bool(getattr(h, "gated_attn_quant_gate", False))
    lqer_on = bool(getattr(h, "lqer_enabled", False))
    awq_on = bool(getattr(h, "awq_lite_enabled", False))
    lqer_cands = {}
    awq_selected = collections.defaultdict(list)
    if awq_on:
        awq_cands = []
        for (name, tensor) in state_dict.items():
            t = tensor.detach().cpu().contiguous()
            if t.is_floating_point() and t.numel() > 65536 and name in act_stats:
                bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
                if bits < h.awq_lite_bits:
                    for score, start, end in _awq_lite_group_candidates(
                        t, act_stats[name], h.awq_lite_group_size
                    ):
                        awq_cands.append((score, name, start, end))
        awq_cands.sort(key=lambda x: -x[0])
        for (_score, name, start, end) in awq_cands[: h.awq_lite_group_top_k]:
            awq_selected[name].append((start, end))
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        # Dedicated int8-per-row path for attn_gate_w (bypasses both GPTQ and
        # fp16 passthrough). Applied BEFORE the numel<=65536 passthrough check
        # so the gate tensor is routed here instead of to fp16.
        if (
            quant_gate
            and t.is_floating_point()
            and t.ndim == 2
            and name.endswith(".attn_gate_w")
            # Dense GatedAttn: (num_heads, dim) = (8, 512) = 4096.
            # Sparse gate: (num_heads, gate_window) = (8, 12) = 96.
            # Both need int8-per-row routing; the 1024 lower bound in stock
            # PR-1736 presumed dense-only. Widen to catch both.
            and 32 <= t.numel() <= 8192
        ):
            gq, gs = _quantize_gate_int8_row(t)
            result[name + ".gq"] = gq
            result[name + ".gs"] = gs
            meta[name] = "gate_int8_row"
            continue
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        if "tok_emb" in name:
            cs = h.embed_clip_sigmas
        elif ".mlp." in name:
            cs = h.mlp_clip_sigmas
        elif ".attn." in name:
            cs = h.attn_clip_sigmas
        else:
            cs = h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        clip_range = 2 ** (bits - 1) - 1
        q, s, protect_meta = gptq_quantize_weight(
            t,
            hessians[name],
            clip_sigmas=cs,
            clip_range=clip_range,
            protect_groups=awq_selected.get(name),
            group_size=h.awq_lite_group_size if name in awq_selected else None,
            protect_clip_range=(2 ** (h.awq_lite_bits - 1) - 1)
            if name in awq_selected
            else None,
        )
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
        W_q = q.float() * s.float().view(-1, 1)
        if protect_meta is not None:
            result[name + ".awqg_start"] = protect_meta["starts"]
            result[name + ".awqg_s_hi"] = protect_meta["s_hi"]
            result[name + ".awqg_size"] = torch.tensor(
                protect_meta["size"], dtype=torch.int16
            )
            meta[name] = meta[name] + f"+awqgrpint{h.awq_lite_bits}"
            gsz = protect_meta["size"]
            for start in protect_meta["starts"].tolist():
                W_q[:, start : start + gsz] = (
                    q[:, start : start + gsz].float()
                    * protect_meta["s_hi"].float().view(-1, 1)
                )
        if lqer_on:
            # LQER is fit on top of the fully realized GPTQ base, which already
            # includes any higher-precision AWQ-protected groups.
            scope = str(getattr(h, "lqer_scope", "all")).lower()
            scope_ok = (
                scope == "all"
                or (scope == "mlp" and ".mlp." in name)
                or (scope == "attn" and ".attn." in name)
                or (scope == "embed" and "tok_emb" in name)
            )
            if scope_ok:
                E = t.float() - W_q
                err_norm = float(E.norm())
                if err_norm > 0:
                    lqer_cands[name] = (E, err_norm)
    if lqer_on and lqer_cands:
        if bool(getattr(h, "lqer_gain_select", False)):
            scored = []
            for (name, (E, base_err)) in lqer_cands.items():
                fit = _lqer_fit_quantized(E, h)
                if fit is None:
                    continue
                new_err = float((E - fit["delta"]).norm())
                gain = base_err - new_err
                if gain > 0:
                    scored.append((gain, name, fit))
            scored.sort(key=lambda x: -x[0])
            for (_gain, name, fit) in scored[: h.lqer_top_k]:
                if fit["kind"] == "asym":
                    result[name + ".lqA_a"] = fit["qA"]
                    result[name + ".lqAs_a"] = fit["sA"]
                    result[name + ".lqB_a"] = fit["qB"]
                    result[name + ".lqBs_a"] = fit["sB"]
                    meta[name] = meta[name] + "+lqer_asym"
                else:
                    result[name + ".lqA"] = fit["qA"]
                    result[name + ".lqAs"] = fit["sA"]
                    result[name + ".lqB"] = fit["qB"]
                    result[name + ".lqBs"] = fit["sB"]
                    meta[name] = meta[name] + "+lqer"
        else:
            top = sorted(lqer_cands.items(), key=lambda kv: -kv[1][1])[: h.lqer_top_k]
            asym_on = bool(getattr(h, "lqer_asym_enabled", False))
            asym_g = int(getattr(h, "lqer_asym_group", 64))
            for (name, (E, _)) in top:
                U, S, Vh = torch.linalg.svd(E, full_matrices=False)
                r = min(h.lqer_rank, S.numel())
                A = (U[:, :r] * S[:r]).contiguous()
                B = Vh[:r, :].contiguous()
                if asym_on and B.numel() % asym_g == 0:
                    qA, sA, qB, sB = _lqer_pack_asym(A, B, asym_g)
                    result[name + ".lqA_a"] = qA
                    result[name + ".lqAs_a"] = sA
                    result[name + ".lqB_a"] = qB
                    result[name + ".lqBs_a"] = sB
                    meta[name] = meta[name] + "+lqer_asym"
                else:
                    qA, sA, qB, sB = _lqer_pack(A, B, h.lqer_factor_bits)
                    result[name + ".lqA"] = qA
                    result[name + ".lqAs"] = sA
                    result[name + ".lqB"] = qB
                    result[name + ".lqBs"] = sB
                    meta[name] = meta[name] + "+lqer"
    categories = collections.defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for (name, orig) in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        if info == "gate_int8_row":
            gq = result[name + ".gq"]
            gs = result[name + ".gs"]
            out[name] = (gq.float() * gs.float().view(-1, 1)).to(orig_dtype)
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            W = q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
        else:
            W = q.float() * float(s.item())
        if "awqgrpint" in info:
            starts = result[name + ".awqg_start"].tolist()
            s_hi = result[name + ".awqg_s_hi"].float()
            gsz = int(result[name + ".awqg_size"].item())
            for start in starts:
                W[:, start : start + gsz] = (
                    q[:, start : start + gsz].float() * s_hi.view(-1, 1)
                )
        if "lqer_asym" in info:
            qA_t = result[name + ".lqA_a"]
            sA_t = result[name + ".lqAs_a"]
            qB_t = result[name + ".lqB_a"]
            sB_t = result[name + ".lqBs_a"]
            qA = qA_t.float() * float(sA_t)
            g_sz = qB_t.numel() // sB_t.numel()
            qB = (qB_t.reshape(-1, g_sz).float() * sB_t.float().view(-1, 1)).reshape(
                qB_t.shape
            )
            W = W + qA @ qB
        elif "lqer" in info:
            qA = result[name + ".lqA"].float() * result[name + ".lqAs"].float().view(-1, 1)
            qB = result[name + ".lqB"].float() * result[name + ".lqBs"].float().view(-1, 1)
            W = W + qA @ qB
        out[name] = W.to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"


# ── Per-group lrzip compression (ported from PR#1586 via PR#1667/1729) ────────

_GROUP_ORDER = [
    "_tok_emb.weight.q",
    "attn.c_k.weight.q", "attn.c_q.weight.q",
    "attn.c_v.weight.q", "attn.proj.weight.q",
    "mlp.fc.weight.q", "mlp.proj.weight.q",
]
_SIMSORT_KEYS = {"_tok_emb.weight.q", "attn.c_q.weight.q", "mlp.fc.weight.q"}
_PACK_MAGIC = b"PGRP"


def _similarity_sort_l1(matrix):
    import numpy as _np
    n = matrix.shape[0]
    used = _np.zeros(n, dtype=bool)
    order = [0]
    used[0] = True
    cur = matrix[0].astype(_np.float32)
    for _ in range(n - 1):
        dists = _np.sum(_np.abs(matrix[~used].astype(_np.float32) - cur), axis=1)
        unused = _np.where(~used)[0]
        best = unused[_np.argmin(dists)]
        order.append(best)
        used[best] = True
        cur = matrix[best].astype(_np.float32)
    return _np.array(order, dtype=_np.uint16)


def _lrzip_compress(data, tmpdir, label):
    inp = os.path.join(tmpdir, f"{label}.bin")
    out = f"{inp}.lrz"
    with open(inp, "wb") as f:
        f.write(data)
    subprocess.run(["lrzip", "-z", "-L", "9", "-o", out, inp], capture_output=True, check=True)
    with open(out, "rb") as f:
        result = f.read()
    os.remove(inp); os.remove(out)
    return result


def _lrzip_decompress(data, tmpdir, label):
    inp = os.path.join(tmpdir, f"{label}.lrz")
    out = os.path.join(tmpdir, f"{label}.bin")
    with open(inp, "wb") as f:
        f.write(data)
    subprocess.run(["lrzip", "-d", "-f", "-o", out, inp], capture_output=True, check=True)
    with open(out, "rb") as f:
        result = f.read()
    os.remove(inp); os.remove(out)
    return result


def _pack_streams(streams):
    import struct
    n = len(streams)
    hdr = _PACK_MAGIC + struct.pack("<I", n)
    for s in streams:
        hdr += struct.pack("<I", len(s))
    return hdr + b"".join(streams)


def _unpack_streams(blob):
    import struct
    assert blob[:4] == _PACK_MAGIC
    n = struct.unpack("<I", blob[4:8])[0]
    off = 8
    lengths = [struct.unpack("<I", blob[off + i*4:off + i*4 + 4])[0] for i in range(n)]
    off += n * 4
    streams = []
    for length in lengths:
        streams.append(blob[off:off + length])
        off += length
    return streams


def _compress(raw, compressor):
    if compressor == "brotli":
        import brotli
        return brotli.compress(raw, quality=11)
    if compressor == "lzma":
        import lzma
        return lzma.compress(raw, preset=9)
    raise ValueError(f"unknown compressor {compressor!r}")


def _decompress(blob, compressor):
    if compressor == "brotli":
        import brotli
        return brotli.decompress(blob)
    if compressor == "lzma":
        import lzma
        return lzma.decompress(blob)
    raise ValueError(f"unknown compressor {compressor!r}")


def _serialize_pergroup(quant_result, quant_meta, num_layers, tmpdir):
    import brotli
    import numpy as _np
    groups = collections.defaultdict(list)
    remainder = {}
    for name, t in sorted(quant_result.items()):
        if t.dtype != torch.int8:
            remainder[name] = t
            continue
        parts = name.split(".")
        routed = False
        if parts[0] == "blocks" and parts[1].isdigit():
            key = ".".join(parts[2:])
            if key in _GROUP_ORDER:
                groups[key].append((int(parts[1]), t))
                routed = True
        else:
            group_key = "_" + name
            if group_key in _GROUP_ORDER:
                groups[group_key] = [(0, t)]
                routed = True
        if not routed:
            # int8 tensor that doesn't fit a known group (e.g. gate_int8_row
            # tensors like attn.attn_gate_w.gq from GATED_ATTN). Stash in
            # the brotli-compressed remainder blob so it round-trips.
            remainder[name] = t

    streams = []
    all_perms = b""
    shape_manifest = {}

    for group_key in _GROUP_ORDER:
        if group_key not in groups:
            streams.append(b"")
            continue
        tensors = sorted(groups[group_key], key=lambda x: x[0])
        blob = b""
        grp_shapes = []
        for idx, t in tensors:
            arr = t.numpy()
            orig_shape = arr.shape
            if arr.ndim == 2:
                if group_key in _SIMSORT_KEYS:
                    order = _similarity_sort_l1(arr)
                    all_perms += order.tobytes()
                    arr = arr[order]
                arr = _np.ascontiguousarray(arr.T)
            blob += arr.tobytes()
            grp_shapes.append(orig_shape)
        shape_manifest[group_key] = grp_shapes
        compressed = _lrzip_compress(blob, tmpdir, group_key.replace(".", "_"))
        streams.append(compressed)

    remainder_buf = io.BytesIO()
    torch.save({"r": remainder, "m": quant_meta, "s": shape_manifest}, remainder_buf)
    streams.append(brotli.compress(remainder_buf.getvalue(), quality=11, lgwin=24))
    streams.append(brotli.compress(all_perms, quality=11) if all_perms else b"")

    return _pack_streams(streams)


def _deserialize_pergroup(blob, num_layers, tmpdir):
    import brotli
    import numpy as _np
    streams = _unpack_streams(blob)
    n_groups = len(_GROUP_ORDER)

    remainder_state = torch.load(
        io.BytesIO(brotli.decompress(streams[n_groups])), map_location="cpu"
    )
    quant_meta = remainder_state["m"]
    quant_result = dict(remainder_state["r"])
    shape_manifest = remainder_state["s"]
    all_perms = brotli.decompress(streams[n_groups + 1]) if streams[n_groups + 1] else b""

    def _decompress_one(args):
        i, gk, data = args
        if not data:
            return gk, b""
        return gk, _lrzip_decompress(data, tmpdir, f"d_{gk.replace('.', '_')}")

    from concurrent.futures import ThreadPoolExecutor as _TPool
    with _TPool(max_workers=n_groups) as pool:
        futs = [pool.submit(_decompress_one, (i, gk, streams[i])) for i, gk in enumerate(_GROUP_ORDER)]
        raw_groups = {f.result()[0]: f.result()[1] for f in futs}

    perm_off = 0
    for group_key in _GROUP_ORDER:
        raw = raw_groups.get(group_key, b"")
        if not raw:
            continue
        grp_shapes = shape_manifest[group_key]
        data_arr = _np.frombuffer(raw, dtype=_np.int8)

        if group_key.startswith("_"):
            tensor_names = [group_key[1:]]
        else:
            tensor_names = [f"blocks.{i}.{group_key}" for i in range(num_layers)]

        offset = 0
        for tname, orig_shape in zip(tensor_names, grp_shapes):
            n_elem = 1
            for d in orig_shape:
                n_elem *= d
            chunk = data_arr[offset:offset + n_elem].copy()
            offset += n_elem

            if len(orig_shape) == 2:
                rows, cols = orig_shape
                chunk = chunk.reshape(cols, rows).T

                if group_key in _SIMSORT_KEYS:
                    perm = _np.frombuffer(all_perms[perm_off:perm_off + rows * 2], dtype=_np.uint16)
                    perm_off += rows * 2
                    inv_perm = _np.empty_like(perm)
                    inv_perm[perm] = _np.arange(rows, dtype=_np.uint16)
                    chunk = chunk[inv_perm]

                chunk = chunk.reshape(orig_shape)

            quant_result[tname] = torch.from_numpy(_np.ascontiguousarray(chunk))

    return quant_result, quant_meta


def _unbank_state_dict(state_dict, num_layers):
    sd = {}
    n = num_layers
    for k, v in state_dict.items():
        t = v.detach().cpu() if v is not None else None
        if k == "qo_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_q.weight"] = t[i]
                sd[f"blocks.{i}.attn.proj.weight"] = t[n + i]
        elif k == "kv_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_k.weight"] = t[i]
                sd[f"blocks.{i}.attn.c_v.weight"] = t[n + i]
        elif k == "mlp_up_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.fc.weight"] = t[i]
        elif k == "mlp_down_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.proj.weight"] = t[i]
        else:
            if t is not None:
                sd[k] = t
    return sd


def _rebank_state_dict(flat_sd, num_layers, model_dim, kv_dim, hidden_dim):
    sd = {}
    n = num_layers
    sd["qo_bank"] = torch.zeros(2 * n, model_dim, model_dim)
    sd["kv_bank"] = torch.zeros(2 * n, kv_dim, model_dim)
    for i in range(n):
        sd["qo_bank"][i] = flat_sd[f"blocks.{i}.attn.c_q.weight"]
        sd["qo_bank"][n + i] = flat_sd[f"blocks.{i}.attn.proj.weight"]
        sd["kv_bank"][i] = flat_sd[f"blocks.{i}.attn.c_k.weight"]
        sd["kv_bank"][n + i] = flat_sd[f"blocks.{i}.attn.c_v.weight"]
    sd["mlp_up_bank"] = torch.zeros(n, hidden_dim, model_dim)
    sd["mlp_down_bank"] = torch.zeros(n, model_dim, hidden_dim)
    for i in range(n):
        sd["mlp_up_bank"][i] = flat_sd[f"blocks.{i}.mlp.fc.weight"]
        sd["mlp_down_bank"][i] = flat_sd[f"blocks.{i}.mlp.proj.weight"]
    for k, v in flat_sd.items():
        if not (
            k.startswith("blocks.")
            and any(
                p in k
                for p in [
                    ".attn.c_q.", ".attn.c_k.", ".attn.c_v.",
                    ".attn.proj.", ".mlp.fc.", ".mlp.proj.",
                ]
            )
        ):
            sd[k] = v
    return sd



def _compressed_code_size(code):
    import brotli
    code_raw = code.encode("utf-8")
    try:
        minified = subprocess.run(
            ["pyminify", "--no-rename-locals", "--no-hoist-literals", "--remove-literal-statements", "--remove-asserts", "--prefer-single-line", "-"],
            input=code_raw, capture_output=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        minified = code_raw
    compressed = brotli.compress(minified, quality=11)
    encoded = base64.b85encode(compressed)
    wrapper = b"import brotli as B,base64 as b\nexec(B.decompress(b.b85decode(\"" + encoded + b"\")))\n"
    return len(code_raw), len(wrapper)


def serialize(h, base_model, code):
    code_bytes_uncompressed, code_bytes = _compressed_code_size(code)
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size (uncompressed): {code_bytes_uncompressed} bytes")
        log(f"Code size (compressed): {code_bytes} bytes")
    sd_cpu = _unbank_state_dict(base_model.state_dict(), h.num_layers)
    device = torch.device("cuda", h.local_rank)
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    log("GPTQ:collecting Hessians from calibration data...")
    hessians, act_stats = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, act_stats, h)
    if h.compressor == "pergroup":
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="pgrp_")
        log("Serialize: per-group lrzip compression...")
        t1 = time.perf_counter()
        quant_blob = _serialize_pergroup(quant_result, quant_meta, h.num_layers, tmpdir)
        log(f"Serialize: per-group compression done in {time.perf_counter()-t1:.1f}s")
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    else:
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    flat_template = _unbank_state_dict(eval_model.state_dict(), h.num_layers)
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    if quant_blob_disk[:4] == _PACK_MAGIC:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="pgrp_dec_")
        log("Deserialize: per-group lrzip decompression...")
        t0 = time.perf_counter()
        quant_result, quant_meta = _deserialize_pergroup(
            quant_blob_disk, h.num_layers, tmpdir
        )
        log(f"Deserialize: decompression done in {time.perf_counter()-t0:.1f}s")
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    else:
        quant_state = torch.load(
            io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu"
        )
        quant_result, quant_meta = quant_state["w"], quant_state["m"]
    deq_flat = dequantize_mixed(quant_result, quant_meta, flat_template)
    head_dim = h.model_dim // h.num_heads
    kv_dim = h.num_kv_heads * head_dim
    hidden_dim = int(h.mlp_mult * h.model_dim)
    deq_state = _rebank_state_dict(deq_flat, h.num_layers, h.model_dim, kv_dim, hidden_dim)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(h, device, val_data, model, forward_logits_fn=None):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size

    # TODO: Don't truncate this.
    seq_end = seq_start + ((seq_end - seq_start) // local_batch_seqs) * local_batch_seqs

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    run_forward_logits = (
        (model.module.forward_logits if hasattr(model, "module") else model.forward_logits)
        if forward_logits_fn is None
        else forward_logits_fn
    )
    model.eval()
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1]
            y = local[1:]
            bos_pos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(
                bos_pos, x.numel(), x.device, h.eval_seq_len, 64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = run_forward_logits(
                    x[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                ).detach()
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="none",
            )
            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += float(y.numel())
            prev_ids = x
            tgt_ids = y
            sidecar_slice = val_data.val_bytes[raw_start + 1 : raw_end].to(
                device=device, dtype=torch.int32, non_blocking=True
            )
            val_byte_count += sidecar_slice.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def _find_docs(all_tokens):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1])
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _build_ttt_global_batches(doc_entries, h, ascending=False):
    batch_size = h.ttt_batch_size
    global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
    global_batches = [
        global_doc_entries[i : i + batch_size]
        for i in range(0, len(global_doc_entries), batch_size)
    ]
    indexed = list(enumerate(global_batches))
    if not ascending:
        indexed.sort(key=lambda ib: -max(dl for _, (_, dl) in ib[1]))
    return indexed


def _init_batch_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(4, "little"))


def _claim_next_batch(counter_path, queue_len):
    try:
        with open(counter_path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int.from_bytes(f.read(4), "little")
            f.seek(0)
            f.write((idx + 1).to_bytes(4, "little"))
            f.flush()
    except FileNotFoundError:
        return queue_len
    return idx


def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_start = ci * chunk_size
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(
    ptl,
    x,
    y,
    chunk_offsets,
    chunk_lens,
    pos_idx,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    loss_sum,
    byte_sum,
    token_count,
    y_bytes=None,
):
    pos = pos_idx[: x.size(1)].unsqueeze(0)
    mask = (
        (chunk_lens.unsqueeze(1) > 0)
        & (pos >= chunk_offsets.unsqueeze(1))
        & (pos < (chunk_offsets + chunk_lens).unsqueeze(1))
    )
    mask_f64 = mask.to(torch.float64)
    if y_bytes is not None:
        tok_bytes = y_bytes.to(torch.float64)
    else:
        tok_bytes = base_bytes_lut[y].to(torch.float64)
        tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(
            torch.float64
        )
    loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
    byte_sum += (tok_bytes * mask_f64).sum()
    token_count += chunk_lens.to(torch.float64).sum()


def _loss_bpb_from_sums(loss_sum, token_count, byte_sum):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
    return val_loss, val_bpb


def _add_to_counter(path, delta):
    try:
        with open(path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            cur = int.from_bytes(f.read(8), "little", signed=True)
            cur += int(delta)
            f.seek(0)
            f.write(int(cur).to_bytes(8, "little", signed=True))
            f.flush()
            return cur
    except FileNotFoundError:
        return int(delta)


def _init_int64_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(8, "little", signed=True))


def _select_ttt_doc_entries(docs, h):
    doc_entries = list(enumerate(docs))
    if h.val_doc_fraction < 1.0:
        sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
        sampled_indices = sorted(
            random.Random(h.seed).sample(range(len(docs)), sample_n)
        )
        return [(i, docs[i]) for i in sampled_indices]
    return doc_entries


def train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, val_tokens, batch_seqs=None):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    seq_len = h.eval_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = h.global_ttt_chunk_tokens
    batch_seqs = h.global_ttt_batch_seqs if batch_seqs is None else batch_seqs
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.global_ttt_lr, momentum=h.global_ttt_momentum
    )
    t_start = time.perf_counter()
    for ci in range(num_chunks):
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        is_last_chunk = ci == num_chunks - 1
        if is_last_chunk or h.global_ttt_epochs <= 0:
            continue
        base_model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs <= 0:
            continue
        warmup_chunks = max(0, min(h.global_ttt_warmup_chunks, num_chunks - 1))
        if warmup_chunks > 0 and ci < warmup_chunks:
            warmup_denom = max(warmup_chunks - 1, 1)
            warmup_t = ci / warmup_denom
            lr_now = (
                h.global_ttt_warmup_start_lr
                + (h.global_ttt_lr - h.global_ttt_warmup_start_lr) * warmup_t
            )
        else:
            decay_steps = max(num_chunks - 1 - warmup_chunks, 1)
            decay_ci = max(ci - warmup_chunks, 0)
            lr_now = h.global_ttt_lr * 0.5 * (
                1.0 + math.cos(math.pi * decay_ci / decay_steps)
            )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        my_seq_s = chunk_seqs * h.rank // h.world_size
        my_seq_e = chunk_seqs * (h.rank + 1) // h.world_size
        my_chunk_seqs = my_seq_e - my_seq_s
        for _ in range(h.global_ttt_epochs):
            for bs in range(0, my_chunk_seqs, batch_seqs):
                be = min(bs + batch_seqs, my_chunk_seqs)
                actual_bs = my_seq_s + bs
                start_tok = chunk_start + actual_bs * seq_len
                end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                if end_tok > val_tokens.numel():
                    continue
                local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x_flat = local[:-1]
                y_flat = local[1:]
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if h.global_ttt_respect_doc_boundaries:
                            bos_pos = (x_flat == BOS_ID).nonzero(as_tuple=True)[0].tolist()
                            cu_seqlens, max_seqlen = _build_cu_seqlens(
                                bos_pos, x_flat.numel(), x_flat.device, h.eval_seq_len, 64
                            )
                            loss = base_model(
                                x_flat[None],
                                y_flat[None],
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                            )
                        else:
                            x = x_flat.reshape(-1, seq_len)
                            y = y_flat.reshape(-1, seq_len)
                            loss = base_model(x, y)
                loss.backward()
                if dist.is_available() and dist.is_initialized():
                    for p in ttt_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.mul_(1.0 / h.world_size)
                if h.global_ttt_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ttt_params, h.global_ttt_grad_clip)
                optimizer.step()
        base_model.eval()
        if h.rank == 0:
            elapsed = time.perf_counter() - t_start
            log(
                f"tttg: c{ci+1}/{num_chunks} lr:{lr_now:.6f} t:{elapsed:.1f}s"
            )
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()


def _compute_ngram_hints_for_val(h, val_data, log0=print):
    """Stage 1A: precompute ngram hints over full val token sequence.
    Returns (hint_global, gate_global, boost_global) tensors on CPU, or None if tilt disabled.

    Compliance: single L->R pass over val tokens; uses val data only; produces hint
    aligned to target positions [t] for predicting all_tokens[t+1] from prefix [:t+1].
    Same compute as inline precompute, just relocated to run BEFORE eval timer.
    """
    if not getattr(h, "ngram_tilt_enabled", False):
        return None
    all_tokens = val_data.val_tokens
    targets_np_all = all_tokens.cpu().numpy().astype("uint16", copy=False)[1:]
    t_h0 = time.perf_counter()
    hints_pkg = build_hints_for_targets(
        target_token_ids_np=targets_np_all,
        tokenizer_path=h.tokenizer_path,
        vocab_size=h.vocab_size,
        log0=log0,
        token_order=h.token_order,
        token_threshold=h.token_threshold,
        token_boost=h.token_boost,
        within_tau=h.within_tau,
        within_boost=h.within_boost,
        word_order=h.word_order,
        word_normalize=h.word_normalize,
        word_tau=h.word_tau,
        word_boost=h.word_boost,
        agree_add_boost=h.agree_add_boost,
        seed_prefix_token=int(all_tokens[0].item()),
    )
    hint_global = torch.from_numpy(hints_pkg["hint_ids"].astype("int64"))
    gate_global = torch.from_numpy(hints_pkg["gate_mask"])
    boost_global = torch.from_numpy(hints_pkg["boost"].astype("float32"))
    log0(
        f"ngram_tilt:precompute_outside_timer_done elapsed={time.perf_counter()-t_h0:.2f}s "
        f"total_targets={hint_global.numel()}"
    )
    return (hint_global, gate_global, boost_global)


def eval_val_ttt_phased(h, base_model, device, val_data, forward_ttt_train, precomputed_hints=None):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    all_tokens = val_data.val_tokens
    all_tokens_idx = all_tokens.to(torch.int32)
    # === PR #1145 n-gram tilt: precompute prefix-only hints over val targets ===
    # Hints are aligned to target positions: hint_global[i] is the hint for
    # predicting token all_tokens[i+1] given prefix all_tokens[:i+1].
    # Stored on CPU as int64; gathered per-chunk to GPU alongside y indices.
    ngram_hint_global = None
    ngram_gate_global = None
    ngram_boost_global = None
    if precomputed_hints is not None:
        # v5 Stage 1A: hints were precomputed BEFORE eval timer started.
        # Save measured eval time = the precompute elapsed (~168s for full tilt).
        ngram_hint_global, ngram_gate_global, ngram_boost_global = precomputed_hints
        log(
            f"ngram_tilt:using_precomputed_hints "
            f"total_targets={ngram_hint_global.numel()} (precompute time excluded from eval)"
        )
    elif getattr(h, "ngram_tilt_enabled", False):
        targets_np_all = all_tokens.cpu().numpy().astype("uint16", copy=False)[1:]
        t_h0 = time.perf_counter()
        hints_pkg = build_hints_for_targets(
            target_token_ids_np=targets_np_all,
            tokenizer_path=h.tokenizer_path,
            vocab_size=h.vocab_size,
            log0=log,
            token_order=h.token_order,
            token_threshold=h.token_threshold,
            token_boost=h.token_boost,
            within_tau=h.within_tau,
            within_boost=h.within_boost,
            word_order=h.word_order,
            word_normalize=h.word_normalize,
            word_tau=h.word_tau,
            word_boost=h.word_boost,
            agree_add_boost=h.agree_add_boost,
            seed_prefix_token=int(all_tokens[0].item()),
        )
        ngram_hint_global = torch.from_numpy(hints_pkg["hint_ids"].astype("int64"))
        ngram_gate_global = torch.from_numpy(hints_pkg["gate_mask"])
        ngram_boost_global = torch.from_numpy(hints_pkg["boost"].astype("float32"))
        log(
            f"ngram_tilt:precompute_done elapsed={time.perf_counter()-t_h0:.2f}s "
            f"total_targets={ngram_hint_global.numel()}"
        )
    docs = _find_docs(all_tokens)
    doc_entries = _select_ttt_doc_entries(docs, h)
    prefix_doc_limit = max(0, min(len(doc_entries), int(h.phased_ttt_prefix_docs)))
    num_phases = max(1, int(h.phased_ttt_num_phases))
    phase_boundaries = []
    for pi in range(num_phases):
        boundary = prefix_doc_limit * (pi + 1) // num_phases
        phase_boundaries.append(boundary)
    current_phase = 0
    current_phase_boundary = phase_boundaries[0]
    log(
        "ttt_phased:"
        f" total_docs:{len(doc_entries)} prefix_docs:{prefix_doc_limit} "
        f"suffix_docs:{len(doc_entries) - prefix_doc_limit}"
        f" num_phases:{num_phases} boundaries:{phase_boundaries}"
    )
    chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
    eval_batch_set = None
    if h.ttt_eval_batches:
        eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
    use_ascending = eval_batch_set is not None
    global_batches_sorted = _build_ttt_global_batches(
        doc_entries, h, ascending=use_ascending
    )
    queue_len = len(global_batches_sorted)
    counter_path = f"/tmp/ttt_counter_{h.run_id}"
    prefix_counter_path = f"/tmp/ttt_prefix_counter_{h.run_id}"
    pause_flag_path = f"/tmp/ttt_pause_flag_{h.run_id}"
    if h.rank == 0:
        _init_batch_counter(counter_path)
        _init_int64_counter(prefix_counter_path)
        try:
            os.remove(pause_flag_path)
        except FileNotFoundError:
            pass
    if dist.is_available() and dist.is_initialized():
        path_list = [counter_path, prefix_counter_path, pause_flag_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path, prefix_counter_path, pause_flag_path = path_list
        dist.barrier()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    t_start = time.perf_counter()
    reusable_lora = BatchedTTTLoRA(
        h.ttt_batch_size, base_model, h.ttt_lora_rank,
        k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
    ).to(device)

    def _build_opt(lora):
        if h.ttt_optimizer == "sgd":
            return torch.optim.SGD(
                lora.parameters(), lr=h.ttt_lora_lr,
                momentum=h.ttt_beta1, weight_decay=h.ttt_weight_decay,
            )
        return torch.optim.AdamW(
            lora.parameters(), lr=h.ttt_lora_lr,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True,
        )

    reusable_opt = _build_opt(reusable_lora)
    local_scored_docs = []
    global_ttt_done = prefix_doc_limit == 0
    try:
      while True:
        queue_idx = _claim_next_batch(counter_path, queue_len)
        if queue_idx >= queue_len:
            break
        orig_batch_idx, batch_entries = global_batches_sorted[queue_idx]
        batch = [doc for _, doc in batch_entries]
        bsz = len(batch)
        prev_loss = loss_sum.item()
        prev_bytes = byte_sum.item()
        prev_tokens = token_count.item()
        if bsz == reusable_lora.bsz:
            reusable_lora.reset()
            for s in reusable_opt.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        v.zero_()
                    elif k == "step":
                        s[k] = 0
            cur_lora = reusable_lora
            cur_opt = reusable_opt
        else:
            cur_lora = BatchedTTTLoRA(
                bsz, base_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            cur_opt = _build_opt(cur_lora)
        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        num_chunks_t = torch.tensor(num_chunks, dtype=torch.int64, device=device)
        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)
            tok_starts = torch.zeros(bsz, dtype=torch.int64)
            tok_wls = torch.zeros(bsz, dtype=torch.int64)
            chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
            chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
            for b in range(bsz):
                if not active[b]:
                    continue
                doc_start, doc_len = batch[b]
                win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len
                )
                tok_starts[b] = doc_start + win_start
                tok_wls[b] = win_len
                chunk_offsets_cpu[b] = chunk_offset
                chunk_lens_cpu[b] = chunk_len
            _, context_size, chunk_offset, _ = _compute_chunk_window(
                ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len
            )
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens.numel() - 1)
            gathered_gpu = all_tokens_idx[idx].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(
                device, non_blocking=True
            )
            chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
            chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
            x = torch.where(valid, gathered_gpu[:, :context_size], 0)
            y = torch.where(valid, gathered_gpu[:, 1 : context_size + 1], 0)
            ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
            # n-gram tilt path: gather hints aligned to y, pass into forward_ttt
            hint_ids_gpu = None
            gate_mask_gpu = None
            boost_gpu = None
            if ngram_hint_global is not None:
                hint_idx_cpu = (
                    tok_starts.unsqueeze(1) + col_idx[:context_size].unsqueeze(0)
                ).clamp_(min=0, max=ngram_hint_global.numel() - 1)
                hint_ids_gpu = ngram_hint_global[hint_idx_cpu].to(
                    device=device, dtype=torch.int64, non_blocking=True
                )
                gate_mask_gpu = ngram_gate_global[hint_idx_cpu].to(
                    device=device, non_blocking=True
                )
                boost_gpu = ngram_boost_global[hint_idx_cpu].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                hint_ids_gpu = torch.where(valid, hint_ids_gpu, torch.zeros_like(hint_ids_gpu))
                gate_mask_gpu = gate_mask_gpu & valid
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if hint_ids_gpu is not None:
                    per_tok_loss, log_q_hint = forward_ttt_train(
                        x, y, lora=cur_lora, hint_ids=hint_ids_gpu
                    )
                else:
                    per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                    log_q_hint = None
            # CaseOps sidecar-driven byte budget. Mirror the index pattern
            # used to build y from all_tokens: y[b, j] corresponds to the
            # token at global position tok_starts[b] + 1 + j (when valid).
            y_bytes_arg = None
            if val_data.caseops_enabled and val_data.val_bytes is not None:
                y_idx = (
                    tok_starts.unsqueeze(1)
                    + 1
                    + col_idx[:context_size].unsqueeze(0)
                )
                y_idx = y_idx.clamp_(max=val_data.val_bytes.numel() - 1)
                y_bytes_arg = val_data.val_bytes[y_idx].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
                # Mirror the `valid` masking used for y so out-of-range tokens
                # contribute zero bytes (matches y=0 substitution above).
                y_bytes_arg = torch.where(
                    valid, y_bytes_arg, torch.zeros_like(y_bytes_arg)
                )
            # n-gram tilt application: use tilted ptl for BPB accumulation,
            # but keep original per_tok_loss for TTT-LoRA backward (training
            # objective is base NLL — tilt is a scoring-time overlay).
            if hint_ids_gpu is not None and log_q_hint is not None:
                tilted_loss = apply_tilt_to_ptl_torch_fast(
                    ptl=per_tok_loss,
                    log_q_hint=log_q_hint,
                    target_ids=y,
                    hint_ids=hint_ids_gpu,
                    gate_mask=gate_mask_gpu,
                    boost=boost_gpu,
                )
            else:
                tilted_loss = per_tok_loss
            with torch.no_grad():
                _accumulate_bpb(
                    tilted_loss,
                    x,
                    y,
                    chunk_offsets,
                    chunk_lens,
                    ctx_pos,
                    val_data.base_bytes_lut,
                    val_data.has_leading_space_lut,
                    val_data.is_boundary_token_lut,
                    loss_sum,
                    byte_sum,
                    token_count,
                    y_bytes=y_bytes_arg,
                )
            if needs_train:
                activate_chunk_mask = (num_chunks_t - 1 > ci).float()
                for gi in range(h.ttt_grad_steps):
                    if gi > 0:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                    per_doc = per_tok_loss[
                        :, chunk_offset : chunk_offset + chunk_size
                    ].mean(dim=-1)
                    cur_opt.zero_grad(set_to_none=True)
                    (per_doc * activate_chunk_mask).sum().backward()
                    cur_opt.step()
            else:
                del per_tok_loss
        batch_num = orig_batch_idx + 1
        doc_lens = [dl for _, dl in batch]
        should_report = batch_num in eval_batch_set if eval_batch_set is not None else True
        if should_report:
            cur_tokens = token_count.item()
            cur_loss_val = loss_sum.item()
            cur_bytes_val = byte_sum.item()
            dt = cur_tokens - prev_tokens
            db = cur_bytes_val - prev_bytes
            if dt > 0 and db > 0:
                b_loss = (cur_loss_val - prev_loss) / dt
                b_bpb = b_loss / math.log(2.0) * (dt / db)
            else:
                b_loss = b_bpb = 0.0
            r_loss = cur_loss_val / max(cur_tokens, 1)
            r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
            elapsed = time.perf_counter() - t_start
            log(
                f"ttp: b{batch_num}/{queue_len} bl:{b_loss:.4f} bb:{b_bpb:.4f} "
                f"rl:{r_loss:.4f} rb:{r_bpb:.4f} dl:{min(doc_lens)}-{max(doc_lens)} "
                f"gd:{int(global_ttt_done)}"
            )
        if not global_ttt_done:
            local_scored_docs.extend(
                (orig_batch_idx, pos, doc_start, doc_len)
                for pos, (doc_start, doc_len) in enumerate(batch)
            )
            prefix_done = _add_to_counter(prefix_counter_path, len(batch_entries))
            if prefix_done >= current_phase_boundary:
                try:
                    with open(pause_flag_path, "x"):
                        pass
                except FileExistsError:
                    pass
            should_pause = os.path.exists(pause_flag_path)
            if should_pause:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                gathered_scored_docs = [None] * h.world_size
                if dist.is_available() and dist.is_initialized():
                    dist.all_gather_object(gathered_scored_docs, local_scored_docs)
                else:
                    gathered_scored_docs = [local_scored_docs]
                scored_docs_for_global = []
                for rank_docs in gathered_scored_docs:
                    if rank_docs:
                        scored_docs_for_global.extend(rank_docs)
                scored_docs_for_global.sort(key=lambda x: (x[0], x[1]))
                scored_docs_for_global = scored_docs_for_global[:current_phase_boundary]
                scored_token_chunks = [
                    val_data.val_tokens[doc_start : doc_start + doc_len]
                    for _, _, doc_start, doc_len in scored_docs_for_global
                ]
                if scored_token_chunks:
                    global_ttt_tokens = torch.cat(scored_token_chunks)
                else:
                    global_ttt_tokens = val_data.val_tokens[:0]
                if h.rank == 0:
                    prefix_done = 0
                    try:
                        with open(prefix_counter_path, "rb") as f:
                            prefix_done = int.from_bytes(
                                f.read(8), "little", signed=True
                            )
                    except FileNotFoundError:
                        pass
                    log(
                        f"ttpp: phase:{current_phase + 1}/{num_phases} pd:{prefix_done} "
                        f"gd:{len(scored_docs_for_global)} "
                        f"t:{time.perf_counter() - t_start:.1f}s"
                    )
                train_val_ttt_global_sgd_distributed(
                    h, device, val_data, base_model, global_ttt_tokens
                )
                for p in base_model.parameters():
                    p.requires_grad_(False)
                reusable_lora = BatchedTTTLoRA(
                    h.ttt_batch_size, base_model, h.ttt_lora_rank,
                    k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
                ).to(device)
                reusable_opt = _build_opt(reusable_lora)
                current_phase += 1
                if current_phase >= num_phases:
                    global_ttt_done = True
                else:
                    current_phase_boundary = phase_boundaries[current_phase]
                    if h.rank == 0:
                        try:
                            os.remove(pause_flag_path)
                        except FileNotFoundError:
                            pass
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if h.rank == 0:
                    log(f"ttpr: phase:{current_phase}/{num_phases} t:{time.perf_counter() - t_start:.1f}s")
        del cur_lora, cur_opt
    finally:
        pass
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()
    return _loss_bpb_from_sums(loss_sum, token_count, byte_sum)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(
        f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms"
    )
    return val_loss, val_bpb


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    model = compiled_model
    log(f"model_params:{sum(p.numel()for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = DocumentPackingLoader(h, device)
    max_wallclock_ms = (
        1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    )
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(
            f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms"
        )

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    _clip_params = [p for p in base_model.parameters() if p.requires_grad]
    def step_fn(step, lr_scale):
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            x, y, cu_seqlens, _max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, cu_seqlens=cu_seqlens, max_seqlen=h.train_seq_len)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        if step <= h.muon_momentum_warmup_steps:

            frac = (

                min(step / h.muon_momentum_warmup_steps, 1.0)

                if h.muon_momentum_warmup_steps > 0

                else 1.0

            )

            muon_momentum = (

                1 - frac

            ) * h.muon_momentum_warmup_start + frac * h.muon_momentum

            for group in optimizers.optimizer_muon.param_groups:

                group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(_clip_params, h.grad_clip_norm)
        optimizers.step(distributed=h.distributed)
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for (name, tensor) in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        num_tokens_local = h.train_batch_tokens // h.world_size
        for blk in base_model.blocks:
            blk.attn.rotary(num_tokens_local, device, torch.bfloat16)
        cu_bucket_size = train_loader.cu_bucket_size
        warmup_cu_buckets = tuple(cu_bucket_size * i for i in range(1, 5))
        warmup_cu_iters = 3
        x, y, cu_seqlens, _ = train_loader.next_batch(
            h.train_batch_tokens, h.grad_accum_steps
        )
        log(f"warmup_cu_buckets:{','.join(str(b) for b in warmup_cu_buckets)} iters_each:{warmup_cu_iters}")
        def _run_cu_bucket_warmup():
            for bucket_len in warmup_cu_buckets:
                boundaries = list(range(0, x.size(1), max(h.train_seq_len, 1)))
                if boundaries[-1] != x.size(1):
                    boundaries.append(x.size(1))
                cu = torch.full((bucket_len,), x.size(1), dtype=torch.int32, device=device)
                cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
                for _ in range(warmup_cu_iters):
                    optimizers.zero_grad_all()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        wloss = model(x, y, cu_seqlens=cu, max_seqlen=h.train_seq_len)
                    (wloss / h.grad_accum_steps).backward()
            optimizers.zero_grad_all()
        _run_cu_bucket_warmup()
        if h.num_loops > 0:
            base_model.looping_active = True
            _run_cu_bucket_warmup()
            base_model.looping_active = False
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if (
                warmup_step <= 5
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == h.warmup_steps
            ):
                log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if (
                    warmup_step <= 5
                    or (warmup_step + 1) % 10 == 0
                    or warmup_step + 1 == h.warmup_steps
                ):
                    log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for (opt, state) in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        train_loader = DocumentPackingLoader(h, device)
    _live_state = base_model.state_dict(keep_vars=True)
    ema_state = {
        name: t.detach().float().clone()
        for (name, t) in _live_state.items()
    }
    _ema_pairs = [(ema_state[name], t) for (name, t) in _live_state.items()]
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    forced_stop_step = int(os.environ.get("FORCE_STOP_STEP", "0"))
    stop_after_step = forced_stop_step if forced_stop_step > 0 else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = (
            step == h.iterations
            or stop_after_step is not None
            and step >= stop_after_step
        )
        should_validate = (
            last_step or h.val_loss_every > 0 and step % h.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                h, device, val_data, model, compiled_forward_logits
            )
            log(
                f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if (
            h.num_loops > 0
            and not base_model.looping_active
            and frac >= h.enable_looping_at
        ):
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for ema_t, t in _ema_pairs:
                ema_t.mul_(ema_decay).add_(t.detach(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        reached_cap = (
            forced_stop_step <= 0
            and max_wallclock_ms is not None
            and approx_training_time_ms >= max_wallclock_ms
        )
        if h.distributed and forced_stop_step <= 0 and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {
        name: t.to(dtype=current_state[name].dtype) for (name, t) in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model, compiled_forward_logits


def train_and_eval(h, device):
    global BOS_ID
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    if h.artifact_dir and h.is_main_process:
        os.makedirs(h.artifact_dir, exist_ok=True)
    val_data = ValidationData(h, device)
    log(
        f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
    )
    log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    # TTT_EVAL_ONLY: skip training + GPTQ, jump straight to TTT eval on a
    # pre-existing quantized artifact. Used to test TTT-only improvements
    # (e.g., PR-1767's alpha/warm-start/WD) without retraining.
    ttt_eval_only = os.environ.get("TTT_EVAL_ONLY", "0") == "1"
    quantize_only = os.environ.get("QUANTIZE_ONLY", "0") == "1"
    if ttt_eval_only:
        log("TTT_EVAL_ONLY=1 — skipping training + GPTQ, loading saved artifact for TTT eval")
        log(f"ttt_lora_alpha: {BatchedLinearLoRA._ALPHA}")
        log(f"ttt_warm_start_a: {BatchedLinearLoRA._WARM_START_A}")
        log(f"ttt_weight_decay: {h.ttt_weight_decay}")
    elif quantize_only:
        log("QUANTIZE_ONLY=1 — skipping training, loading saved full-precision checkpoint")
        log(f"quantize_only checkpoint: {h.model_path}")
        if BOS_ID is None:
            BOS_ID = 1
        base_model = GPT(h).to(device).bfloat16()
        state = torch.load(h.model_path, map_location="cpu")
        base_model.load_state_dict(state, strict=True)
        del state
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
        if h.distributed:
            dist.barrier()
    else:
        base_model, compiled_model, compiled_forward_logits = train_model(
            h, device, val_data
        )
        torch._dynamo.reset()
        timed_eval(
            "diagnostic pre-quantization post-ema",
            eval_val,
            h,
            device,
            val_data,
            compiled_model,
            compiled_forward_logits,
        )
        if os.environ.get("PREQUANT_ONLY", "0") == "1":
            log("PREQUANT_ONLY=1 — skipping serialize/GPTQ/post-quant eval/TTT")
            return
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
        if h.distributed:
            dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    if not ttt_eval_only:
        compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
        compiled_forward_logits = torch.compile(
            eval_model.forward_logits, dynamic=False, fullgraph=True
        )
        timed_eval(
            "diagnostic quantized",
            eval_val,
            h,
            device,
            val_data,
            compiled_model,
            compiled_forward_logits,
        )
        del eval_model
    if h.ttt_enabled:
        if not ttt_eval_only:
            del compiled_model
        if ttt_eval_only:
            del eval_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        if h.rope_yarn:
            _yarn_seqlen = h.train_batch_tokens // h.grad_accum_steps
            for block in ttt_model.blocks:
                block.attn.rotary(_yarn_seqlen, device, torch.bfloat16)
        else:
            for block in ttt_model.blocks:
                block.attn.rotary._cos_cached = None
                block.attn.rotary._sin_cached = None
                block.attn.rotary._seq_len_cached = 0
                block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)

        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        def _fwd_ttt_inner_with_hints(input_ids, target_ids, lora, hint_ids):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora, hint_ids=hint_ids)

        _fwd_ttt_compiled_inner = None
        _fwd_ttt_compiled_inner_hints = None

        def _fwd_ttt(input_ids, target_ids, lora, hint_ids=None):
            nonlocal _fwd_ttt_compiled_inner, _fwd_ttt_compiled_inner_hints
            if hint_ids is None:
                if _fwd_ttt_compiled_inner is None:
                    _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
                return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)
            if _fwd_ttt_compiled_inner_hints is None:
                _fwd_ttt_compiled_inner_hints = torch.compile(
                    _fwd_ttt_inner_with_hints, dynamic=True
                )
            return _fwd_ttt_compiled_inner_hints(
                input_ids, target_ids, lora=lora, hint_ids=hint_ids
            )

        fwd_ttt_compiled = _fwd_ttt
        log(f"ttt_lora:warming up compile (random tokens, no val data)")
        if BOS_ID is None:
            BOS_ID = 1
        t_warmup = time.perf_counter()
        warmup_bszes = [h.ttt_batch_size]
        for bsz in warmup_bszes:
            wl = BatchedTTTLoRA(
                bsz, ttt_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            wo = torch.optim.AdamW(
                wl.parameters(),
                lr=h.ttt_lora_lr,
                betas=(h.ttt_beta1, h.ttt_beta2),
                eps=1e-10,
                weight_decay=h.ttt_weight_decay,
                fused=True,
            )
            for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
                xw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                yw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = fwd_ttt_compiled(xw, yw, lora=wl)
                ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
                wo.step()
                wo.zero_grad(set_to_none=True)
            del wl, wo
        torch.cuda.empty_cache()
        compile_elapsed = time.perf_counter() - t_warmup
        log(f"ttt_lora:compile warmup done ({compile_elapsed:.1f}s)")
        # v5 Stage 1A: precompute ngram hints BEFORE eval timer (single pass causal,
        # uses val tokens only — same compliance as inline). For full tilt this saves
        # ~168s of measured eval time without losing any tilt benefit.
        precomputed_hints = None
        if h.ngram_tilt_enabled and getattr(h, "ngram_hint_precompute_outside", True):
            log("v5:precomputing ngram hints OUTSIDE eval timer")
            precomputed_hints = _compute_ngram_hints_for_val(h, val_data, log0=log)
        log("\nbeginning TTT eval timer")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(
            h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled,
            precomputed_hints=precomputed_hints,
        )
        torch.cuda.synchronize()
        ttt_eval_elapsed = time.perf_counter() - t_ttt
        log(
            "quantized_ttt_phased "
            f"val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} "
            f"eval_time:{1e3*ttt_eval_elapsed:.0f}ms"
        )
        log(f"total_eval_time:{ttt_eval_elapsed:.1f}s")
        del ttt_model


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 64
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs(h.artifact_dir if h.artifact_dir else "logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for (k, v) in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log("Source code:", console=False)
        log("=" * 100, console=False)
        with open(__file__, "r", encoding="utf-8") as _src:
            log(_src.read(), console=False)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log("=" * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
