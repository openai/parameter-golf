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
