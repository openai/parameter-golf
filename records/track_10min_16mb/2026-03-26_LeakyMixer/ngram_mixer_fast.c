/*
 * Fast n-gram mixer for Parameter Golf — v2.
 *
 * Optimized: neural log-softmax and entropy are pre-computed in numpy (batch-vectorized).
 * C code only handles: hash-table n-gram cache update/lookup + one logaddexp per token.
 *
 * Compile: cc -O3 -shared -fPIC -o ngram_mixer_fast.so ngram_mixer_fast.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- Configuration ---- */
#define MAX_VOCAB    2048
#define MAX_ORDER    8
#define HASH_BITS    22
#define HASH_SIZE    (1 << HASH_BITS)
#define HASH_MASK    (HASH_SIZE - 1)

/* ---- Sparse entry for one n-gram context ---- */
typedef struct {
    uint64_t key;
    uint32_t total;
    uint16_t n_types;
    uint16_t capacity;
    uint16_t *tokens;
    uint32_t *counts;
} Entry;

typedef struct {
    int vocab_size;
    int max_order;
    float alpha_base;
    float max_entropy;

    Entry *tables[MAX_ORDER + 1];

    /* History ring buffer */
    uint16_t hist[MAX_ORDER];
    int hist_len;
    int hist_start;

    /* Dense unigram counts */
    uint32_t unigram_counts[MAX_VOCAB];
    uint32_t unigram_total;
} NgramMixer;

/* FNV-1a */
static inline uint64_t hash_ctx(const uint16_t *ctx, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)ctx[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline Entry *ht_find(Entry *table, uint64_t key, int create) {
    uint32_t idx = (uint32_t)(key & HASH_MASK);
    for (uint32_t p = 0; p < 128; p++) {
        uint32_t slot = (idx + p) & HASH_MASK;
        if (table[slot].key == key && table[slot].total > 0)
            return &table[slot];
        if (table[slot].total == 0) {
            if (create) { table[slot].key = key; return &table[slot]; }
            return NULL;
        }
    }
    return NULL;
}

static void entry_add(Entry *e, uint16_t tid) {
    for (int i = 0; i < e->n_types; i++) {
        if (e->tokens[i] == tid) { e->counts[i]++; e->total++; return; }
    }
    if (e->n_types >= e->capacity) {
        uint16_t nc = e->capacity == 0 ? 4 : (uint16_t)(e->capacity * 2);
        e->tokens = (uint16_t *)realloc(e->tokens, nc * sizeof(uint16_t));
        e->counts = (uint32_t *)realloc(e->counts, nc * sizeof(uint32_t));
        e->capacity = nc;
    }
    e->tokens[e->n_types] = tid;
    e->counts[e->n_types] = 1;
    e->n_types++;
    e->total++;
}

/* ---- Public API ---- */

NgramMixer *mixer_create(int vocab_size, int max_order, float alpha_base) {
    NgramMixer *m = (NgramMixer *)calloc(1, sizeof(NgramMixer));
    if (!m) return NULL;
    m->vocab_size = vocab_size < MAX_VOCAB ? vocab_size : MAX_VOCAB;
    m->max_order = max_order < MAX_ORDER ? max_order : MAX_ORDER;
    m->alpha_base = alpha_base;
    m->max_entropy = logf((float)m->vocab_size);
    for (int o = 1; o <= m->max_order; o++)
        m->tables[o] = (Entry *)calloc(HASH_SIZE, sizeof(Entry));
    return m;
}

void mixer_destroy(NgramMixer *m) {
    if (!m) return;
    for (int o = 1; o <= m->max_order; o++) {
        if (m->tables[o]) {
            for (int i = 0; i < HASH_SIZE; i++) {
                free(m->tables[o][i].tokens);
                free(m->tables[o][i].counts);
            }
            free(m->tables[o]);
        }
    }
    free(m);
}

static inline void mixer_update(NgramMixer *m, uint16_t tid) {
    m->unigram_counts[tid]++;
    m->unigram_total++;

    int hlen = m->hist_len;
    int mo = m->max_order;
    int limit = hlen < mo ? hlen : mo;
    for (int order = 1; order <= limit; order++) {
        uint16_t ctx[MAX_ORDER];
        for (int i = 0; i < order; i++) {
            int idx = (m->hist_start + hlen - order + i) % mo;
            ctx[i] = m->hist[idx];
        }
        Entry *e = ht_find(m->tables[order], hash_ctx(ctx, order), 1);
        if (e) entry_add(e, tid);
    }

    if (hlen < mo) {
        m->hist[(m->hist_start + hlen) % mo] = tid;
        m->hist_len++;
    } else {
        m->hist[m->hist_start] = tid;
        m->hist_start = (m->hist_start + 1) % mo;
    }
}

/*
 * v2 batch processor: neural log-probs and entropy are pre-computed.
 *
 * neural_lp:   [n_tokens, vocab_size] float32  — log-softmax of neural logits
 * entropy:     [n_tokens] float32               — entropy of neural distribution
 * target_ids:  [n_tokens] uint16
 * out_nll:     [n_tokens] float32 output        — mixed NLL per token
 */
void mixer_process_batch_v2(
    NgramMixer *m,
    const float *neural_lp,
    const float *entropy,
    const uint16_t *target_ids,
    float *out_nll,
    int n_tokens
) {
    int V = m->vocab_size;
    float alpha_base = m->alpha_base;
    float max_ent = m->max_entropy;

    for (int t = 0; t < n_tokens; t++) {
        uint16_t true_tok = target_ids[t];
        float neural_lp_true = neural_lp[t * V + true_tok];
        float ent = entropy[t];

        /* Find highest-order n-gram match with count >= 2 */
        int found_order = -1;
        float ngram_lp_true = 0.0f;

        int hlen = m->hist_len;
        int mo = m->max_order;
        int limit = hlen < mo ? hlen : mo;

        for (int order = limit; order >= 1; order--) {
            uint16_t ctx[MAX_ORDER];
            for (int i = 0; i < order; i++) {
                int idx = (m->hist_start + hlen - order + i) % mo;
                ctx[i] = m->hist[idx];
            }
            Entry *e = ht_find(m->tables[order], hash_ctx(ctx, order), 0);
            if (e && e->total >= 2) {
                float total = (float)e->total;
                float delta = 0.75f;
                /* Look for true_tok in sparse array */
                float cnt = 0.0f;
                for (int j = 0; j < e->n_types; j++) {
                    if (e->tokens[j] == true_tok) { cnt = (float)e->counts[j]; break; }
                }
                if (cnt > delta) {
                    ngram_lp_true = logf((cnt - delta) / total);
                } else {
                    /* Backoff: assign mass from discounting to unseen tokens */
                    float backoff = (delta * e->n_types) / total;
                    ngram_lp_true = logf(backoff / V + 1e-10f);
                }
                found_order = order;
                break;
            }
        }

        /* Unigram fallback */
        if (found_order < 0 && m->unigram_total > 0) {
            float c = (float)m->unigram_counts[true_tok];
            float total = (float)m->unigram_total;
            ngram_lp_true = c > 0 ? logf(c / total) : -20.0f;
            found_order = 0;
        }

        /* Mix */
        float nll;
        if (found_order >= 0) {
            float order_boost = found_order > 0 ? ((float)found_order / mo) * 0.3f : 0.0f;
            float alpha = alpha_base * (ent / max_ent) + order_boost;
            if (alpha < 0.01f) alpha = 0.01f;
            if (alpha > 0.95f) alpha = 0.95f;

            float a = logf(1.0f - alpha) + neural_lp_true;
            float b = logf(alpha) + ngram_lp_true;
            float mx = a > b ? a : b;
            nll = -(mx + logf(expf(a - mx) + expf(b - mx)));
        } else {
            nll = -neural_lp_true;
        }
        out_nll[t] = nll;

        mixer_update(m, true_tok);
    }
}

/* Keep v1 for backward compat */
void mixer_process_batch(
    NgramMixer *m,
    const float *neural_logits,
    const uint16_t *target_ids,
    float *out_mixed_nll,
    int n_tokens
) {
    int V = m->vocab_size;
    float *lp = (float *)malloc(n_tokens * V * sizeof(float));
    float *ent = (float *)malloc(n_tokens * sizeof(float));
    if (!lp || !ent) { free(lp); free(ent); return; }

    for (int t = 0; t < n_tokens; t++) {
        const float *logits = neural_logits + t * V;
        float max_l = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > max_l) max_l = logits[i];
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) {
            lp[t * V + i] = logits[i] - max_l;
            sum_exp += expf(lp[t * V + i]);
        }
        float log_sum = logf(sum_exp);
        float h = 0.0f;
        for (int i = 0; i < V; i++) {
            lp[t * V + i] -= log_sum;
            float p = expf(lp[t * V + i]);
            if (p > 1e-10f) h -= p * lp[t * V + i];
        }
        ent[t] = h;
    }
    mixer_process_batch_v2(m, lp, ent, target_ids, out_mixed_nll, n_tokens);
    free(lp);
    free(ent);
}
