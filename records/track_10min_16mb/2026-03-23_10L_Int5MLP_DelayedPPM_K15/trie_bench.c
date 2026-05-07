/*
 * Fast streaming n-gram cache with PPM backoff.
 * Compile: gcc -O3 -march=native -shared -fPIC -o libtrie.so trie_bench.c
 *
 * Uses open-addressing hash tables with generation counters.
 * PPM backoff: tries longest context first, escapes to shorter if no confident match.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_BITS 20
#define TABLE_SIZE (1 << TABLE_BITS)
#define TABLE_MASK (TABLE_SIZE - 1)
#define MAX_LEVELS 8

typedef struct {
    uint64_t key;
    uint32_t gen;
    int32_t  top_tok;
    int32_t  top_count;
    int32_t  total;
} Bucket;

static inline uint64_t fnv_hash(const int64_t *tokens, int k) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < k; i++) {
        h ^= (uint64_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h | 1;
}

static inline Bucket *tbl_find(Bucket *tbl, uint32_t gen, uint64_t key) {
    uint32_t idx = (uint32_t)(key * 0x9e3779b97f4a7c15ULL >> 44) & TABLE_MASK;
    for (int p = 0; p < 128; p++) {
        Bucket *b = &tbl[idx];
        if (b->gen != gen) return NULL;
        if (b->key == key) return b;
        idx = (idx + 1) & TABLE_MASK;
    }
    return NULL;
}

static inline void tbl_insert(Bucket *tbl, uint32_t gen, uint64_t key, int32_t next_tok) {
    uint32_t idx = (uint32_t)(key * 0x9e3779b97f4a7c15ULL >> 44) & TABLE_MASK;
    for (int p = 0; p < 128; p++) {
        Bucket *b = &tbl[idx];
        if (b->gen != gen) {
            b->key = key;
            b->gen = gen;
            b->top_tok = next_tok;
            b->top_count = 1;
            b->total = 1;
            return;
        }
        if (b->key == key) {
            b->total++;
            if (next_tok == b->top_tok) {
                b->top_count++;
            }
            return;
        }
        idx = (idx + 1) & TABLE_MASK;
    }
}

/* Single fixed-k streaming trie (kept for backward compat) */
static uint32_t g_gen = 0;
static Bucket *g_table = NULL;

int64_t streaming_trie_process(
    const int64_t *tokens, int64_t N, int32_t k, int32_t bos_id, float min_conf,
    int32_t *hit_flags, int32_t *pred_tokens, float *pred_confs
) {
    if (!g_table) {
        g_table = (Bucket *)calloc(TABLE_SIZE, sizeof(Bucket));
        if (!g_table) return -1;
    }
    g_gen++;
    int64_t doc_start = 0, n_hits = 0;
    for (int64_t i = 1; i < N; i++) {
        hit_flags[i] = 0;
        if (tokens[i - 1] == bos_id) { g_gen++; doc_start = i - 1; }
        int64_t doc_pos = i - doc_start;
        if (doc_pos < k) continue;
        uint64_t h = fnv_hash(&tokens[i - k], k);
        Bucket *b = tbl_find(g_table, g_gen, h);
        if (b) {
            float conf = (float)b->top_count / (float)b->total;
            if (conf >= min_conf) {
                hit_flags[i] = 1;
                pred_tokens[i] = b->top_tok;
                pred_confs[i] = conf;
                n_hits++;
            }
        }
        tbl_insert(g_table, g_gen, h, (int32_t)tokens[i]);
    }
    return n_hits;
}

/*
 * Fixed-k streaming trie with delayed updates.
 *
 * delay = number of most-recent target positions to exclude from the bank.
 * For delay=train_seq_len, the bank only contains occurrences whose target
 * token lies strictly outside the model's current context window.
 */
int64_t streaming_trie_process_delayed(
    const int64_t *tokens, int64_t N, int32_t k, int32_t bos_id, float min_conf,
    int32_t delay,
    int32_t *hit_flags, int32_t *pred_tokens, float *pred_confs
) {
    if (!g_table) {
        g_table = (Bucket *)calloc(TABLE_SIZE, sizeof(Bucket));
        if (!g_table) return -1;
    }
    g_gen++;
    int64_t doc_start = 0, n_hits = 0;
    for (int64_t i = 1; i < N; i++) {
        hit_flags[i] = 0;
        if (tokens[i - 1] == bos_id) { g_gen++; doc_start = i - 1; }

        int64_t doc_pos = i - doc_start;
        if (doc_pos >= k) {
            uint64_t h = fnv_hash(&tokens[i - k], k);
            Bucket *b = tbl_find(g_table, g_gen, h);
            if (b) {
                float conf = (float)b->top_count / (float)b->total;
                if (conf >= min_conf) {
                    hit_flags[i] = 1;
                    pred_tokens[i] = b->top_tok;
                    pred_confs[i] = conf;
                    n_hits++;
                }
            }
        }

        int64_t d = i - (int64_t)delay;
        if (d > doc_start) {
            int64_t delayed_doc_pos = d - doc_start;
            if (delayed_doc_pos >= k) {
                uint64_t h_delayed = fnv_hash(&tokens[d - k], k);
                tbl_insert(g_table, g_gen, h_delayed, (int32_t)tokens[d]);
            }
        }
    }
    return n_hits;
}

/*
 * PPM with backoff.
 *
 * k_values: array of prefix lengths, MUST be sorted descending (longest first).
 * min_confs: per-level minimum confidence to accept the prediction.
 * min_counts: per-level minimum total count required before trusting.
 * n_levels: number of backoff levels.
 *
 * For each position:
 *   1. Try k_values[0] (longest). If found AND conf >= min_confs[0]
 *      AND total >= min_counts[0], accept this prediction.
 *   2. Else try k_values[1], etc.
 *   3. If no level fires, no hit.
 *   4. Update ALL levels regardless of which fired.
 *
 * Outputs:
 *   hit_flags[i]  = 1 if any level fired, 0 otherwise
 *   pred_tokens[i] = predicted token
 *   pred_confs[i]  = confidence of the winning level
 *   match_levels[i] = which k fired (0-indexed into k_values), -1 if none
 *
 * Returns total hits.
 */
int64_t streaming_ppm_process(
    const int64_t *tokens,
    int64_t N,
    int32_t bos_id,
    const int32_t *k_values,
    const float   *min_confs,
    const int32_t *min_counts,
    int32_t n_levels,
    int32_t *hit_flags,
    int32_t *pred_tokens,
    float   *pred_confs,
    int32_t *match_levels
) {
    if (n_levels > MAX_LEVELS) return -1;

    Bucket *tables[MAX_LEVELS];
    uint32_t gens[MAX_LEVELS];
    uint64_t hashes[MAX_LEVELS];

    for (int lv = 0; lv < n_levels; lv++) {
        tables[lv] = (Bucket *)calloc(TABLE_SIZE, sizeof(Bucket));
        if (!tables[lv]) return -1;
        gens[lv] = 1;
    }

    int64_t doc_start = 0;
    int64_t n_hits = 0;
    int32_t max_k = k_values[0];

    for (int64_t i = 1; i < N; i++) {
        hit_flags[i] = 0;
        match_levels[i] = -1;

        if (tokens[i - 1] == bos_id) {
            for (int lv = 0; lv < n_levels; lv++) gens[lv]++;
            doc_start = i - 1;
        }

        int64_t doc_pos = i - doc_start;
        int32_t next_tok = (int32_t)tokens[i];

        /* Pre-compute hashes for all levels we can use */
        for (int lv = 0; lv < n_levels; lv++) {
            if (doc_pos >= k_values[lv]) {
                hashes[lv] = fnv_hash(&tokens[i - k_values[lv]], k_values[lv]);
            }
        }

        /* Query: try longest context first, backoff on failure */
        for (int lv = 0; lv < n_levels; lv++) {
            int32_t k = k_values[lv];
            if (doc_pos < k) continue;

            Bucket *b = tbl_find(tables[lv], gens[lv], hashes[lv]);
            if (b && b->total >= min_counts[lv]) {
                float conf = (float)b->top_count / (float)b->total;
                if (conf >= min_confs[lv]) {
                    hit_flags[i] = 1;
                    pred_tokens[i] = b->top_tok;
                    pred_confs[i] = conf;
                    match_levels[i] = lv;
                    n_hits++;
                    break;  /* PPM: accept first (longest) confident match */
                }
            }
        }

        /* Update ALL levels (not just the one that fired) */
        for (int lv = 0; lv < n_levels; lv++) {
            if (doc_pos >= k_values[lv]) {
                tbl_insert(tables[lv], gens[lv], hashes[lv], next_tok);
            }
        }
    }

    for (int lv = 0; lv < n_levels; lv++) free(tables[lv]);
    return n_hits;
}

/*
 * PPM with delayed updates.
 *
 * delay = number of most-recent target positions to exclude from all banks.
 * This lets the cache use only long-range history outside the model context.
 */
int64_t streaming_ppm_process_delayed(
    const int64_t *tokens,
    int64_t N,
    int32_t bos_id,
    const int32_t *k_values,
    const float   *min_confs,
    const int32_t *min_counts,
    int32_t n_levels,
    int32_t delay,
    int32_t *hit_flags,
    int32_t *pred_tokens,
    float   *pred_confs,
    int32_t *match_levels
) {
    if (n_levels > MAX_LEVELS) return -1;

    Bucket *tables[MAX_LEVELS];
    uint32_t gens[MAX_LEVELS];
    uint64_t hashes[MAX_LEVELS];

    for (int lv = 0; lv < n_levels; lv++) {
        tables[lv] = (Bucket *)calloc(TABLE_SIZE, sizeof(Bucket));
        if (!tables[lv]) return -1;
        gens[lv] = 1;
    }

    int64_t doc_start = 0;
    int64_t n_hits = 0;

    for (int64_t i = 1; i < N; i++) {
        hit_flags[i] = 0;
        match_levels[i] = -1;

        if (tokens[i - 1] == bos_id) {
            for (int lv = 0; lv < n_levels; lv++) gens[lv]++;
            doc_start = i - 1;
        }

        int64_t doc_pos = i - doc_start;

        /* Query current position */
        for (int lv = 0; lv < n_levels; lv++) {
            if (doc_pos >= k_values[lv]) {
                hashes[lv] = fnv_hash(&tokens[i - k_values[lv]], k_values[lv]);
            }
        }
        for (int lv = 0; lv < n_levels; lv++) {
            int32_t k = k_values[lv];
            if (doc_pos < k) continue;

            Bucket *b = tbl_find(tables[lv], gens[lv], hashes[lv]);
            if (b && b->total >= min_counts[lv]) {
                float conf = (float)b->top_count / (float)b->total;
                if (conf >= min_confs[lv]) {
                    hit_flags[i] = 1;
                    pred_tokens[i] = b->top_tok;
                    pred_confs[i] = conf;
                    match_levels[i] = lv;
                    n_hits++;
                    break;
                }
            }
        }

        /* Insert only delayed occurrences from the same document */
        int64_t d = i - (int64_t)delay;
        if (d > doc_start) {
            int64_t delayed_doc_pos = d - doc_start;
            int32_t next_tok = (int32_t)tokens[d];
            for (int lv = 0; lv < n_levels; lv++) {
                int32_t k = k_values[lv];
                if (delayed_doc_pos >= k) {
                    uint64_t h_delayed = fnv_hash(&tokens[d - k], k);
                    tbl_insert(tables[lv], gens[lv], h_delayed, next_tok);
                }
            }
        }
    }

    for (int lv = 0; lv < n_levels; lv++) free(tables[lv]);
    return n_hits;
}
