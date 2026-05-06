#include <algorithm>
#include <cstdint>
#include <cstring>

#ifdef __linux__
#include <sys/mman.h>
#endif

static constexpr uint64_t PRIMES[] = {
    36313ULL,   27191ULL,   51647ULL,   81929ULL,   131071ULL,  196613ULL,
    262147ULL,  393241ULL,  524309ULL,  655373ULL,  786433ULL,  917521ULL,
    1048583ULL, 1179653ULL, 1310729ULL, 1441801ULL, 1572869ULL, 1703941ULL,
    1835017ULL, 1966087ULL, 2097169ULL, 2228243ULL, 2359319ULL, 2490389ULL,
    2621471ULL, 2752549ULL, 2883617ULL, 3014687ULL, 3145757ULL, 3276833ULL,
    3407903ULL, 3538973ULL,
};
static constexpr int N_PRIMES = 32;
static constexpr uint64_t PAIR_MIX = 1000003ULL;
static constexpr uint64_t PREFIX_BASE = 1099511628211ULL;
static constexpr uint64_t LEN_MIX = 0x9E3779B185EBCA87ULL;
static constexpr uint64_t TABLE_MIX = 0x9e3779b97f4a7c15ULL;
static constexpr uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;

struct CtxEntry {
    uint64_t key;
    uint32_t count;
    uint16_t best_tok;
    uint16_t best_count;
};

struct PairEntry {
    uint64_t key;
    uint32_t count;
    uint32_t _pad;
};

struct OpenTable {
    uint32_t mask;
    static constexpr int MAX_PROBES = 16;

    CtxEntry* ctx = nullptr;
    PairEntry* pair = nullptr;
    size_t cap = 0;

    ~OpenTable() { free_tables(); }

    void free_tables() {
#ifdef __linux__
        if (ctx) { munmap(ctx, cap * sizeof(CtxEntry)); ctx = nullptr; }
        if (pair) { munmap(pair, cap * sizeof(PairEntry)); pair = nullptr; }
#else
        delete[] ctx; ctx = nullptr;
        delete[] pair; pair = nullptr;
#endif
    }

    void init(int bits) {
        free_tables();
        cap = size_t(1) << bits;
        mask = uint32_t(cap - 1);
#ifdef __linux__
        ctx = (CtxEntry*)mmap(nullptr, cap * sizeof(CtxEntry),
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        pair = (PairEntry*)mmap(nullptr, cap * sizeof(PairEntry),
                                PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
#else
        ctx = new CtxEntry[cap];
        pair = new PairEntry[cap];
#endif
        clear();
    }

    void clear() {
        for (size_t i = 0; i < cap; i++) ctx[i] = {EMPTY_KEY, 0, 0, 0};
        for (size_t i = 0; i < cap; i++) pair[i] = {EMPTY_KEY, 0, 0};
    }

    void reset() { clear(); }

    void prefetch_ctx(uint64_t key) const {
        uint32_t slot = uint32_t((key * TABLE_MIX) & mask);
        __builtin_prefetch(&ctx[slot], 0, 0);
    }
    void prefetch_update(uint64_t ctx_key, uint64_t pair_key) const {
        __builtin_prefetch(&ctx[uint32_t((ctx_key * TABLE_MIX) & mask)], 1, 0);
        __builtin_prefetch(&pair[uint32_t((pair_key * TABLE_MIX) & mask)], 1, 0);
    }

    void ctx_lookup(uint64_t key, int& out_tok, double& out_conf,
                    uint32_t& out_count) const {
        uint32_t slot = uint32_t((key * TABLE_MIX) & mask);
        for (int p = 0; p < MAX_PROBES; p++) {
            uint32_t s = (slot + p) & mask;
            if (ctx[s].key == key) {
                out_count = ctx[s].count;
                out_tok = ctx[s].best_tok;
                out_conf = double(ctx[s].best_count) / double(out_count);
                return;
            }
            if (ctx[s].key == EMPTY_KEY) break;
        }
        out_tok = -1; out_conf = 0.0; out_count = 0;
    }

    void update(uint64_t ctx_key, uint64_t pair_key, uint16_t token) {
        uint32_t pair_count = 0;
        {
            uint32_t slot = uint32_t((pair_key * TABLE_MIX) & mask);
            for (int p = 0; p < MAX_PROBES; p++) {
                uint32_t s = (slot + p) & mask;
                if (pair[s].key == pair_key) {
                    pair[s].count++; pair_count = pair[s].count; break;
                }
                if (pair[s].key == EMPTY_KEY) {
                    pair[s].key = pair_key; pair[s].count = 1;
                    pair_count = 1; break;
                }
            }
        }
        {
            uint32_t slot = uint32_t((ctx_key * TABLE_MIX) & mask);
            for (int p = 0; p < MAX_PROBES; p++) {
                uint32_t s = (slot + p) & mask;
                if (ctx[s].key == ctx_key) {
                    ctx[s].count++;
                    if (token == ctx[s].best_tok) ctx[s].best_count++;
                    else if (pair_count > ctx[s].best_count) {
                        ctx[s].best_tok = token;
                        ctx[s].best_count = uint16_t(std::min(pair_count, 65535u));
                    }
                    return;
                }
                if (ctx[s].key == EMPTY_KEY) {
                    ctx[s] = {ctx_key, 1, token, 1}; return;
                }
            }
        }
    }
};

class ContextMixer {
    static constexpr int OPEN_MIN = 8;
    static constexpr int OPEN_MAX = 16;
    static constexpr int N_OPEN = OPEN_MAX - OPEN_MIN + 1;

    OpenTable open_[N_OPEN];

    struct OrderConfig { double threshold; uint32_t min_count; };
    OrderConfig cfg_[N_OPEN];

    bool order_active_[N_OPEN];
    int order_stride_;

    static constexpr int WITHIN_ORDERS = 3;
    OpenTable within_[WITHIN_ORDERS];
    uint64_t within_hash_;
    uint32_t within_len_;
    double within_threshold_, within_beta_;

    static constexpr int WORD_ORDER = 4;
    OpenTable word_table_;
    uint64_t word_ring_[4];
    int word_ring_head_, word_ring_fill_;
    uint64_t current_word_hash_;
    int current_word_len_;
    double word_threshold_, word_beta_;

    double base_beta_, agree_bonus_;

    const int64_t* tokens_ = nullptr;
    int64_t n_tokens_ = 0;
    const int16_t* base_bytes_ = nullptr;
    const uint8_t* has_ls_ = nullptr;
    const uint8_t* is_bnd_ = nullptr;

    static void compute_hashes(const int64_t* tokens, int64_t pos, int max_ord,
                               uint64_t* hashes) {
        uint64_t h = 0;
        int lim = std::min(max_ord, int(pos));
        for (int k = 0; k < lim; k++) {
            h ^= uint64_t(tokens[pos - k - 1]) * PRIMES[k % N_PRIMES];
            hashes[k] = h;
        }
        for (int k = lim; k < max_ord; k++) hashes[k] = 0;
    }

    static uint64_t pair_key(uint64_t ctx, uint16_t tok, int order) {
        return (ctx * PAIR_MIX) ^ (uint64_t(tok) * PRIMES[order % N_PRIMES]);
    }

    static uint64_t extend_prefix(uint64_t h, uint16_t tok, uint32_t pos) {
        return (h * PREFIX_BASE) ^ ((uint64_t(tok) + 1) * PRIMES[pos % N_PRIMES]);
    }

    void token_hint(const uint64_t* hashes, int max_avail,
                    int& out_tok, double& out_beta) {
        for (int order = std::min(OPEN_MAX, max_avail); order >= OPEN_MIN; order--) {
            int oi = order - OPEN_MIN;
            if (!order_active_[oi]) continue;
            uint64_t ch = hashes[order - 1];
            int hint; double conf; uint32_t count;
            open_[oi].ctx_lookup(ch, hint, conf, count);
            if (hint >= 0 && conf >= cfg_[oi].threshold
                          && count >= cfg_[oi].min_count) {
                out_tok = hint;
                out_beta = base_beta_ * conf;
                return;
            }
        }
        out_tok = -1; out_beta = 0.0;
    }

    void token_update(const uint64_t* hashes, int max_avail, uint16_t token) {
        for (int order = OPEN_MIN; order <= std::min(OPEN_MAX, max_avail); order++) {
            int oi = order - OPEN_MIN;
            if (!order_active_[oi]) continue;
            uint64_t ch = hashes[order - 1];
            uint64_t pk = pair_key(ch, token, order);
            open_[oi].update(ch, pk, token);
        }
    }

    void within_hint(bool is_bnd, bool is_ws, int& out_tok, double& out_beta) {
        if (is_bnd || is_ws || within_len_ == 0) {
            out_tok = -1; out_beta = 0.0; return;
        }
        uint64_t ctx = within_hash_ ^ (uint64_t(within_len_) * LEN_MIX);
        int oi = std::min(int(within_len_) - 1, WITHIN_ORDERS - 1);
        int hint; double conf; uint32_t count;
        within_[oi].ctx_lookup(ctx, hint, conf, count);
        if (hint >= 0 && conf >= within_threshold_ && count >= 1) {
            out_tok = hint; out_beta = within_beta_;
        } else {
            out_tok = -1; out_beta = 0.0;
        }
    }

    void within_update(uint16_t token, bool is_bnd, bool is_ws) {
        if (is_bnd) { within_hash_ = 0; within_len_ = 0; return; }
        if (is_ws || within_len_ == 0) {
            within_hash_ = extend_prefix(0, token, 0);
            within_len_ = 1; return;
        }
        uint64_t ctx = within_hash_ ^ (uint64_t(within_len_) * LEN_MIX);
        uint64_t pk = (ctx * PAIR_MIX) ^ (uint64_t(token) * PRIMES[0]);
        int oi = std::min(int(within_len_) - 1, WITHIN_ORDERS - 1);
        within_[oi].update(ctx, pk, token);
        within_hash_ = extend_prefix(within_hash_, token, within_len_);
        within_len_++;
    }

    uint64_t word_ctx_hash() const {
        uint64_t h = 0;
        int n = std::min(word_ring_fill_, WORD_ORDER);
        for (int j = 0; j < n; j++) {
            int idx = (word_ring_head_ - n + j + WORD_ORDER) % WORD_ORDER;
            h ^= word_ring_[idx] * PRIMES[j % N_PRIMES];
        }
        return h;
    }

    void word_hint(bool is_ws, int& out_tok, double& out_beta) {
        if (!is_ws || word_ring_fill_ < WORD_ORDER) {
            out_tok = -1; out_beta = 0.0; return;
        }
        uint64_t ctx = word_ctx_hash();
        int hint; double conf; uint32_t count;
        word_table_.ctx_lookup(ctx, hint, conf, count);
        if (hint >= 0 && conf >= word_threshold_ && count >= 3) {
            out_tok = hint; out_beta = word_beta_;
        } else {
            out_tok = -1; out_beta = 0.0;
        }
    }

    void flush_word() {
        if (current_word_len_ == 0) return;
        word_ring_[word_ring_head_] = current_word_hash_;
        word_ring_head_ = (word_ring_head_ + 1) % WORD_ORDER;
        if (word_ring_fill_ < WORD_ORDER) word_ring_fill_++;
        current_word_hash_ = 0; current_word_len_ = 0;
    }

    void word_update(uint16_t token, bool is_bnd, bool is_ws) {
        if (is_bnd) { flush_word(); return; }
        if (is_ws) {
            flush_word();
            if (word_ring_fill_ >= WORD_ORDER) {
                uint64_t ctx = word_ctx_hash();
                uint64_t pk = pair_key(ctx, token, WORD_ORDER);
                word_table_.update(ctx, pk, token);
            }
        }
        current_word_hash_ = current_word_hash_ * 31 + token;
        current_word_len_++;
    }

    void prefetch_open_lookups(const uint64_t* hashes, int max_avail) const {
        for (int order = std::min(OPEN_MAX, max_avail); order >= OPEN_MIN; order--) {
            int oi = order - OPEN_MIN;
            if (!order_active_[oi]) continue;
            open_[oi].prefetch_ctx(hashes[order - 1]);
        }
    }

    void prefetch_open_updates(const uint64_t* hashes, int max_avail, uint16_t token) const {
        for (int order = OPEN_MIN; order <= std::min(OPEN_MAX, max_avail); order++) {
            int oi = order - OPEN_MIN;
            if (!order_active_[oi]) continue;
            uint64_t ch = hashes[order - 1];
            uint64_t pk = pair_key(ch, token, order);
            open_[oi].prefetch_update(ch, pk);
        }
    }

public:
    ContextMixer(double base_beta = 1.0, double agree_bonus = 0.5,
                 double within_threshold = 0.80, double within_beta = 0.75,
                 double word_threshold = 0.80, double word_beta = 0.50,
                 int open_table_bits = 22, double token_threshold_scale = 1.0,
                 int order_stride = 1)
        : within_hash_(0), within_len_(0),
          within_threshold_(within_threshold), within_beta_(within_beta),
          word_ring_head_(0), word_ring_fill_(0),
          current_word_hash_(0), current_word_len_(0),
          word_threshold_(word_threshold), word_beta_(word_beta),
          base_beta_(base_beta), agree_bonus_(agree_bonus),
          order_stride_(order_stride) {

        std::memset(word_ring_, 0, sizeof(word_ring_));

        for (int i = 0; i < N_OPEN; i++) {
            int order = OPEN_MIN + i;
            order_active_[i] = ((order - OPEN_MIN) % order_stride == 0);
            if (order_active_[i])
                open_[i].init(open_table_bits);
        }

        double s = token_threshold_scale;
        for (int o = 8; o <= 10; o++)  cfg_[o - OPEN_MIN] = {0.70 * s, 3};
        for (int o = 11; o <= 13; o++) cfg_[o - OPEN_MIN] = {0.60 * s, 2};
        for (int o = 14; o <= 16; o++) cfg_[o - OPEN_MIN] = {0.50 * s, 2};

        for (int i = 0; i < WITHIN_ORDERS; i++)
            within_[i].init(20);

        word_table_.init(20);
    }

    void set_tokens(const int64_t* t, int64_t n) {
        tokens_ = t; n_tokens_ = n;
    }

    void set_luts(const int16_t* bb, const uint8_t* ls, const uint8_t* bd) {
        base_bytes_ = bb; has_ls_ = ls; is_bnd_ = bd;
    }

    void reset() {
        for (auto& o : open_) if (o.ctx) o.reset();
        for (auto& w : within_) w.reset();
        word_table_.reset();
        within_hash_ = 0; within_len_ = 0;
        word_ring_head_ = 0; word_ring_fill_ = 0;
        current_word_hash_ = 0; current_word_len_ = 0;
    }

    void get_hints_batch(const int64_t* pos, int n,
                         int32_t* hints, double* betas) {

        uint64_t hashes[OPEN_MAX];
        uint64_t next_hashes[OPEN_MAX];

        if (n > 0) {
            int64_t p0 = pos[0];
            compute_hashes(tokens_, p0, OPEN_MAX, hashes);
            int ma0 = std::min(OPEN_MAX, int(p0));
            prefetch_open_lookups(hashes, ma0);
        }

        // CAUSAL FIX (matches @abaybektursun's fix in PR #1420 — see
        // https://github.com/openai/parameter-golf/pull/1420#issuecomment-4199452189):
        //   1. Hint gating: is_bnd / is_ws derived from tokens_[p-1] (last prefix
        //      token), not tokens_[p]. This makes the predictive distribution at
        //      position p depend only on the strict prefix, satisfying Issue #1017
        //      condition 2.
        //   2. Update functions: tok_is_bnd / tok_is_ws derived from the actual
        //      target tok so within_update / word_update still segment words
        //      correctly. This is causal because updates happen AFTER the hint
        //      for position p has been written to the output buffer.
        //
        // (Variable naming and structure copied verbatim from PR #1420's fix.
        //  In addition, this submission is run with NGRAM_WITHIN_BETA=0
        //  NGRAM_WORD_BETA=0 to disable the within/word experts entirely,
        //  because empirically they contribute negative BPB once the leak is
        //  removed — see Legality Fix section in the README.)
        for (int i = 0; i < n; i++) {
            int64_t p = pos[i];
            auto tok = uint16_t(tokens_[p]);
            auto prev_tok = (p > 0) ? uint16_t(tokens_[p - 1]) : uint16_t(0);
            bool is_bnd = is_bnd_ && is_bnd_[prev_tok];
            bool is_ws = has_ls_ && has_ls_[prev_tok];
            int max_avail = std::min(OPEN_MAX, int(p));

            if (i + 1 < n) {
                int64_t np = pos[i + 1];
                compute_hashes(tokens_, np, OPEN_MAX, next_hashes);
                int nma = std::min(OPEN_MAX, int(np));
                prefetch_open_lookups(next_hashes, nma);
            }

            int tok_hint, within_tok, word_tok;
            double tok_beta, within_b, word_b;
            token_hint(hashes, max_avail, tok_hint, tok_beta);
            within_hint(is_bnd, is_ws, within_tok, within_b);
            word_hint(is_ws, word_tok, word_b);

            struct Cand { int hint; double beta; };
            Cand cands[3]; int nc = 0;
            if (tok_hint >= 0) cands[nc++] = {tok_hint, tok_beta};
            if (within_tok >= 0) cands[nc++] = {within_tok, within_b};
            if (word_tok >= 0) cands[nc++] = {word_tok, word_b};

            int best_hint = -1; double best_beta = 0.0;
            if (nc > 0) {
                for (int a = 0; a < nc; a++)
                    for (int b = 0; b < nc; b++)
                        if (b != a && cands[b].hint == cands[a].hint)
                            { cands[a].beta += agree_bonus_; break; }
                int bi = 0;
                for (int a = 1; a < nc; a++)
                    if (cands[a].beta > cands[bi].beta) bi = a;
                best_hint = cands[bi].hint;
                best_beta = cands[bi].beta;
            }

            hints[i] = best_hint;
            betas[i] = best_beta;

            prefetch_open_updates(hashes, max_avail, tok);

            bool tok_is_bnd = is_bnd_ && is_bnd_[tok];
            bool tok_is_ws = has_ls_ && has_ls_[tok];
            token_update(hashes, max_avail, tok);
            within_update(tok, tok_is_bnd, tok_is_ws);
            word_update(tok, tok_is_bnd, tok_is_ws);

            std::memcpy(hashes, next_hashes, sizeof(hashes));
        }
    }

};



extern "C" {

void* ctxmixer_new(double base_beta, double agree_bonus,
                   double within_threshold, double within_beta,
                   double word_threshold, double word_beta,
                   int open_table_bits, double token_threshold_scale,
                   int order_stride) {
    return new ContextMixer(base_beta, agree_bonus,
                            within_threshold, within_beta,
                            word_threshold, word_beta,
                            open_table_bits, token_threshold_scale,
                            order_stride);
}

void ctxmixer_delete(void* self) {
    delete static_cast<ContextMixer*>(self);
}

void ctxmixer_set_tokens(void* self, const int64_t* tokens, int64_t n) {
    static_cast<ContextMixer*>(self)->set_tokens(tokens, n);
}

void ctxmixer_set_luts(void* self,
                       const int16_t* bb,
                       const uint8_t* ls,
                       const uint8_t* bd) {
    static_cast<ContextMixer*>(self)->set_luts(bb, ls, bd);
}

void ctxmixer_reset(void* self) {
    static_cast<ContextMixer*>(self)->reset();
}

void ctxmixer_get_hints_batch(void* self, const int64_t* pos, int n,
                              int32_t* out_hints, double* out_betas) {
    static_cast<ContextMixer*>(self)->get_hints_batch(pos, n, out_hints, out_betas);
}

}  // extern "C"
