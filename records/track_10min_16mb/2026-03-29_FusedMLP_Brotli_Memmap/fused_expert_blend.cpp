/*
 * fused_expert_ext — N-gram hint generator with open-addressing hash tables.
 *
 * Three expert types:
 *   1. Token PPM (orders 8-16): Long-range context, open-addressed
 *   2. Within-word (orders 1-3): BPE subword completion
 *   3. Word-start: Word-level bigram
 *
 * Key: confidence-scaled beta (β × conf) adapts per-prediction.
 * Incremental hash: O(1) per order.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace nb = nanobind;

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

// ── Open-addressed table ───────────────────────────────────────────────────

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

    std::vector<CtxEntry> ctx;
    std::vector<PairEntry> pair;

    void init(int bits) {
        uint32_t cap = 1u << bits;
        mask = cap - 1;
        ctx.assign(cap, {EMPTY_KEY, 0, 0, 0});
        pair.assign(cap, {EMPTY_KEY, 0, 0});
#ifdef __linux__
        madvise(ctx.data(), cap * sizeof(CtxEntry), MADV_HUGEPAGE);
        madvise(pair.data(), cap * sizeof(PairEntry), MADV_HUGEPAGE);
#endif
    }

    void reset() {
        std::fill(ctx.begin(), ctx.end(), CtxEntry{EMPTY_KEY, 0, 0, 0});
        std::fill(pair.begin(), pair.end(), PairEntry{EMPTY_KEY, 0, 0});
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

// ── ContextMixer ───────────────────────────────────────────────────────────

class ContextMixer {
    static constexpr int OPEN_MIN = 8;
    static constexpr int OPEN_MAX = 16;
    static constexpr int N_OPEN = OPEN_MAX - OPEN_MIN + 1;  // 9

    OpenTable open_[N_OPEN];

    struct OrderConfig { double threshold; uint32_t min_count; };
    OrderConfig cfg_[N_OPEN];

    // Which orders are active (bitmask, default all)
    bool order_active_[N_OPEN];

    // Within-word (open-addressed, orders 1-3)
    static constexpr int WITHIN_ORDERS = 3;
    OpenTable within_[WITHIN_ORDERS];
    uint64_t within_hash_;
    uint32_t within_len_;
    double within_threshold_, within_beta_;

    // Word-start
    static constexpr int WORD_ORDER = 4;
    OpenTable word_table_;
    std::vector<uint64_t> word_ring_;
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

    // ── Token hint ─────────────────────────────────────────────────────

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

    // ── Within-word ────────────────────────────────────────────────────

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

    // ── Word-start ─────────────────────────────────────────────────────

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

public:
    ContextMixer(double base_beta = 1.0, double agree_bonus = 0.5,
                 double within_threshold = 0.80, double within_beta = 0.75,
                 double word_threshold = 0.80, double word_beta = 0.50,
                 int open_table_bits = 22, double token_threshold_scale = 1.0,
                 int order_stride = 1)
        : within_hash_(0), within_len_(0),
          within_threshold_(within_threshold), within_beta_(within_beta),
          word_ring_(WORD_ORDER, 0), word_ring_head_(0), word_ring_fill_(0),
          current_word_hash_(0), current_word_len_(0),
          word_threshold_(word_threshold), word_beta_(word_beta),
          base_beta_(base_beta), agree_bonus_(agree_bonus) {

        // Active orders: 8, 8+stride, 8+2*stride, ... up to 16
        // order_stride=1: all 9 orders. order_stride=2: 8,10,12,14,16 (5 orders)
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

    void set_tokens(nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> t) {
        tokens_ = t.data(); n_tokens_ = int64_t(t.shape(0));
    }

    void set_luts(
        nb::ndarray<const int16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> bb,
        nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> ls,
        nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> bd) {
        base_bytes_ = bb.data(); has_ls_ = ls.data(); is_bnd_ = bd.data();
    }

    void reset() {
        for (auto& o : open_) o.reset();
        for (auto& w : within_) w.reset();
        word_table_.reset();
        within_hash_ = 0; within_len_ = 0;
        word_ring_head_ = 0; word_ring_fill_ = 0;
        current_word_hash_ = 0; current_word_len_ = 0;
    }

    void get_hints_batch(
        nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> positions,
        nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out_hints,
        nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> out_betas) {

        const int n = int(positions.shape(0));
        const int64_t* pos = positions.data();
        int32_t* hints = out_hints.data();
        double* betas = out_betas.data();

        uint64_t hashes[OPEN_MAX];

        for (int i = 0; i < n; i++) {
            int64_t p = pos[i];
            auto tok = uint16_t(tokens_[p]);
            bool is_bnd = is_bnd_ && is_bnd_[tok];
            bool is_ws = has_ls_ && has_ls_[tok];

            int max_avail = std::min(OPEN_MAX, int(p));
            compute_hashes(tokens_, p, OPEN_MAX, hashes);

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

            token_update(hashes, max_avail, tok);
            within_update(tok, is_bnd, is_ws);
            word_update(tok, is_bnd, is_ws);
        }
    }

    double compute_bytes(
        nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> targets,
        nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> prev_tokens) {
        const int n = int(targets.shape(0));
        const int64_t* tgt = targets.data();
        const int64_t* prev = prev_tokens.data();
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            total += base_bytes_[tgt[i]];
            if (has_ls_[tgt[i]] && !is_bnd_[prev[i]]) total += 1.0;
        }
        return total;
    }
};

NB_MODULE(fused_expert_ext, m) {
    m.doc() = "N-gram hint generator with open-addressing (orders 8-16 + within-word + word-start)";

    nb::class_<ContextMixer>(m, "ContextMixer")
        .def(nb::init<double, double, double, double, double, double, int, double, int>(),
             nb::arg("base_beta") = 1.0, nb::arg("agree_bonus") = 0.5,
             nb::arg("within_threshold") = 0.80, nb::arg("within_beta") = 0.75,
             nb::arg("word_threshold") = 0.80, nb::arg("word_beta") = 0.50,
             nb::arg("open_table_bits") = 22, nb::arg("token_threshold_scale") = 1.0,
             nb::arg("order_stride") = 1)
        .def("set_tokens", &ContextMixer::set_tokens, nb::arg("tokens"))
        .def("set_luts", &ContextMixer::set_luts,
             nb::arg("base_bytes"), nb::arg("has_leading_space"), nb::arg("is_boundary"))
        .def("reset", &ContextMixer::reset)
        .def("get_hints_batch", &ContextMixer::get_hints_batch,
             nb::arg("positions"), nb::arg("out_hints"), nb::arg("out_betas"))
        .def("compute_bytes", &ContextMixer::compute_bytes,
             nb::arg("targets"), nb::arg("prev_tokens"));
}
