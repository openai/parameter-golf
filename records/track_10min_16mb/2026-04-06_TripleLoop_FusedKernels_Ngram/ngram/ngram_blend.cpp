/*
 * fast_ngram_ext — C++ accelerated n-gram lookup, blend, and causal update.
 *
 * Replaces the pure Python/NumPy inner loop in eval_ngram.py.
 * Same algorithm, same hash functions, same data structures (flat arrays).
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

static constexpr uint64_t PRIMES[9] = {
    36313ULL, 27191ULL, 51647ULL, 81929ULL, 131071ULL,
    174763ULL, 233017ULL, 310019ULL, 412553ULL,
};

class NGramBlender {
    int min_order_, max_order_, n_orders_;
    int min_count_, ngram_buckets_;
    uint64_t bucket_mask_;

    // Alpha config
    int alpha_mode_;  // 0=fixed, 1=entropy, 2=order_entropy
    double fixed_alpha_;
    double ent_base_, ent_range_, ent_scale_, ent_thresh_;
    double order_ent_center_, order_ent_slope_;

    // Mixing function: 0=linear, 1=logistic, 2=geometric
    int mixing_fn_;

    // Count tables: flat arrays indexed by (hash & mask)
    std::vector<std::vector<uint32_t>> ctx_tables_;
    std::vector<std::vector<uint32_t>> full_tables_;

    // Borrowed pointer to full token array
    const int64_t* tokens_ = nullptr;
    int64_t n_tokens_ = 0;

    // --- Hash functions (match Python exactly) ---

    inline uint64_t hash_ctx(int64_t pos, int ctx_w) const {
        uint64_t h = 0;
        for (int k = 0; k < ctx_w; k++) {
            auto tok = static_cast<uint64_t>(tokens_[pos - (ctx_w - k)]);
            h ^= tok * PRIMES[k % 9];
        }
        return h;
    }

    inline uint64_t hash_with_target(uint64_t ctx_h, uint64_t target,
                                     int ctx_w) const {
        return ctx_h ^ (target * PRIMES[ctx_w % 9]);
    }

    // --- Core stride processing on raw pointers ---

    void process_stride_impl(const int64_t* pos, int n_pos,
                             const double* nll, const double* ent,
                             double* out) {
        // Per-position best n-gram match
        std::vector<double> best_p(n_pos, -1.0);
        std::vector<int> best_ord(n_pos, 0);

        // --- LOOKUP: highest order first (backoff) ---
        for (int oi = n_orders_ - 1; oi >= 0; oi--) {
            int order = min_order_ + oi;
            int ctx_w = order - 1;

            for (int p = 0; p < n_pos; p++) {
                if (best_p[p] >= 0.0) continue;
                if (pos[p] < order) continue;

                uint64_t ch = hash_ctx(pos[p], ctx_w);
                uint64_t ck = ch & bucket_mask_;
                auto tgt = static_cast<uint64_t>(tokens_[pos[p]]);
                uint64_t fk = hash_with_target(ch, tgt, ctx_w) & bucket_mask_;

                auto cc = static_cast<double>(ctx_tables_[oi][ck]);
                auto fc = static_cast<double>(full_tables_[oi][fk]);

                if (cc >= static_cast<double>(min_count_)) {
                    double pn = std::min(fc, cc) / std::max(cc, 1.0);
                    best_p[p] = std::clamp(pn, 0.0, 1.0);
                    best_ord[p] = order;
                }
            }
        }

        // --- MIX ---
        for (int p = 0; p < n_pos; p++) {
            if (best_p[p] < 0.0) {
                out[p] = nll[p];
                continue;
            }

            double alpha;
            if (alpha_mode_ == 2 && ent) {
                double mo = static_cast<double>(best_ord[p]);
                double center =
                    order_ent_center_ - order_ent_slope_ * (mo - min_order_);
                double sig =
                    1.0 / (1.0 + std::exp(-ent_scale_ * (ent[p] - center)));
                alpha = ent_base_ + ent_range_ * sig;
            } else if (alpha_mode_ == 1 && ent) {
                double sig = 1.0 / (1.0 + std::exp(-ent_scale_ *
                                                    (ent[p] - ent_thresh_)));
                alpha = ent_base_ + ent_range_ * sig;
            } else {
                alpha = fixed_alpha_;
            }

            double mp = std::exp(-nll[p]);
            double mixed;

            if (mixing_fn_ == 0) {
                mixed = (1.0 - alpha) * mp + alpha * best_p[p];
            } else if (mixing_fn_ == 1) {
                constexpr double eps = 1e-7;
                double pm = std::clamp(mp, eps, 1.0 - eps);
                double pn_c = std::clamp(best_p[p], eps, 1.0 - eps);
                double lm = std::log(pm / (1.0 - pm));
                double ln = std::log(pn_c / (1.0 - pn_c));
                double combined = (1.0 - alpha) * lm + alpha * ln;
                mixed = 1.0 / (1.0 + std::exp(-combined));
            } else {
                constexpr double eps = 1e-12;
                double log_mix =
                    (1.0 - alpha) * std::log(std::max(mp, eps)) +
                    alpha * std::log(std::max(best_p[p], eps));
                mixed = std::exp(log_mix);
            }

            out[p] = -std::log(std::max(mixed, 1e-12));
        }

        // --- UPDATE (after scoring — strict causality) ---
        for (int oi = 0; oi < n_orders_; oi++) {
            int order = min_order_ + oi;
            int ctx_w = order - 1;

            for (int p = 0; p < n_pos; p++) {
                if (pos[p] < order) continue;

                uint64_t ch = hash_ctx(pos[p], ctx_w);
                uint64_t ck = ch & bucket_mask_;
                auto tgt = static_cast<uint64_t>(tokens_[pos[p]]);
                uint64_t fk =
                    hash_with_target(ch, tgt, ctx_w) & bucket_mask_;

                ctx_tables_[oi][ck]++;
                full_tables_[oi][fk]++;
            }
        }
    }

   public:
    NGramBlender(int min_order, int max_order, int ngram_buckets, int min_count)
        : min_order_(min_order),
          max_order_(max_order),
          n_orders_(max_order - min_order + 1),
          min_count_(min_count),
          ngram_buckets_(ngram_buckets),
          bucket_mask_(static_cast<uint64_t>(ngram_buckets - 1)),
          alpha_mode_(0),
          fixed_alpha_(0.40),
          ent_base_(0.05),
          ent_range_(0.55),
          ent_scale_(2.0),
          ent_thresh_(4.0),
          order_ent_center_(3.0),
          order_ent_slope_(0.25),
          mixing_fn_(0) {
        ctx_tables_.resize(n_orders_);
        full_tables_.resize(n_orders_);
        for (int i = 0; i < n_orders_; i++) {
            ctx_tables_[i].assign(ngram_buckets, 0);
            full_tables_[i].assign(ngram_buckets, 0);
        }
    }

    void set_tokens(
        nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            tokens) {
        tokens_ = tokens.data();
        n_tokens_ = static_cast<int64_t>(tokens.shape(0));
    }

    void configure_alpha(int mode, double fixed_alpha, double ent_base,
                         double ent_range, double ent_scale, double ent_thresh,
                         double order_ent_center, double order_ent_slope) {
        alpha_mode_ = mode;
        fixed_alpha_ = fixed_alpha;
        ent_base_ = ent_base;
        ent_range_ = ent_range;
        ent_scale_ = ent_scale;
        ent_thresh_ = ent_thresh;
        order_ent_center_ = order_ent_center;
        order_ent_slope_ = order_ent_slope;
    }

    void set_mixing_fn(int fn) { mixing_fn_ = fn; }

    void reset() {
        for (int i = 0; i < n_orders_; i++) {
            std::fill(ctx_tables_[i].begin(), ctx_tables_[i].end(), 0);
            std::fill(full_tables_[i].begin(), full_tables_[i].end(), 0);
        }
    }

    // Process a single stride segment
    nb::ndarray<nb::numpy, double, nb::ndim<1>> process_stride(
        nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            positions,
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            model_nll,
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            entropy) {
        const int n_pos = static_cast<int>(positions.shape(0));
        auto* out = new double[n_pos];

        process_stride_impl(
            positions.data(), n_pos, model_nll.data(),
            (entropy.shape(0) > 0) ? entropy.data() : nullptr, out);

        nb::capsule owner(out, [](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        size_t shape[1] = {static_cast<size_t>(n_pos)};
        return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, 1, shape,
                                                           owner);
    }

    // Process multiple stride segments in one call (amortizes FFI overhead)
    nb::ndarray<nb::numpy, double, nb::ndim<1>> process_batch(
        nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            all_positions,
        nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            segment_lengths,
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            all_model_nll,
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>
            all_entropy) {
        const int n_segs = static_cast<int>(segment_lengths.shape(0));
        const int32_t* seg_lens = segment_lengths.data();
        const int total = static_cast<int>(all_positions.shape(0));
        const double* ent_base_ptr =
            (all_entropy.shape(0) > 0) ? all_entropy.data() : nullptr;

        auto* out = new double[total];

        int offset = 0;
        for (int s = 0; s < n_segs; s++) {
            int len = seg_lens[s];
            process_stride_impl(
                all_positions.data() + offset, len,
                all_model_nll.data() + offset,
                ent_base_ptr ? ent_base_ptr + offset : nullptr,
                out + offset);
            offset += len;
        }

        nb::capsule owner(out, [](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        size_t shape[1] = {static_cast<size_t>(total)};
        return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, 1, shape,
                                                           owner);
    }
};

NB_MODULE(fast_ngram_ext, m) {
    m.doc() = "C++ accelerated n-gram blend for eval_ngram.py";

    nb::class_<NGramBlender>(m, "NGramBlender")
        .def(nb::init<int, int, int, int>(), nb::arg("min_order"),
             nb::arg("max_order"), nb::arg("ngram_buckets"),
             nb::arg("min_count"))
        .def("set_tokens", &NGramBlender::set_tokens, nb::arg("tokens"))
        .def("configure_alpha", &NGramBlender::configure_alpha,
             nb::arg("mode"), nb::arg("fixed_alpha"), nb::arg("ent_base"),
             nb::arg("ent_range"), nb::arg("ent_scale"), nb::arg("ent_thresh"),
             nb::arg("order_ent_center"), nb::arg("order_ent_slope"))
        .def("set_mixing_fn", &NGramBlender::set_mixing_fn, nb::arg("fn"))
        .def("reset", &NGramBlender::reset)
        .def("process_stride", &NGramBlender::process_stride,
             nb::arg("positions"), nb::arg("model_nll"),
             nb::arg("entropy"))
        .def("process_batch", &NGramBlender::process_batch,
             nb::arg("all_positions"), nb::arg("segment_lengths"),
             nb::arg("all_model_nll"), nb::arg("all_entropy"));
}
