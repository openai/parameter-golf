// PPMDState — C++ port of Python `PPMDState` in scripts/eval_path_a_ppmd.py.
//
// Numerical contract: bit-for-bit agreement with the Python reference's
// `_ppmd_byte_probs_with_provider` and `_ppmd_byte_prob_with_provider`
// (>= 15 decimals). All probability arithmetic uses `double`.
//
// Context-key encoding: 0..5 bytes packed into a single uint64_t.
//   bits [63..61]  3-bit length (0..5)
//   bits [39..0]   payload, 5x 8-bit slots; slot i = window[len-k+i]
// (Order > 5 is rejected at construction time.)

#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ppmd {

constexpr int kMaxOrder = 5;

inline uint64_t pack_ctx(const uint8_t* bytes, int len) {
    uint64_t key = static_cast<uint64_t>(len) << 61;
    for (int i = 0; i < len; ++i) {
        key |= static_cast<uint64_t>(bytes[i]) << ((4 - i) * 8);
    }
    return key;
}

inline int ctx_len_from_key(uint64_t key) {
    return static_cast<int>(key >> 61);
}

inline std::string ctx_bytes_from_key(uint64_t key) {
    int len = ctx_len_from_key(key);
    std::string out(len, '\0');
    for (int i = 0; i < len; ++i) {
        out[i] = static_cast<char>((key >> ((4 - i) * 8)) & 0xFF);
    }
    return out;
}

struct CtxCounts {
    // Per-byte counts. 0 means absent; we also track total + unique to avoid
    // re-summing on every probe.
    std::array<uint32_t, 256> counts{};
    uint32_t total = 0;
    uint32_t unique = 0;
};

class VirtualPPMDState;

class PPMDState {
public:
    explicit PPMDState(int order = 5);

    int order() const { return order_; }
    const std::vector<uint8_t>& window() const { return window_; }
    const std::unordered_map<uint64_t, CtxCounts>& ctx_counts() const { return ctx_counts_; }

    void update_byte(int b);
    void update_bytes(const uint8_t* data, size_t len);

    // Full normalized 256-byte distribution; matches Python
    // `_ppmd_byte_probs_with_provider`.
    std::array<double, 256> byte_probs() const;
    // Single-byte fast path; matches `_ppmd_byte_prob_with_provider`.
    double byte_prob(int target_b) const;

    double confidence() const;

    VirtualPPMDState clone_virtual() const;

    // SHA-256 hex digest matching Python `state_digest`.
    std::string state_digest() const;

private:
    int order_;
    std::vector<uint8_t> window_;
    std::unordered_map<uint64_t, CtxCounts> ctx_counts_;
};

}  // namespace ppmd
