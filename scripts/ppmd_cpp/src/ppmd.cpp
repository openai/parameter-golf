// Implementation of PPMDState. See ppmd.hpp for invariants.

#include "ppmd.hpp"

#include "backoff.hpp"
#include "sha256.hpp"
#include "virtual_ppmd.hpp"

#include <algorithm>
#include <stdexcept>

namespace ppmd {

PPMDState::PPMDState(int order) : order_(order) {
    if (order_ < 0) {
        throw std::invalid_argument("PPM order must be non-negative");
    }
    if (order_ > kMaxOrder) {
        throw std::invalid_argument("PPM order > 5 not supported by packed ctx encoding");
    }
}

void PPMDState::update_byte(int b_in) {
    if (b_in < 0 || b_in > 255) {
        throw std::invalid_argument("byte out of range 0..255");
    }
    uint8_t b = static_cast<uint8_t>(b_in);
    int wlen = static_cast<int>(window_.size());
    int kmax = std::min(order_, wlen);
    for (int k = 0; k <= kmax; ++k) {
        uint64_t key = pack_ctx(window_.data() + (wlen - k), k);
        auto& cc = ctx_counts_[key];
        if (cc.counts[b] == 0) {
            cc.unique += 1;
        }
        cc.counts[b] += 1;
        cc.total += 1;
    }
    window_.push_back(b);
    if (static_cast<int>(window_.size()) > order_) {
        window_.erase(window_.begin());
    }
}

void PPMDState::update_bytes(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        update_byte(data[i]);
    }
}

std::array<double, 256> PPMDState::byte_probs() const {
    auto provider = [this](uint64_t key) -> const CtxCounts* {
        auto it = ctx_counts_.find(key);
        if (it == ctx_counts_.end()) return nullptr;
        return &it->second;
    };
    return backoff_byte_probs(provider, window_, order_);
}

double PPMDState::byte_prob(int target_b) const {
    auto provider = [this](uint64_t key) -> const CtxCounts* {
        auto it = ctx_counts_.find(key);
        if (it == ctx_counts_.end()) return nullptr;
        return &it->second;
    };
    return backoff_byte_prob(provider, window_, order_, target_b);
}

double PPMDState::confidence() const {
    int wlen = static_cast<int>(window_.size());
    int kmax = std::min(order_, wlen);
    for (int k = kmax; k >= 0; --k) {
        uint64_t key = pack_ctx(window_.data() + (wlen - k), k);
        auto it = ctx_counts_.find(key);
        if (it == ctx_counts_.end()) continue;
        const CtxCounts& cc = it->second;
        if (cc.total == 0) continue;
        uint32_t mx = 0;
        for (int b = 0; b < 256; ++b) {
            if (cc.counts[b] > mx) mx = cc.counts[b];
        }
        return static_cast<double>(mx) /
               static_cast<double>(cc.total + cc.unique);
    }
    return 0.0;
}

VirtualPPMDState PPMDState::clone_virtual() const {
    return VirtualPPMDState(&ctx_counts_, window_, order_);
}

std::string PPMDState::state_digest() const {
    Sha256 h;
    if (!window_.empty()) {
        h.update(window_.data(), window_.size());
    }
    std::vector<std::pair<std::string, const CtxCounts*>> entries;
    entries.reserve(ctx_counts_.size());
    for (const auto& kv : ctx_counts_) {
        entries.emplace_back(ctx_bytes_from_key(kv.first), &kv.second);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (const auto& e : entries) {
        const std::string& ctx = e.first;
        uint16_t len = static_cast<uint16_t>(ctx.size());
        uint8_t lenbuf[2] = {static_cast<uint8_t>(len & 0xFF),
                             static_cast<uint8_t>((len >> 8) & 0xFF)};
        h.update(lenbuf, 2);
        if (!ctx.empty()) {
            h.update(reinterpret_cast<const uint8_t*>(ctx.data()), ctx.size());
        }
        const CtxCounts* cc = e.second;
        for (int b = 0; b < 256; ++b) {
            if (cc->counts[b] == 0) continue;
            uint8_t bb = static_cast<uint8_t>(b);
            h.update(&bb, 1);
            uint64_t cnt = static_cast<uint64_t>(cc->counts[b]);
            uint8_t cntbuf[8];
            for (int i = 0; i < 8; ++i) {
                cntbuf[i] = static_cast<uint8_t>((cnt >> (i * 8)) & 0xFF);
            }
            h.update(cntbuf, 8);
        }
    }
    return h.hexdigest();
}

}  // namespace ppmd
