// Implementation of VirtualPPMDState. Uses shared backoff helpers from
// backoff.hpp.

#include "virtual_ppmd.hpp"

#include "backoff.hpp"

#include <algorithm>
#include <stdexcept>

namespace ppmd {

const CtxCounts* VirtualPPMDState::combined_counts(uint64_t key) const {
    const CtxCounts* base = nullptr;
    if (base_counts_) {
        auto it = base_counts_->find(key);
        if (it != base_counts_->end()) base = &it->second;
    }
    auto it_ov = overlay_counts_.find(key);
    const CtxCounts* overlay =
        (it_ov == overlay_counts_.end()) ? nullptr : &it_ov->second;
    if (base == nullptr && overlay == nullptr) return nullptr;
    if (base == nullptr) return overlay;
    if (overlay == nullptr) return base;
    // Both present: synthesize merged counts in a thread-local scratch slot.
    // The backoff loop consumes the pointer immediately, single-threaded.
    thread_local CtxCounts merged;
    merged = *base;
    for (int b = 0; b < 256; ++b) {
        uint32_t add = overlay->counts[b];
        if (add == 0) continue;
        if (merged.counts[b] == 0) merged.unique += 1;
        merged.counts[b] += add;
        merged.total += add;
    }
    return &merged;
}

std::array<double, 256> VirtualPPMDState::byte_probs() const {
    auto provider = [this](uint64_t key) -> const CtxCounts* {
        return combined_counts(key);
    };
    return backoff_byte_probs(provider, window_, order_);
}

double VirtualPPMDState::byte_prob(int target_b) const {
    auto provider = [this](uint64_t key) -> const CtxCounts* {
        return combined_counts(key);
    };
    return backoff_byte_prob(provider, window_, order_, target_b);
}

VirtualPPMDState VirtualPPMDState::fork_and_update(int b_in) const {
    if (b_in < 0 || b_in > 255) {
        throw std::invalid_argument("byte out of range 0..255");
    }
    uint8_t b = static_cast<uint8_t>(b_in);
    VirtualPPMDState child(*this);  // shallow copy: same base ptr; copy overlay+window
    int wlen = static_cast<int>(child.window_.size());
    int kmax = std::min(child.order_, wlen);
    for (int k = 0; k <= kmax; ++k) {
        uint64_t key = pack_ctx(child.window_.data() + (wlen - k), k);
        // Python adds increments to overlay only; combined_counts sums
        // base+overlay at lookup time. Mirror exactly.
        auto& cc = child.overlay_counts_[key];
        if (cc.counts[b] == 0) cc.unique += 1;
        cc.counts[b] += 1;
        cc.total += 1;
    }
    child.window_.push_back(b);
    if (static_cast<int>(child.window_.size()) > child.order_) {
        child.window_.erase(child.window_.begin());
    }
    return child;
}

void VirtualPPMDState::each_context(
    std::function<void(uint64_t, const CtxCounts&)> fn) const {
    // First pass: all keys from base_counts_ (merged with overlay if present).
    if (base_counts_) {
        for (const auto& kv : *base_counts_) {
            const CtxCounts* cc = combined_counts(kv.first);
            if (cc) fn(kv.first, *cc);
        }
    }
    // Second pass: overlay-only keys (not in base).
    for (const auto& kv : overlay_counts_) {
        if (base_counts_ &&
            base_counts_->find(kv.first) != base_counts_->end()) continue;
        const CtxCounts* cc = combined_counts(kv.first);
        if (cc) fn(kv.first, *cc);
    }
}

}  // namespace ppmd
