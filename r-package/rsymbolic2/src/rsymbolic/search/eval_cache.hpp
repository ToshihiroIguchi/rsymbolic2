// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Duplicate-evaluation cache for the opt-in `eval_cache` option: a memoisation of
// sse_current keyed by the tree's evaluation-relevant content. Header-only, STL-only.
//
// The evaluation-relevant fields per node (postfix order) are: `kind`, then
//   Constant -> the slot `index` AND the bit pattern of `value`,
//   Variable -> the feature `index`,
//   Unary    -> `uop`,
//   Binary   -> `bop`.
// The Constant slot `index` is deliberately included: sse_current re-extracts the
// constants via initial_constants(tree), which writes values[node.index] = node.value,
// and evaluate() then reads constants[node.index] back. With the contiguous slot
// indices the search maintains (reindex_constants after every structural edit) this
// round-trip returns each node's own `value` — but if two Constant nodes ever aliased
// one slot with different values, the LAST node's value would win for BOTH, so the
// slot index IS evaluation-relevant in general. Including it keeps the equivalence
// sound unconditionally and loses no hits on the search's reindexed trees.

// FNV-1a 64-bit hash over the evaluation-relevant node fields, postfix order.
inline std::uint64_t tree_hash(const Tree& tree) {
    std::uint64_t h = 14695981039346656037ULL;  // FNV offset basis
    const auto mix = [&h](const void* p, std::size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            h ^= static_cast<std::uint64_t>(b[i]);
            h *= 1099511628211ULL;  // FNV prime
        }
    };
    for (const Node& node : tree) {
        const std::uint8_t kind = static_cast<std::uint8_t>(node.kind);
        mix(&kind, 1);
        switch (node.kind) {
            case NodeKind::Constant: {
                const std::uint32_t idx = static_cast<std::uint32_t>(node.index);
                mix(&idx, sizeof(idx));
                // Bit pattern, not the double value: NaN payloads hash stably and
                // -0.0 vs 0.0 are (conservatively) distinct — a miss, never a lie.
                std::uint64_t bits;
                std::memcpy(&bits, &node.value, sizeof(bits));
                mix(&bits, sizeof(bits));
                break;
            }
            case NodeKind::Variable: {
                const std::uint32_t idx = static_cast<std::uint32_t>(node.index);
                mix(&idx, sizeof(idx));
                break;
            }
            case NodeKind::Unary: {
                const std::uint8_t op = static_cast<std::uint8_t>(node.uop);
                mix(&op, 1);
                break;
            }
            case NodeKind::Binary: {
                const std::uint8_t op = static_cast<std::uint8_t>(node.bop);
                mix(&op, 1);
                break;
            }
        }
    }
    return h;
}

// Field-wise equality on the same evaluation-relevant fields as tree_hash. Constant
// values compare BITWISE (memcpy to uint64_t), so NaN-valued constants compare equal
// to themselves — deliberately NOT a Node::operator== / IEEE `==`, which would make a
// NaN constant unequal to itself and re-miss forever.
inline bool tree_eval_equal(const Tree& a, const Tree& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Node& x = a[i];
        const Node& y = b[i];
        if (x.kind != y.kind) return false;
        switch (x.kind) {
            case NodeKind::Constant: {
                if (x.index != y.index) return false;
                std::uint64_t xb, yb;
                std::memcpy(&xb, &x.value, sizeof(xb));
                std::memcpy(&yb, &y.value, sizeof(yb));
                if (xb != yb) return false;
                break;
            }
            case NodeKind::Variable:
                if (x.index != y.index) return false;
                break;
            case NodeKind::Unary:
                if (x.uop != y.uop) return false;
                break;
            case NodeKind::Binary:
                if (x.bop != y.bop) return false;
                break;
        }
    }
    return true;
}

// Fixed-slot direct-mapped memo table: slot = key & (slots.size()-1), one entry per
// slot, store() overwrites unconditionally (last write wins — no probing, no eviction
// policy, no growth). Collision safety: a hit requires the slot to be used, the full
// 64-bit key to match, AND tree_eval_equal on the stored tree — so a slot collision or
// even a 64-bit hash collision is reported as a miss and re-evaluated. Collisions can
// only lower the hit rate, never change a returned value. Memory bound: the table is
// sized once at construction (kEvalCacheSlots = 1024 at the use site), each entry
// holding one tree of <= max_nodes (default 30) nodes — a few hundred KB per island,
// island-local (share-nothing), freed with the island.
struct EvalCache {
    struct Entry {
        std::uint64_t key = 0;
        Tree tree;
        double sse = 0.0;
        bool used = false;
    };

    std::vector<Entry> slots;  // empty => cache inactive (default: no allocation)
    std::uint64_t hits = 0, misses = 0;  // maintained by the caller (score_sse)

    EvalCache() = default;
    // `slots_pow2` must be a power of two (the slot mask assumes it); 0 = inactive.
    explicit EvalCache(std::size_t slots_pow2) : slots(slots_pow2) {}

    bool active() const { return !slots.empty(); }

    // Cached SSE for an evaluation-equivalent tree, or nullptr on any mismatch.
    const double* lookup(std::uint64_t key, const Tree& tree) const {
        const Entry& e = slots[key & (slots.size() - 1)];
        if (e.used && e.key == key && tree_eval_equal(e.tree, tree)) return &e.sse;
        return nullptr;
    }

    void store(std::uint64_t key, const Tree& tree, double sse) {
        Entry& e = slots[key & (slots.size() - 1)];
        e.key  = key;
        e.tree = tree;
        e.sse  = sse;
        e.used = true;
    }
};

}  // namespace rsymbolic
