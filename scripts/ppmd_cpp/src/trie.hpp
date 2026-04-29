// Trie — flat-arena candidate-byte trie for Path A scoring.
//
// Mirrors Python `TrieNode` (scripts/eval_path_a_ppmd.py L76-78). Children are
// stored as a singly-linked list per node, kept in INSERTION ORDER so the
// scorer's DFS visits edges in the same order as the Python reference (this
// preserves bit-exact equivalence of floating-point sums).
//
// Per-node arrays:   first_child[node], last_child[node],
//                    first_terminal[node], last_terminal[node]
// Per-edge arrays:   child_byte[edge], child_node[edge], next_sibling[edge]
// Per-term arrays:   terminal_token_id[term], next_terminal[term]

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace ppmd {

class Trie {
public:
    Trie() {
        // Allocate root node (index 0).
        first_child_.push_back(-1);
        last_child_.push_back(-1);
        first_terminal_.push_back(-1);
        last_terminal_.push_back(-1);
    }

    int32_t num_nodes() const { return static_cast<int32_t>(first_child_.size()); }
    int32_t num_edges() const { return static_cast<int32_t>(child_byte_.size()); }
    int32_t num_terminals() const { return static_cast<int32_t>(terminal_token_id_.size()); }

    // Insert a token's byte string. Multiple tokens may share a terminal node.
    void insert(int32_t token_id, const uint8_t* bytes, std::size_t len);

    // Accessors used by scorer.cpp / module.cpp.
    int32_t first_child(int32_t node) const { return first_child_[node]; }
    int32_t next_sibling(int32_t edge) const { return next_sibling_[edge]; }
    int32_t child_node(int32_t edge) const { return child_node_[edge]; }
    uint8_t child_byte(int32_t edge) const { return child_byte_[edge]; }
    int32_t first_terminal(int32_t node) const { return first_terminal_[node]; }
    int32_t next_terminal(int32_t term) const { return next_terminal_[term]; }
    int32_t terminal_token_id(int32_t term) const { return terminal_token_id_[term]; }

private:
    // Node arrays.
    std::vector<int32_t> first_child_;
    std::vector<int32_t> last_child_;
    std::vector<int32_t> first_terminal_;
    std::vector<int32_t> last_terminal_;
    // Edge arrays.
    std::vector<int32_t> next_sibling_;
    std::vector<int32_t> child_node_;
    std::vector<uint8_t> child_byte_;
    // Terminal arrays.
    std::vector<int32_t> next_terminal_;
    std::vector<int32_t> terminal_token_id_;
};

}  // namespace ppmd
