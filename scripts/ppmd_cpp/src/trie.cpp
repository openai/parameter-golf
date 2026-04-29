// Implementation of Trie. Children are appended (not prepended) so that DFS
// visits siblings in insertion order — matching Python `dict` insertion order.

#include "trie.hpp"

namespace ppmd {

void Trie::insert(int32_t token_id, const uint8_t* bytes, std::size_t len) {
    int32_t node = 0;  // root
    for (std::size_t i = 0; i < len; ++i) {
        uint8_t b = bytes[i];
        // Walk existing siblings looking for byte b.
        int32_t edge = first_child_[node];
        int32_t found = -1;
        while (edge != -1) {
            if (child_byte_[edge] == b) {
                found = edge;
                break;
            }
            edge = next_sibling_[edge];
        }
        if (found == -1) {
            // Create new child node.
            int32_t new_node = static_cast<int32_t>(first_child_.size());
            first_child_.push_back(-1);
            last_child_.push_back(-1);
            first_terminal_.push_back(-1);
            last_terminal_.push_back(-1);
            // Append new edge to current node's child list.
            int32_t new_edge = static_cast<int32_t>(child_byte_.size());
            child_byte_.push_back(b);
            child_node_.push_back(new_node);
            next_sibling_.push_back(-1);
            int32_t last = last_child_[node];
            if (last == -1) {
                first_child_[node] = new_edge;
            } else {
                next_sibling_[last] = new_edge;
            }
            last_child_[node] = new_edge;
            node = new_node;
        } else {
            node = child_node_[found];
        }
    }
    // Append terminal at this node.
    int32_t new_term = static_cast<int32_t>(terminal_token_id_.size());
    terminal_token_id_.push_back(token_id);
    next_terminal_.push_back(-1);
    int32_t last = last_terminal_[node];
    if (last == -1) {
        first_terminal_[node] = new_term;
    } else {
        next_terminal_[last] = new_term;
    }
    last_terminal_[node] = new_term;
}

}  // namespace ppmd
