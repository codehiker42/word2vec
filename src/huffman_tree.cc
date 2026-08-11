#include "huffman_tree.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

HuffmanTree::Path::Path(bool bin_label, SubTreeIndexT subtree)
    : bin_label_(bin_label), subtree_(subtree) {}

HuffmanTree::HuffmanTree(const std::vector<size_t>& word_freqs) {
  if (word_freqs.size() < 3) {
    throw std::invalid_argument(
        "the word_freqs size should be greater than 2.");
  }
  const size_t eff_size = word_freqs.size() - 1;  // without index:0
  const size_t tree_size = 2 * eff_size;
  std::vector<SubTreeIndexT> parents(tree_size - 1);
  std::vector<bool> label(tree_size - 1);
  std::vector<size_t> subtree_freqs(tree_size,
                                    std::numeric_limits<size_t>::max());
  std::copy(word_freqs.begin(), word_freqs.end(), subtree_freqs.begin());

  // The index:0 of word_freqs isn't used
  for (size_t l = eff_size, r = l + 1, i = r; i < tree_size; ++i) {
    const size_t min_idx1 =
        l < 1 ? r++ : (subtree_freqs.at(l) < subtree_freqs.at(r) ? l-- : r++);
    const size_t min_idx2 =
        l < 1 ? r++ : (subtree_freqs.at(l) < subtree_freqs.at(r) ? l-- : r++);
    subtree_freqs[i] = subtree_freqs[min_idx1] + subtree_freqs[min_idx2];
    parents[min_idx1] = i;
    parents[min_idx2] = i;
    label[min_idx2] = true;
  }

  sub_trees_.emplace_back(std::move(std::vector<Path>{}));  // index:0
  for (size_t i = 1, s = word_freqs.size(), root = tree_size - 1; i < s; ++i) {
    std::vector<HuffmanTree::Path> paths;
    auto node_id = i;
    while (node_id != root) {
      bool prev_label = label.at(node_id);
      node_id = parents.at(node_id);
      paths.emplace_back(prev_label, node_id - eff_size - 1);
    }
    std::reverse(paths.begin(), paths.end());
    sub_trees_.emplace_back(std::move(paths));
  }
}

const std::vector<HuffmanTree::Path> HuffmanTree::PathsFromRoot(
    const W2VType::WordIndexT word_index) const {
  return word_index >= sub_trees_.size() ? std::vector<HuffmanTree::Path>{}
                                         : sub_trees_.at(word_index);
}