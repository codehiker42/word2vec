#ifndef _HUFFMAN_TREE_H
#define _HUFFMAN_TREE_H

#include <vector>

#include "w2v_type.h"

class HuffmanTree {
 public:
  using SubTreeIndexT = size_t;
  struct Path {
    Path() = default;
    Path(bool bin_label, SubTreeIndexT subtree);

    bool bin_label_;  // left or right
    SubTreeIndexT subtree_;
  };
  /* The class works under the contract; the argument word_freqs has elements
   * that are sorted in descending order, except index:0, which is for NEW_LINE.
   */
  HuffmanTree(const std::vector<size_t>& word_freqs);

  const std::vector<Path> PathsFromRoot(
      const W2VType::WordIndexT word_index) const;

 private:
  std::vector<std::vector<Path>> sub_trees_;
};

#endif