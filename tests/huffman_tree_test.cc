#include "huffman_tree.h"

#include <gtest/gtest.h>

#include <array>

TEST(HuffmanTreeTest, InValidArg) {
  EXPECT_THROW(HuffmanTree({1}), std::invalid_argument);
  EXPECT_THROW(HuffmanTree({2, 1}), std::invalid_argument);
}

TEST(HuffmanTreeTest, TwoNodes) {
  // index:0 is ignored
  HuffmanTree huffman_tree({0, 3, 2});

  EXPECT_TRUE(huffman_tree.PathsFromRoot(0).empty());  // index:0
  EXPECT_TRUE(huffman_tree.PathsFromRoot(3).empty());  // index:3, out of bound

  const std::vector<HuffmanTree::Path>& most_freq =
      huffman_tree.PathsFromRoot(1);
  const std::vector<HuffmanTree::Path>& least_freq =
      huffman_tree.PathsFromRoot(2);

  EXPECT_TRUE(most_freq.size() == 1);
  EXPECT_TRUE(least_freq.size() == 1);

  EXPECT_TRUE(most_freq.at(0).bin_label_ == 1);
  EXPECT_TRUE(least_freq.at(0).bin_label_ == 0);
  EXPECT_TRUE(most_freq.at(0).subtree_ == least_freq.at(0).subtree_);
}

struct HuffmanSubTreeDesc {
  std::vector<size_t> freqs_;
  std::vector<HuffmanTree::SubTreeIndexT> subtree_node_;
  std::vector<HuffmanTree::SubTreeIndexT> parent_of_word_;
  std::vector<bool> subtree_node_label_;
  std::vector<bool> word_node_label_;
};

class HuffmanTreeFixtureTest
    : public testing::TestWithParam<HuffmanSubTreeDesc> {
 protected:
  void SetUp() override { const HuffmanSubTreeDesc& desc = GetParam(); }

  // Per-test teardown (optional)
  void TearDown() override {}
};

INSTANTIATE_TEST_SUITE_P(
    SubTreeDescParameters, HuffmanTreeFixtureTest,
    testing::Values(
        HuffmanSubTreeDesc({0, 3, 2}, {0}, {0, 0, 0}, {0}, {0, 1, 0}),
        HuffmanSubTreeDesc({0, 3, 3}, {0}, {0, 0, 0}, {0}, {0, 1, 0}),
        HuffmanSubTreeDesc({0, 5, 4, 3}, {1, 0}, {0, 1, 0, 0}, {1, 0},
                           {0, 0, 1, 0}),
        HuffmanSubTreeDesc({0, 7, 4, 3}, {1, 0}, {0, 1, 0, 0}, {0, 1},
                           {0, 1, 1, 0}),
        HuffmanSubTreeDesc({0, 128, 64, 32, 16, 8, 4, 2, 1}, {1, 2, 3, 4, 5, 6},
                           {0, 6, 5, 4, 3, 2, 1, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 1, 1, 1, 1, 1, 1, 1, 0}),
        HuffmanSubTreeDesc({0, 96, 48, 24, 12, 6, 3, 2, 2}, {1, 2, 3, 4, 5, 6},
                           {0, 6, 5, 4, 3, 2, 1, 0, 0}, {1, 1, 1, 1, 1, 1},
                           {0, 0, 0, 0, 0, 0, 0, 1, 0})));

TEST_P(HuffmanTreeFixtureTest, SubTreeNodeTraversing) {
  const HuffmanSubTreeDesc& desc = GetParam();

  HuffmanTree hufffman_tree(desc.freqs_);

  for (auto i = 1; i < desc.freqs_.size(); ++i) {
    const auto& paths = hufffman_tree.PathsFromRoot(i);
    ASSERT_FALSE(paths.empty());
    ASSERT_EQ(paths.back().bin_label_, desc.word_node_label_.at(i));
    ASSERT_EQ(paths.back().subtree_, desc.parent_of_word_.at(i));

    auto subtree_id = desc.parent_of_word_.at(i);
    for (int ss = std::ssize(paths), j = ss - 2; j >= 0; --j) {
      ASSERT_EQ(paths.at(j).bin_label_,
                desc.subtree_node_label_.at(subtree_id));
      ASSERT_EQ(paths.at(j).subtree_, desc.subtree_node_.at(subtree_id));
      subtree_id = desc.subtree_node_.at(subtree_id);
    }
    ASSERT_EQ(subtree_id, desc.freqs_.size() - 3);  // root
  }
}