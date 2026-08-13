#include "dictionary.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "word_reader_test.h"

using ::testing::InSequence;
using ::testing::Return;

struct DictTestValues {
  std::vector<std::string> words_;
  size_t min_freq_;
  size_t n_lines_;
  size_t total_freqs_;

  std::unordered_map<std::string, size_t> expected_freqs_;
  std::vector<std::string> sorted_seq_;
};

struct DictionaryTestFixture : public testing::TestWithParam<DictTestValues> {};

INSTANTIATE_TEST_SUITE_P(
    DictionaryParameters, DictionaryTestFixture,
    testing::Values(DictTestValues({"foo"}, 0, 0, 1, {{"foo", 1}}, {"foo"}),
                    DictTestValues({WordReader::NEW_LINE, "foo",
                                    WordReader::NEW_LINE, "foo"},
                                   0, 2, 2, {{"foo", 1}}, {"foo"}),
                    DictTestValues({WordReader::NEW_LINE, WordReader::NEW_LINE},
                                   0, 2, 0, {}, {}),
                    DictTestValues({"foo", "foo", "bar", "bar", "bar"}, 3, 0, 3,
                                   {{"bar", 1}}, {"bar"}),
                    DictTestValues({"foo", "foo", "bar", "bar", "bar"}, 2, 0, 5,
                                   {{"bar", 3}, {"foo", 2}}, {"bar", "foo"}),
                    DictTestValues({"foo", "foo", "bar", "bar", "bar", "foo"},
                                   2, 0, 6, {{"bar", 3}, {"foo", 3}},
                                   {"bar", "foo"})));

TEST_P(DictionaryTestFixture, ParameterChecking) {
  const DictTestValues& param = GetParam();
  // given
  MockWordReader* mock_reader = new MockWordReader();
  {
    InSequence seq;
    for (const std::string& word : param.words_) {
      EXPECT_CALL(*mock_reader, Next())
          .WillOnce(Return(std::optional<std::string>{word}))
          .RetiresOnSaturation();
    }
    EXPECT_CALL(*mock_reader, Next())
        .WillOnce(Return(std::optional<std::string>{}));
  }

  // when
  Dictionary dictionary(kUnexistPath, param.min_freq_,
                        MockReaderFactory(mock_reader));
  // then
  EXPECT_EQ(dictionary.VocabSize(),
            param.expected_freqs_.size() + 1);  // along with NEW_LINE
  EXPECT_EQ(dictionary.NumTotalFreqs(), param.total_freqs_);

  EXPECT_EQ(dictionary.WordFreq(0), param.n_lines_);
  for (int i = 1, ss = std::ssize(param.sorted_seq_); i < ss; ++i) {
    const W2VType::WordIndexT w_id = static_cast<W2VType::WordIndexT>(i);
    const std::string& word = param.sorted_seq_.at(i - 1);
    EXPECT_EQ(*dictionary.GetIndex(word), w_id);
    EXPECT_EQ(dictionary.WordFreq(w_id), param.expected_freqs_.at(word));
    EXPECT_EQ(dictionary.Word(w_id), word);
  }
}
