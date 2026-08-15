#include "sentence_reader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <barrier>
#include <future>
#include <numeric>

#include "word_reader_test.h"

struct SentenceValue {
  std::vector<std::string> words_training_;
  std::vector<std::string> words_dict_;
  size_t max_words_sentence_;
  size_t n_lines_;
};

class SentenceReadingTest : public testing::TestWithParam<SentenceValue> {
 protected:
  void SetUp() override {
    const SentenceValue& param = GetParam();
    MockWordReader* mock_for_dict =
        CreateMockWordReader(param.words_dict_, FreqVec(param.words_dict_));
    // min_freq <- 0
    p_dictionary =
        new Dictionary(kUnexistPath, 0, MockReaderFactory(mock_for_dict));
  }

  void TearDown() override {
    if (p_dictionary) {
      delete p_dictionary;
    }
  }

  std::vector<size_t> FreqVec(const std::vector<std::string>& unique_words) {
    std::vector<size_t> freqs(unique_words.size());
    std::iota(freqs.begin(), freqs.end(), 1);
    std::reverse(freqs.begin(), freqs.end());
    return freqs;
  }

  Dictionary* p_dictionary;
};

INSTANTIATE_TEST_SUITE_P(
    ExplicitValues, SentenceReadingTest,
    testing::Values(
        SentenceValue({"zz", WordReader::NEW_LINE, "ef", "gh", "ef", "zz", "ab",
                       WordReader::NEW_LINE, "cd", "gh"},
                      {"ab", "cd", "ef", "gh", "ij"},  // five words voca
                      std::numeric_limits<size_t>::max(), 2),
        SentenceValue({"zz", "ab", "gh", "zz", "ef", "zz", "ab", "cd", "gh",
                       "cd"},
                      {"ab", "cd", "ef", "gh", "ij"},  // five words voca
                      3 /* three words in a sentence */, 3)));

TEST_P(SentenceReadingTest, SentenceWordStreamChecking) {
  const SentenceValue& param = GetParam();
  // given
  MockWordReader* mock_reader = CreateMockWordReader(param.words_training_);

  // when
  const Dictionary& dict = *p_dictionary;
  SentenceReader sentence_reader(kUnexistPath, dict, 0.0 /* zero sample*/,
                                 MockReaderFactory(mock_reader),
                                 param.max_words_sentence_);

  // then
  size_t n_sentences = 0, i_train_word = 0;
  std::vector<W2VType::WordIndexT> sentence = sentence_reader.Next();
  while (!sentence.empty()) {
    ++n_sentences;
    for (size_t i = 0; i < sentence.size(); ++i) {
      auto w_id_op = dict.GetIndex(param.words_training_.at(i_train_word));
      while (!w_id_op || *w_id_op == 0 /* new line*/) {
        w_id_op = dict.GetIndex(param.words_training_.at(++i_train_word));
      }
      EXPECT_EQ(sentence.at(i), *w_id_op);
      ++i_train_word;
    }
    while (i_train_word < param.words_training_.size() &&
           param.words_training_.at(i_train_word) == WordReader::NEW_LINE) {
      ++i_train_word;
    }
    sentence = sentence_reader.Next();
  }
  EXPECT_EQ(n_sentences, param.n_lines_);
}

class MockSentenceReader : public AbstractSentenceReader {
 public:
  MOCK_METHOD(std::vector<W2VType::WordIndexT>, Next, (), (override));
};

std::vector<std::vector<W2VType::WordIndexT>> GenerateFakeSentences(
    const size_t end_seq, const size_t sentence_length) {
  const W2VType::WordIndexT fill = 42;
  std::vector<std::vector<W2VType::WordIndexT>> fake_sentences;
  for (auto i = 1; i <= end_seq; ++i) {
    fake_sentences.emplace_back(sentence_length, fill);
    fake_sentences.back().front() = static_cast<W2VType::WordIndexT>(i);
  }
  return fake_sentences;
}

struct MockSentenceReaderFunctor {
  MockSentenceReaderFunctor(size_t end_seq, size_t sentence_len)
      : end_seq_(end_seq), sentence_len_(sentence_len) {}

  std::unique_ptr<AbstractSentenceReader> operator()(
      const std::filesystem::path& /* unused */, const Dictionary& /* unused */,
      const float /* unused */) const {
    MockSentenceReader* mock_sentence_reader = new MockSentenceReader();
    {
      ::testing::InSequence seq;
      for (auto i = 1; i <= end_seq_; ++i) {
        std::vector<W2VType::WordIndexT> fake_sentence(sentence_len_, 42);
        fake_sentence.front() = static_cast<W2VType::WordIndexT>(i);
        EXPECT_CALL(*mock_sentence_reader, Next())
            .WillOnce(::testing::Return(fake_sentence))
            .RetiresOnSaturation();
      }
      EXPECT_CALL(*mock_sentence_reader, Next())
          .WillRepeatedly(
              ::testing::Return(std::vector<W2VType::WordIndexT>()));
    }
    return std::unique_ptr<MockSentenceReader>(mock_sentence_reader);
  }

  size_t end_seq_;
  size_t sentence_len_;
};

struct BufferedSentenceReaderTestValues {
  size_t sentence_length_;
  size_t n_sentences_;
  size_t n_iterations_;
  size_t n_threads_;
};

class BufferedSentenceReaderTest
    : public testing::TestWithParam<BufferedSentenceReaderTestValues> {};

INSTANTIATE_TEST_SUITE_P(
    FakeSentenceValues, BufferedSentenceReaderTest,
    testing::Values(
        BufferedSentenceReaderTestValues(100, 10, 2, 1 /*single thread */),
        BufferedSentenceReaderTestValues(100, 10, 2, 2),
        BufferedSentenceReaderTestValues(1000, 100, 4, 5 /* five threads */),
        BufferedSentenceReaderTestValues(9999, 1000, 5, 12 /* 12 threads */),
        BufferedSentenceReaderTestValues(99999, 1000, 10, 24 /* 24 threads */)
      ));

TEST_P(BufferedSentenceReaderTest, ConcurrencyVerifying) {
  const BufferedSentenceReaderTestValues& param = GetParam();
  // given
  std::vector<std::vector<W2VType::WordIndexT>> generated_sentences =
      GenerateFakeSentences(param.n_sentences_, param.sentence_length_);

  // when
  const Dictionary fake_dict;
  BufferedDaemonSentenceReader reader(
      param.n_iterations_, param.n_threads_, kUnexistPath, fake_dict,
      0.0 /*sample:0*/,
      MockSentenceReaderFunctor(param.n_sentences_, param.sentence_length_));

  std::barrier sync_point(param.n_threads_);
  auto consumer = [](BufferedDaemonSentenceReader& reader,
                     std::barrier<>& barrier_sync)
      -> std::vector<std::vector<W2VType::WordIndexT>> {
    std::vector<std::vector<W2VType::WordIndexT>> sentences;
    reader.Register();
    barrier_sync.arrive_and_wait();
    std::vector<W2VType::WordIndexT> sentence = reader.Next();
    while (!sentence.empty()) {
      sentences.emplace_back(std::move(sentence));
      sentence = reader.Next();
    }
    return sentences;
  };
  std::vector<std::future<std::vector<std::vector<W2VType::WordIndexT>>>>
      futures;
  for (auto i = 0; i < param.n_threads_; ++i) {
    futures.emplace_back(std::async(std::launch::async, consumer,
                                    std::ref(reader), std::ref(sync_point)));
  }

  // then
  std::vector<W2VType::WordIndexT> sentence_head_ids(param.n_sentences_ + 1);
  for (auto& future : futures) {
    future.wait();
    const std::vector<std::vector<W2VType::WordIndexT>> sentences =
        future.get();

    EXPECT_FALSE(sentences.empty());
    for (const auto& sen : sentences) {
      EXPECT_EQ(sen.size(), param.sentence_length_);
      ++sentence_head_ids[sen.front()];
    }
  }

  for (auto i = 1; i < sentence_head_ids.size(); ++i) {
    EXPECT_EQ(sentence_head_ids.at(i), param.n_iterations_);
  }
}
