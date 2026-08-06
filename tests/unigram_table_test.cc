#include "unigram_table.h"

#include <gtest/gtest.h>

#include <future>
#include <thread>

TEST(UnigramContractTest, InValidArg) {
  for (size_t table_size = 0; table_size < 1; ++table_size) {
    EXPECT_THROW(UnigramTable({3, 2, 2}, table_size), std::invalid_argument)
        << "the table Size should be >= 1";
  }

  EXPECT_THROW(UnigramTable({}, 2), std::invalid_argument);
  EXPECT_THROW(UnigramTable({100}, 2), std::invalid_argument);
}

TEST(UnigramContractTest, ConcurrentAccessingAndNeverReturningZero) {
  const size_t n_threads = 20;
  const size_t table_size = 3;
  const size_t vocab_size = 2;
  const size_t repeats_each_thread = 100;

  UnigramTable unigram(std::vector<size_t>(vocab_size, 3 /*any*/), table_size);

  for (auto i = 0; i < n_threads; ++i) {
    std::future<void> _fut = std::async(std::launch::async, [&]() {
      for (int repeat = 0; repeat < repeats_each_thread; ++repeat) {
        EXPECT_NE(unigram.RandomeIndex(), 0);
      }
    });
  }
}

TEST(UnigramContractTest, DorminantMostFreq) {
  const size_t table_size = 10;

  // 100^0.75 = 31.62, freq_sum (31.62 + 1 + 1 + 1)
  /// 31.62 / 34.62 = 0.91, 0.91 > 9/10
  UnigramTable unigram(std::vector<size_t>{0, 100, 1, 1, 1}, table_size);

  for (auto r = 0; r < 10000; ++r) {
    EXPECT_EQ(unigram.RandomeIndex(), 1);
  }
}

TEST(UnigramContractTest, AllFreqsAreIdentical) {
  const size_t vocab_size = 4;
  const size_t table_size = 1000;
  const size_t repeat = 10000;
  // index:0 is not used
  // every word is equally presented, 100
  UnigramTable unigram(std::vector<size_t>(vocab_size + 1, 100), table_size);

  std::vector<size_t> freq(vocab_size + 1);
  for (auto r = 0; r < repeat; ++r) {
    ++freq[unigram.RandomeIndex()];
  }

  // then, check every word index appears 25% with 2% margin
  for (auto i = 1; i < vocab_size; ++i) {
    EXPECT_NEAR(static_cast<double>(freq.at(i)) / repeat, 0.25, 0.02);
  }
}
