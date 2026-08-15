#ifndef _WORD_READER_TEST_H
#define _WORD_READER_TEST_H

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "word_reader.h"

const std::filesystem::path kUnexistPath{"UnexistPath/Unknown"};

class MockWordReader : public WordReader {
 public:
  MOCK_METHOD(std::optional<std::string>, Next, (), (override));
};

struct MockReaderFactory {
  MockReaderFactory(MockWordReader* p_mock) : p_mock_(p_mock) {}

  std::unique_ptr<WordReader> operator()(
      const std::filesystem::path& /*unused*/) const {
    return std::unique_ptr<WordReader>(p_mock_);
  }
  MockWordReader* p_mock_;
};

MockWordReader* CreateMockWordReader(const std::vector<std::string>& words) {
  MockWordReader* mock_reader = new MockWordReader();
  {
    ::testing::InSequence seq;
    for (const std::string& word : words) {
      EXPECT_CALL(*mock_reader, Next())
          .WillOnce(::testing::Return(std::optional<std::string>{word}))
          .RetiresOnSaturation();
    }
    EXPECT_CALL(*mock_reader, Next())
        .WillRepeatedly(::testing::Return(std::optional<std::string>{}));
  }
  return mock_reader;
}

MockWordReader* CreateMockWordReader(const std::vector<std::string>& words,
                                     const std::vector<size_t> freqs) {
  size_t size = std::min(words.size(), freqs.size());

  MockWordReader* mock_reader = new MockWordReader();
  {
    ::testing::InSequence seq;
    for (auto i = 0; i < size; ++i) {
      const std::string& word = words.at(i);
      for (auto j = 0; j < freqs.at(i); ++j) {
        EXPECT_CALL(*mock_reader, Next())
            .WillOnce(::testing::Return(std::optional<std::string>{word}))
            .RetiresOnSaturation();
      }
    }
    EXPECT_CALL(*mock_reader, Next())
        .WillRepeatedly(::testing::Return(std::optional<std::string>{}));
  }
  return mock_reader;
}

#endif