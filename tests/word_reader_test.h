#ifndef _WORD_READER_TEST_H
#define _WORD_READER_TEST_H

#include "word_reader.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

#endif