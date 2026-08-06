#include "word_reader.h"

#include <stdexcept>

namespace {
std::ifstream ToStream(const std::filesystem::path& file_path) {
  std::ifstream word_file(file_path, std::ios::in | std::ios::binary);
  if (!word_file) {
    throw std::runtime_error("Can't access " + file_path.generic_string());
  }
  return std::move(word_file);
}
}  // namespace

WordReader::WordReader(const std::filesystem::path& file_path)
    : fstream_(ToStream(file_path)), stream_(fstream_) {}

WordReader::WordReader(std::istream& stream) : stream_(stream) {}

std::optional<std::string> WordReader::Next() {
  if (!stream_) {
    return {};
  }
  char ch;
  std::string word;
  try {
    std::lock_guard<std::mutex> lock(lock_);
    while (stream_.get(ch)) {
      if (std::isalpha(ch)) {
        word += ch;
      } else if (ch == '\n' && word.empty()) {
        return NEW_LINE;
      } else if (ch == '\n' && !word.empty()) {
        stream_.unget();
        return word;
      } else if (std::isspace(ch) && !word.empty()) {
        break;
      }
    }
    return word.empty() ? std::optional<std::string>{} : word;
  } catch (...) {
    return {};
  }
}
