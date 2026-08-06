#ifndef _WORD_READER_H
#define _WORD_READER_H

#include <filesystem>
#include <fstream>
#include <optional>
#include <mutex>
#include <unordered_map>

class WordReader {
 public:
  inline const static std::string NEW_LINE{"\n"};

  WordReader() = delete;
  explicit WordReader(const std::filesystem::path& file_path);
  explicit WordReader(std::istream& stream);
  ~WordReader() = default;

  std::optional<std::string> Next();

 private:
  std::ifstream fstream_;
  std::istream& stream_;
  std::mutex lock_;
};

#endif