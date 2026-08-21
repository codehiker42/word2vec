#ifndef _WORD_READER_H
#define _WORD_READER_H

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

class WordReader {
 public:
  using FactoryFnT =
      std::function<std::unique_ptr<WordReader>(const std::filesystem::path&)>;

  inline const static std::string NEW_LINE{"\n"};
  WordReader() = default;
  virtual ~WordReader() = default;
  virtual std::optional<std::string> Next() = 0;
};

class WordStreamReader : public WordReader {
 public:
  explicit WordStreamReader(const std::filesystem::path& file_path);
  explicit WordStreamReader(std::istream& stream);
  virtual ~WordStreamReader() = default;

  virtual std::optional<std::string> Next();

 private:
  std::ifstream fstream_;
  std::istream& stream_;
  std::mutex lock_;
};

struct WordStreamReaderFunctor {
  std::unique_ptr<WordReader> operator()(
      const std::filesystem::path& file_path) const {
    return std::make_unique<WordStreamReader>(file_path);
  }
};

#endif