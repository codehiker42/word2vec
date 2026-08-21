#ifndef _DICTIONARY_H
#define _DICTIONARY_H

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "w2v_type.h"
#include "word_reader.h"

class UnigramTable;
class HuffmanTree;

class Dictionary {
 public:
  inline static constexpr std::string NEW_LINE_SYM{"</s>"};

  Dictionary() = default;

  Dictionary(
      const std::filesystem::path& file_path, const size_t min_freq,
      WordReader::FactoryFnT&& factory_fn = WordStreamReaderFunctor());

  std::optional<W2VType::WordIndexT> GetIndex(const std::string& word) const;

  void DumpVocab(const std::filesystem::path& file_path) const;

  size_t VocabSize() const;

  size_t NumTotalFreqs() const;  // not including NEW_LINE

  size_t WordFreq(W2VType::WordIndexT w_id) const;

  const std::optional<std::string> Word(W2VType::WordIndexT w_id) const;

  UnigramTable MakeUnigram() const;

  HuffmanTree MakeHuffmanTree() const;

 private:
  std::unordered_map<std::string, W2VType::WordIndexT> word_index_map_;
  std::vector<
      std::unordered_map<std::string, W2VType::WordIndexT>::const_iterator>
      map_iters_;
  std::vector<size_t> word_freqs_;

  size_t n_total_words_{0};
};

#endif