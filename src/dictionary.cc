#include "dictionary.h"

#include <algorithm>
#include <ranges>
#include <stdexcept>

#include "huffman_tree.h"
#include "unigram_table.h"

namespace {
void ReadStreamAndUpdateFreqs(const std::filesystem::path& file_path,
                              const std::function<std::unique_ptr<WordReader>(
                                  const std::filesystem::path&)>& factory_fn,
                              std::unordered_map<std::string, size_t>& freq_map,
                              size_t& n_new_lines) {
  std::unique_ptr<WordReader> word_reader = factory_fn(file_path);
  std::optional<std::string> word_op = word_reader->Next();
  while (word_op) {
    if (*word_op != WordReader::NEW_LINE) {
      ++freq_map[*word_op];
    } else {
      ++n_new_lines;
    }
    word_op = word_reader->Next();
  }
}

std::vector<std::tuple<std::string, size_t>> SortAscend(
    const std::unordered_map<std::string, size_t>& freq_map,
    std::uint64_t min_freq) {
  auto tuple_iter =
      freq_map | std::views::filter([min_freq](const auto& iter) {
        return iter.second >= min_freq;
      }) |
      std::views::transform(
          [](const auto& iter) -> std::tuple<std::string, size_t> {
            return {iter.first, iter.second};
          }) |
      std::views::common;

  std::vector<std::tuple<std::string, size_t>> tup_vec{tuple_iter.begin(),
                                                       tuple_iter.end()};
  std::ranges::sort(tup_vec, [](const auto& left, const auto& right) -> bool {
    if (std::get<1>(left) == std::get<1>(right)) {
      return std::get<0>(left) < std::get<0>(right);
    } else {
      return std::get<1>(left) > std::get<1>(right);  // std::greater
    }
  });
  return tup_vec;
}

}  // unnamed namespace

Dictionary::Dictionary(const std::filesystem::path& file_path,
                       const size_t min_freq,
                       const std::function<std::unique_ptr<WordReader>(
                           const std::filesystem::path&)>& factory_fn) {
  std::unordered_map<std::string, size_t> freq_map;
  size_t n_new_lines = 0;
  ReadStreamAndUpdateFreqs(file_path, factory_fn, freq_map, n_new_lines);

  std::vector<std::tuple<std::string, size_t>> sorted_word_freq{
      SortAscend(freq_map, min_freq)};
  word_index_map_.reserve(sorted_word_freq.size());
  map_iters_.reserve(sorted_word_freq.size() + 1);
  word_freqs_.reserve(sorted_word_freq.size() + 1);
  map_iters_.emplace_back(word_index_map_.end());  // not used
  word_freqs_.emplace_back(n_new_lines);
  W2VType::WordIndexT index = 1;
  for (const auto& [word, freq] : sorted_word_freq) {
    n_total_words_ += freq;
    auto pair = word_index_map_.emplace(word, index++);
    map_iters_.emplace_back(pair.first);  // will be rewritten
    word_freqs_.emplace_back(freq);
  }
  for (const auto& [word, _freq] : sorted_word_freq) {
    const auto iter = word_index_map_.find(word);
    map_iters_[iter->second] = iter;  // because of rehashing
  }
}

void Dictionary::DumpVocab(const std::filesystem::path& file_path) const {
  std::ofstream ostream(file_path);
  if (!ostream.is_open()) {
    throw std::exception();
  }
  for (size_t i = 0; i < word_freqs_.size(); ++i) {
    ostream << (i != 0 ? map_iters_.at(i)->first : NEW_LINE_SYM) << "\t"
            << word_freqs_.at(i) << "\n";
  }
  ostream.close();  // Always close the file after writing
}

std::optional<W2VType::WordIndexT> Dictionary::GetIndex(
    const std::string& word) const {
  if (word == WordReader::NEW_LINE) {
    return 0;
  }
  const auto iter = word_index_map_.find(word);
  return iter != word_index_map_.end() ? iter->second
                                       : std::optional<W2VType::WordIndexT>{};
}

size_t Dictionary::VocabSize() const { return word_index_map_.size() + 1; }

size_t Dictionary::NumTotalFreqs() const { return n_total_words_; }

size_t Dictionary::WordFreq(W2VType::WordIndexT w_id) const {
  return w_id >= word_freqs_.size() ? 0 : word_freqs_.at(w_id);
}

const std::optional<std::string> Dictionary::Word(
    W2VType::WordIndexT w_id) const {
  if (w_id >= map_iters_.size()) {
    return {};
  }
  return map_iters_.at(w_id)->first;
}

UnigramTable Dictionary::MakeUnigram() const {
  return UnigramTable(word_freqs_);
};

HuffmanTree Dictionary::MakeHuffmanTree() const {
  return HuffmanTree(word_freqs_);
}
