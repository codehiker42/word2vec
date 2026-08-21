#include "unigram_table.h"

#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

UnigramTable::UnigramTable(const std::vector<size_t>& freq_vec,
                           const size_t table_size)
    : table_(std::vector<W2VType::WordIndexT>(table_size)) {
  // 0 is reservered for NEW_LINE, therefore at least one word is required
  if (table_size < 1) {
    throw std::invalid_argument("table_size should be at least one");
  }

  if (freq_vec.size() < 2) {
    throw std::invalid_argument("the freq_vec size should be greater than 1");
  }

  const double power = 0.75;
  const double accum_pow =
      std::accumulate(std::next(freq_vec.begin()), freq_vec.end(), 0.0,
                      [power](const double prev, const size_t freq) {
                        return prev + std::pow(freq, power);
                      });
  // The only difference from the original implementation is that this
  // implementation doesn’t count NEW_LINE in the UnigramTable. This provides an
  // equivalent outcome, but not the same as the original.
  double threshold = std::pow(freq_vec.at(1), power) / accum_pow;
  W2VType::WordIndexT word_index = 1;
  for (size_t i = 0; i < table_size; ++i) {
    table_[i] = word_index;
    if (static_cast<double>(i) / table_size > threshold) {
      word_index = std::min(word_index + 1, freq_vec.size() - 1);
      threshold += std::pow(freq_vec.at(word_index), power) / accum_pow;
    }
  }
}

W2VType::WordIndexT UnigramTable::RandomeIndex() const {
  thread_local std::mt19937 gen([] {
    std::random_device rd;
    return std::mt19937::result_type(rd());
  }());

  thread_local std::uniform_int_distribution<size_t> dist(0, table_.size() - 1);

  return table_.at(dist(gen));
}