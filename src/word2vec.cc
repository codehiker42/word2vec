#include "word2vec.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>

#include "dictionary.h"

TrainMethod::TrainMethod(const Dictionary& dict, size_t vec_size,
                         size_t /* unused*/)
    : hidden_layer_(dict.VocabSize(), vec_size), sigmoid_table_(kSigTableSize) {
  for (size_t i = 0; i < kSigTableSize; ++i) {
    const auto e_x =
        std::exp(((static_cast<W2VType::RealT>(i) / kSigTableSize) - 0.5) * 2 *
                 kAbsValueBound);
    sigmoid_table_[i] = e_x / (e_x + 1);
  }
}

std::optional<const W2VType::RealT> TrainMethod::Sigmoid(
    const W2VType::RealT f_val) const {
  const int i = static_cast<int>((f_val + kAbsValueBound) /
                                 (2 * kAbsValueBound) * kSigTableSize);
  return i >= 0 && i < std::ssize(sigmoid_table_)
             ? sigmoid_table_.at(i)
             : std::optional<const W2VType::RealT>{};
}

void TrainMethod::UpdateVector(const W2VType::RealT grad_scalar,
                               W2VType::VecT& v_gradient,
                               W2VType::RealT* hidden_begin,
                               const W2VType::RealT* feat_begin) {
  // update the gradient vector with the hidden vector
  std::transform(v_gradient.cbegin(), v_gradient.cend(), hidden_begin,
                 v_gradient.begin(),
                 [grad_scalar](const auto v_g, const auto v_h) {
                   return v_g + grad_scalar * v_h;
                 });
  // update the hidden vector with a given vector
  std::transform(hidden_begin, hidden_begin + v_gradient.size(), feat_begin,
                 hidden_begin,  // output
                 [grad_scalar](const auto v_h, const auto v_p) {
                   return v_h + grad_scalar * v_p;
                 });
}

const W2VType::VecT HierachicalSoftMax::BackProp(
    W2VType::RealT* begin, W2VType::WordIndexT target_word,
    const W2VType::RealT learning_rate) {
  W2VType::VecT v_gradient(hidden_layer_.VecSize());
  for (const auto path : huffman_tree_.PathsFromRoot(target_word)) {
    const auto hidden_vec_it = hidden_layer_.LayerIterPair(path.subtree_);
    const auto sig_op = Sigmoid(std::inner_product(
        hidden_vec_it.first, hidden_vec_it.second, begin, 0.0));
    if (!sig_op) {
      continue;
    }
    const auto grad_scalar = (1 - path.bin_label_ - *sig_op) * learning_rate;
    UpdateVector(grad_scalar, v_gradient, hidden_vec_it.first, begin);
  }
  return v_gradient;
}

const W2VType::VecT NegativeSampling::BackProp(
    W2VType::RealT* begin, W2VType::WordIndexT target_word,
    const W2VType::RealT learning_rate) {
  W2VType::VecT v_gradient(hidden_layer_.VecSize());
  for (size_t i = 0; i < neg_size_ + 1; ++i) {
    const W2VType::WordIndexT w_id =
        i == 0 ? target_word : unigram_table_.RandomeIndex();
    if (i != 0 && w_id == target_word) {
      continue;
    }
    const auto hidden_vec_it = hidden_layer_.LayerIterPair(w_id);
    const bool label = i == 0;

    const auto sig_op = Sigmoid(std::inner_product(
        hidden_vec_it.first, hidden_vec_it.second, begin, 0.0));
    if (!sig_op) {
      continue;
    }
    const auto grad_scalar = (label - *sig_op) * learning_rate;
    UpdateVector(grad_scalar, v_gradient, hidden_vec_it.first, begin);
  }
  return v_gradient;
}

void ModelArchitecture::UpdateOneWord(W2VType::WordIndexT index,
                                      const W2VType::VecT& v_gradient) {
  auto i_layer_iter = word_layer_.WordVectorIterPair(index);
  std::transform(v_gradient.cbegin(), v_gradient.cend(), i_layer_iter.first,
                 i_layer_iter.first, std::plus<W2VType::RealT>());
}

size_t ContinuousBagOfWords::SizeOfNeighbourWords(
    const W2VType::Element& element) const {
  return std::accumulate(element.neighbours_.begin(), element.neighbours_.end(),
                         0, [](const size_t accum, const auto& it_pair) {
                           return accum +
                                  std::distance(it_pair.first, it_pair.second);
                         });
}

std::unique_ptr<W2VType::RealT> ContinuousBagOfWords::AvgOfNeighbourWords(
    const size_t n_words, const W2VType::Element& element) const {
  std::unique_ptr<W2VType::RealT> v_for_dot(
      CreateBuffer<W2VType::RealT>(word_layer_.VectorSize()));
  // v_for_dot = Sum(v of neighbours)
  for (const auto& neighbour : element.neighbours_) {
    std::for_each(neighbour.first, neighbour.second,
                  [&](W2VType::WordIndexT w_id) {
                    auto i_layer_iter = word_layer_.WordVectorIterPair(w_id);
                    std::transform(i_layer_iter.first, i_layer_iter.second,
                                   v_for_dot.get(), v_for_dot.get(),
                                   std::plus<W2VType::RealT>());
                  });
  }
  // v_for_dot /= n_words
  std::transform(v_for_dot.get(), v_for_dot.get() + word_layer_.VectorSize(),
                 v_for_dot.get(),
                 [n_words](const auto v) { return v / n_words; });

  return v_for_dot;
}

Word2Vec::LearningRate::LearningRate(const W2VType::RealT alpha_from_config,
                                     const size_t n_iterations,
                                     const size_t total_words_freqs)
    : alpha_from_config_(alpha_from_config),
      alpha_denom_(
          static_cast<W2VType::RealT>(n_iterations * total_words_freqs + 1)) {}

W2VType::RealT Word2Vec::LearningRate::Alpha(size_t added_words) {
  auto updated =
      n_trained_words_.fetch_add(added_words, std::memory_order_relaxed);
  auto last_checkin = n_word_last_checkin_.load(std::memory_order_relaxed);
  while (updated - last_checkin > kNumWordsForAlphaUpdating) {
    const auto diff = updated - last_checkin;
    if (n_word_last_checkin_.compare_exchange_weak(last_checkin, updated,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed)) {
      alpha_.store(alpha_from_config_ * std::max(kMinRationAlphaBound,
                                                 (1 - diff / alpha_denom_)),
                   std::memory_order_relaxed);
    }
    updated = n_trained_words_.load(std::memory_order_relaxed);
  }

  return alpha_.load(std::memory_order_relaxed);
}

Word2Vec::Word2Vec(const Dictionary& dictionary) : dictionary_(dictionary) {}

Word2Vec::~Word2Vec() {}

size_t Word2Vec::RandomWindow(const size_t inc_bound) {
  thread_local std::mt19937 gen([] {
    std::random_device rd;
    return std::mt19937::result_type(rd());
  }());
  thread_local std::uniform_int_distribution<size_t> dist(1, inc_bound);

  return dist(gen);
}

const std::vector<W2VType::Element> Word2Vec::GetTrainingElements(
    const std::vector<W2VType::WordIndexT>& sentence,
    const size_t window) const {
  std::vector<W2VType::Element> elems;

  W2VType::SenItT begin = sentence.cbegin();
  for (int i = 0, s = std::ssize(sentence), off = static_cast<int>(window);
       i < s; ++i) {
    elems.emplace_back();
    W2VType::Element& elem = elems.back();
    elem.target_ = std::next(begin, i);
    int left_close = std::max(i - 1, 0);
    if (left_close < i) {
      elem.neighbours_.emplace_back(
          W2VType::SenPairT{std::next(begin, std::max(i - off, 0)),
                            std::next(begin, left_close + 1)});
    }
    int right_open = std::min(i + 1, s - 1);
    if (right_open > i) {
      elem.neighbours_.emplace_back(
          W2VType::SenPairT{std::next(begin, right_open),
                            std::next(begin, std::min(i + off, s - 1) + 1)});
    }
  }
  return elems;
}
