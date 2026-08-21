#ifndef _WORD2VEC_H
#define _WORD2VEC_H

#include <atomic>
#include <optional>
#include <utility>

#include "buffer_array.h"
#include "dictionary.h"
#include "global_constants.h"
#include "huffman_tree.h"
#include "sentence_reader.h"
#include "unigram_table.h"
#include "w2v_options.h"
#include "word_layer.h"

template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;

class TrainMethod {
 public:
  ~TrainMethod() = default;

  // no virtual, just for hiding
  const W2VType::VecT BackProp(W2VType::RealT* /* unused */,
                               W2VType::WordIndexT /* unused */,
                               const W2VType::RealT /* unused */) {
    return {};
  }

  std::optional<const W2VType::RealT> Sigmoid(const W2VType::RealT val) const;

 protected:
  inline static constexpr size_t kAbsValueBound = 6;
  inline static constexpr size_t kSigTableSize = 1000;

  TrainMethod(const Dictionary& dict, size_t vec_size, size_t dummy);

  void UpdateVector(const W2VType::RealT grad_scalar, W2VType::VecT& v_gradient,
                    W2VType::RealT* hidden_begin,
                    const W2VType::RealT* feat_begin);

  Aligned2DBuffer<W2VType::RealT> hidden_layer_;

 private:
  std::vector<W2VType::RealT> sigmoid_table_;
};

class HierachicalSoftMax : public TrainMethod {
 public:
  HierachicalSoftMax(const Dictionary& dict, size_t vec_size, size_t dummy)
      : TrainMethod(dict, vec_size, dummy),
        huffman_tree_(std::move(dict.MakeHuffmanTree())) {}

  const W2VType::VecT BackProp(W2VType::RealT* begin,
                               W2VType::WordIndexT target_word,
                               const W2VType::RealT learning_rate);

 private:
  const HuffmanTree huffman_tree_;
};

class NegativeSampling : public TrainMethod {
 public:
  NegativeSampling(const Dictionary& dict, size_t vec_size,
                   size_t negative_size)
      : TrainMethod(dict, vec_size, negative_size),
        neg_size_(negative_size),
        unigram_table_(std::move(dict.MakeUnigram())) {}

  const W2VType::VecT BackProp(W2VType::RealT* begin,
                               W2VType::WordIndexT target_word,
                               const W2VType::RealT learning_rate);

 private:
  const size_t neg_size_;
  const UnigramTable unigram_table_;
};

class ModelArchitecture {
 public:
  // no virtual, just for hiding
  template <Derived<TrainMethod> TrainT>
  void SGD(TrainT& train_met, const W2VType::Element& element,
           const W2VType::RealT learning_rate) {}

 protected:
  ModelArchitecture(WordLayer& word_layer) : word_layer_(word_layer) {}

  void UpdateOneWord(W2VType::WordIndexT index,
                     const W2VType::VecT& v_gradient);

  WordLayer& word_layer_;
};

class ContinuousBagOfWords : public ModelArchitecture {
 public:
  ContinuousBagOfWords(WordLayer& word_layer) : ModelArchitecture(word_layer) {}

  template <Derived<TrainMethod> TrainT>
  void SGD(TrainT& train_met, const W2VType::Element& element,
           const W2VType::RealT learning_rate);

 protected:
  size_t SizeOfNeighbourWords(const W2VType::Element& element) const;
  std::unique_ptr<W2VType::RealT> AvgOfNeighbourWords(
      const size_t n_words, const W2VType::Element& element) const;
};

class SkipGram : public ModelArchitecture {
 public:
  SkipGram(WordLayer& word_layer) : ModelArchitecture(word_layer) {}

  template <Derived<TrainMethod> TrainT>
  void SGD(TrainT& train_met, const W2VType::Element& element,
           const W2VType::RealT learning_rate);

 protected:
  template <Derived<TrainMethod> TrainT>
  void TrainNeighbourWord(TrainT& train_met, W2VType::WordIndexT neighbour,
                          W2VType::WordIndexT target,
                          const W2VType::RealT learning_rate);
};

class Word2Vec {
 public:
  inline static constexpr int64_t kNumWordsForAlphaUpdating = 1e4;
  inline static constexpr W2VType::RealT kMinRationAlphaBound = 0.0001;
  Word2Vec(const Dictionary& dictionary);
  virtual ~Word2Vec();

  template <Derived<ModelArchitecture> ModelT, Derived<TrainMethod> TrainT>
  WordLayer Train(const Word2VecOptions& options);

 protected:
  struct LearningRate {
    LearningRate(const W2VType::RealT alpha_from_config,
                 const size_t n_iterations, const size_t total_words_freqs);

    W2VType::RealT Alpha(size_t added_words);

    const W2VType::RealT alpha_from_config_;
    const W2VType::RealT alpha_denom_;
    std::atomic<W2VType::RealT> alpha_;
    std::atomic<int64_t> n_trained_words_{0};
    std::atomic<int64_t> n_word_last_checkin_{0};
  };

  template <Derived<ModelArchitecture> ModelT, Derived<TrainMethod> TrainT>
  void Process(ModelT& model, TrainT& train_met,
               BufferedDaemonSentenceReader& sentence_reader,
               LearningRate& learning_rate, const size_t window_size);

  size_t RandomWindow(const size_t inc_bound);

  const std::vector<W2VType::Element> GetTrainingElements(
      const std::vector<W2VType::WordIndexT>& sentence,
      const size_t window) const;

 private:
  const Dictionary& dictionary_;
};

template <Derived<TrainMethod> TrainT>
void ContinuousBagOfWords::SGD(TrainT& train_met,
                               const W2VType::Element& element,
                               const W2VType::RealT learning_rate) {
  const size_t n_words = SizeOfNeighbourWords(element);
  auto v_for_dot = AvgOfNeighbourWords(n_words, element);

  const auto v_gradient =
      train_met.BackProp(v_for_dot.get(), *element.target_, learning_rate);

  //  Update Input vectors, for neighbour_i,  neighbour_i += v_gradient
  for (const auto& neighbour : element.neighbours_) {
    std::for_each(
        neighbour.first, neighbour.second,
        [&](W2VType::WordIndexT w_id) { UpdateOneWord(w_id, v_gradient); });
  }
}

template <Derived<TrainMethod> TrainT>
void SkipGram::SGD(TrainT& train_met, const W2VType::Element& element,
                   const W2VType::RealT learning_rate) {
  for (const auto& neighbour : element.neighbours_) {
    std::for_each(
        neighbour.first, neighbour.second, [&](W2VType::WordIndexT w_id) {
          TrainNeighbourWord(train_met, w_id, *element.target_, learning_rate);
        });
  }
}

template <Derived<TrainMethod> TrainT>
void SkipGram::TrainNeighbourWord(TrainT& train_met,
                                  W2VType::WordIndexT neighbour,
                                  W2VType::WordIndexT target,
                                  const W2VType::RealT learning_rate) {
  const auto v_gradient = train_met.BackProp(
      word_layer_.WordVectorIterPair(neighbour).first, target, learning_rate);
  UpdateOneWord(neighbour, v_gradient);
}

template <Derived<ModelArchitecture> ModelT, Derived<TrainMethod> TrainT>
WordLayer Word2Vec::Train(const Word2VecOptions& options) {
  WordLayer word_layer(dictionary_, options.LayerSize());
  TrainT train_met(dictionary_, options.LayerSize(),
                   options.NegativeSize());  // TODO
  ModelT model(word_layer);

  LearningRate learning_rate(options.LearningRate(), options.IterationSize(),
                             dictionary_.NumTotalFreqs());

  BufferedDaemonSentenceReader sentence_reader(
      options.IterationSize(), options.ThreadSize(), *options.TrainFileName(),
      dictionary_, options.DownSamplingSize());

  std::vector<std::thread> threads;
  threads.reserve(options.ThreadSize());
  for (auto i = 0; i < options.ThreadSize(); ++i) {
    threads.emplace_back(&Word2Vec::Process<ModelT, TrainT>, this,
                         std::ref(model), std::ref(train_met),
                         std::ref(sentence_reader), std::ref(learning_rate),
                         options.WindowSize());
  }
  for (auto& thread : threads) {
    thread.join();
  }

  return word_layer;
}

template <Derived<ModelArchitecture> ModelT, Derived<TrainMethod> TrainT>
void Word2Vec::Process(ModelT& model, TrainT& train_met,
                       BufferedDaemonSentenceReader& sentence_reader,
                       LearningRate& learning_rate, const size_t window_size) {
  sentence_reader.Register();
  for (auto sentence = sentence_reader.Next(); !sentence.empty();
       sentence = sentence_reader.Next()) {
    W2VType::RealT alpha = learning_rate.Alpha(sentence.size());

    for (const auto& elem :
         GetTrainingElements(sentence, RandomWindow(window_size))) {
      model.SGD(train_met, elem, alpha);
    }
  }
}

#endif