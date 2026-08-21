#include "word_layer.h"

#include <iomanip>
#include <memory>
#include <random>
#include <ranges>
#include <stdexcept>

WordLayer::WordLayer(const Dictionary& dictionary, const size_t vector_size)
    : dictionary_(dictionary), buffer_(dictionary.VocabSize(), vector_size) {
  std::mt19937 gen;
  std::uniform_real_distribution<W2VType::RealT> dist(-0.5, 0.5);
  buffer_.FillAll(
      [&](W2VType::RealT /*unused*/) { return dist(gen) / vector_size; });
}

WordLayer::WordLayer(WordLayer&& another)
    : dictionary_(another.dictionary_), buffer_(std::move(another.buffer_)) {}

size_t WordLayer::VectorSize() const { return buffer_.VecSize(); }

std::pair<W2VType::RealT*, W2VType::RealT*> WordLayer::WordVectorIterPair(
    W2VType::WordIndexT w_id) const {
  return buffer_.LayerIterPair(w_id);
}

WordLayer::PairAccessVectorOutput::PairAccessVectorOutput(
    WordLayer::PairAccessVectorOutput::ItFuncT it_func)
    : it_func_(std::move(it_func)) {}

void WordLayer::PairAccessVectorOutput::WriteVector(
    std::ofstream& fstr, const W2VType::WordIndexT w_id) const {
  const auto pair_it = it_func_(w_id);
  for (auto it = pair_it.first; it != pair_it.second; ++it) {
    WriteElement(fstr, it);
  }
}

WordLayer::TextVectorOutput::TextVectorOutput(
    WordLayer::PairAccessVectorOutput::ItFuncT it_func)
    : WordLayer::PairAccessVectorOutput::PairAccessVectorOutput(it_func) {}

void WordLayer::TextVectorOutput::WriteElement(
    std::ofstream& fstr, const W2VType::RealT* p_elem) const {
  fstr << std::setw(6) << std::setprecision(3) << *p_elem;
}

WordLayer::BinaryVectorOutput::BinaryVectorOutput(
    WordLayer::PairAccessVectorOutput::ItFuncT it_func)
    : WordLayer::PairAccessVectorOutput::PairAccessVectorOutput(it_func) {}

void WordLayer::BinaryVectorOutput::WriteElement(
    std::ofstream& fstr, const W2VType::RealT* p_elem) const {
  fstr.write(reinterpret_cast<const char*>(p_elem),
             sizeof(const W2VType::RealT));
}

WordLayer::ClusterVectorOutput::ClusterVectorOutput(
    std::vector<size_t> cluster_vec)
    : cluster_vec_(std::move(cluster_vec)) {}

void WordLayer::ClusterVectorOutput::WriteVector(
    std::ofstream& fstr, const W2VType::WordIndexT w_id) const {
  if (w_id < cluster_vec_.size()) {
    fstr << cluster_vec_.at(w_id);
  }
}

std::vector<size_t> WordLayer::KMeanCluster(const size_t n_clusters) const {
  const size_t size = dictionary_.VocabSize();
  std::vector<size_t> clustered_vec(size);
  std::iota(clustered_vec.begin(), clustered_vec.end(), 0ULL);
  for (size_t i = 1; i < clustered_vec.size(); ++i) {
    clustered_vec[i] %= n_clusters;
  }
  for (size_t i = 0; i < kCluseringIter; ++i) {
    KMeanCluster(n_clusters, clustered_vec);
  }
  return clustered_vec;
}

void WordLayer::KMeanCluster(const size_t n_clusters,
                             std::vector<size_t>& clustered_vec) const {
  const size_t vec_size = buffer_.VecSize();
  const size_t vocab_size = dictionary_.VocabSize();
  std::vector<std::vector<W2VType::RealT>> avg_vecs(
      n_clusters, std::vector<W2VType::RealT>(vec_size));
  std::vector<size_t> counts(n_clusters, 1ULL);  // to avoid 'divide by zero'
  for (size_t i = 1; i < vocab_size; ++i) {      // skip 0
    const auto cl_id = clustered_vec.at(i);
    auto& avg_v = avg_vecs.at(cl_id);
    const auto word_vec_it = buffer_.LayerIterPair(i);
    std::transform(word_vec_it.first, word_vec_it.second, avg_v.cbegin(),
                   avg_v.begin(), std::plus<W2VType::RealT>());
    ++counts[cl_id];
  }
  for (size_t i = 0; i < n_clusters; ++i) {
    const auto cnt = counts.at(i);
    auto& avg_v = avg_vecs.at(i);
    std::transform(avg_v.cbegin(), avg_v.cend(), avg_v.begin(),
                   [cnt](const auto sum) { return sum / cnt; });
    const auto norm = std::sqrt(
        std::inner_product(avg_v.cbegin(), avg_v.cend(), avg_v.cbegin(), 0.0));

    std::transform(avg_v.cbegin(), avg_v.cend(), avg_v.begin(),
                   [norm](const auto sum) { return sum / norm; });
  }
  for (size_t i = 1; i < vocab_size; ++i) {  // skip 0
    const auto word_vec_it = buffer_.LayerIterPair(i);
    std::vector<W2VType::RealT> dot_vals(n_clusters);

    for (size_t ci = 0; ci < n_clusters; ++ci) {
      auto& avg_v = avg_vecs.at(ci);
      dot_vals[ci] = std::inner_product(word_vec_it.first, word_vec_it.second,
                                        avg_v.cbegin(), 0.0);
    }
    clustered_vec[i] =
        std::distance(dot_vals.cbegin(),
                      std::max_element(dot_vals.cbegin(), dot_vals.cend()));
  }
}

std::unique_ptr<WordLayer::WordVectorOutput> WordLayer::VectorOutputFactory(
    size_t n_classes, bool bin_or_not) const {
  if (n_classes == 0) {
    auto fun = std::bind(&Aligned2DBuffer<W2VType::RealT>::LayerIterPair,
                         &this->buffer_, std::placeholders::_1);
    if (bin_or_not) {
      return std::make_unique<WordLayer::BinaryVectorOutput>(fun);
    } else {
      return std::make_unique<WordLayer::TextVectorOutput>(fun);
    }
  } else {
    return std::make_unique<WordLayer::ClusterVectorOutput>(
        KMeanCluster(n_classes));
  }
}

void WordLayer::SaveVectors(const std::filesystem::path& file_path,
                            size_t n_classes, bool bin_or_not) {
  std::ofstream v_ostr(file_path, std::ios::binary);
  if (!v_ostr.is_open()) {
    throw std::runtime_error("Can't open an output vector file");
  }

  const auto w_v_out = VectorOutputFactory(n_classes, bin_or_not);

  // skip 0, new-line
  for (size_t i = 1, v_s = dictionary_.VocabSize(); i < v_s; ++i) {
    v_ostr << *dictionary_.Word(i) << " ";
    w_v_out->WriteVector(v_ostr, i);
    v_ostr << "\n";
  }

  v_ostr.close();
}
