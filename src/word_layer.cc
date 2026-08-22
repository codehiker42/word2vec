#include "word_layer.h"

#include <execution>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
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

namespace {

void CalcCentroids(
    Aligned2DBuffer<W2VType::RealT>& centroids,
    const std::vector<std::vector<W2VType::WordIndexT>>& cluster_vec,
    const std::vector<size_t>& cluster_ids,
    const Aligned2DBuffer<W2VType::RealT>& word_buffer) {
  std::for_each(
      std::execution::par_unseq, cluster_ids.begin(), cluster_ids.end(),
      [&](const size_t ci) {
        const std::vector<W2VType::WordIndexT>& a_cluster = cluster_vec.at(ci);
        if (a_cluster.empty()) {
          return;
        }
        auto cent_vec_it = centroids.LayerIterPair(ci);
        for (W2VType::WordIndexT w_id : a_cluster) {
          const auto word_vec_it = word_buffer.LayerIterPair(w_id);
          std::transform(std::execution::seq, word_vec_it.first,
                         word_vec_it.second, cent_vec_it.first,
                         cent_vec_it.first, std::plus<W2VType::RealT>());
        }
        const size_t cluster_size = a_cluster.size();
        const auto norm = std::sqrt(std::reduce(
            std::execution::seq, cent_vec_it.first, cent_vec_it.second, 0.0,
            [cluster_size](const auto reduced, const auto elem_sum) {
              const auto avg_elem = elem_sum / cluster_size;
              return avg_elem * avg_elem + reduced;
            }));
        std::transform(std::execution::seq, cent_vec_it.first,
                       cent_vec_it.second, cent_vec_it.first,
                       [cluster_size, norm](const auto elem_sum) {
                         return elem_sum / cluster_size / norm;
                       });
      });
}

void Cluster(std::vector<std::vector<W2VType::WordIndexT>>& cluster_vec,
             const std::vector<size_t>& cluster_ids,
             const std::vector<W2VType::WordIndexT>& w_id_vecs,
             const Aligned2DBuffer<W2VType::RealT>& word_buffer) {
  const size_t vocab_size = w_id_vecs.size();
  const size_t n_clusters = cluster_ids.size();

  Aligned2DBuffer<W2VType::RealT> centroids(n_clusters, word_buffer.VecSize());
  CalcCentroids(centroids, cluster_vec, cluster_ids, word_buffer);

  std::vector<W2VType::WordIndexT> cluster_id_result(vocab_size);
  std::for_each(std::execution::par_unseq, std::next(w_id_vecs.begin()),
                w_id_vecs.end(), [&](const auto w_id) {
                  const auto word_vec_it = word_buffer.LayerIterPair(w_id);
                  std::vector<W2VType::RealT> dot_vals(n_clusters);

                  for (size_t ci = 0; ci < n_clusters; ++ci) {
                    auto norm_vec_it = centroids.LayerIterPair(ci);
                    dot_vals[ci] = std::transform_reduce(
                        std::execution::seq, word_vec_it.first,
                        word_vec_it.second, norm_vec_it.first, 0.0);
                  }

                  cluster_id_result[w_id] = std::distance(
                      dot_vals.cbegin(),
                      std::max_element(dot_vals.cbegin(), dot_vals.cend()));
                });

  std::vector<std::vector<W2VType::WordIndexT>> result(n_clusters);
  for (W2VType::WordIndexT w_id = 1; w_id < vocab_size; ++w_id) {
    result[cluster_id_result.at(w_id)].emplace_back(w_id);
  }
  std::swap(result, cluster_vec);
}

}  // unnamed namespace

std::vector<size_t> WordLayer::KMeanCluster(const size_t n_clusters) const {
  const size_t vocab_size = dictionary_.VocabSize();
  std::vector<std::vector<W2VType::WordIndexT>> cluster_vec(n_clusters);
  for (size_t i = 1; i < vocab_size; ++i) {  // initialisation, skip 0
    cluster_vec[i % n_clusters].emplace_back(i);
  }
  std::vector<size_t> cluster_ids(n_clusters);  // const
  std::iota(cluster_ids.begin(), cluster_ids.end(), 0ULL);
  std::vector<W2VType::WordIndexT> w_id_vecs(vocab_size);  // const
  std::iota(w_id_vecs.begin(), w_id_vecs.end(), 0ULL);

  for (size_t i = 0; i < kCluseringIter; ++i) {
    Cluster(cluster_vec, cluster_ids, w_id_vecs, buffer_);
  }

  std::vector<size_t> cluster_result_flattened(vocab_size);

  for (size_t ci = 0, packed_ci = 0; ci < cluster_vec.size(); ++ci) {
    if (cluster_vec.at(ci).empty()) {
      continue;
    }
    for (W2VType::WordIndexT w_id : cluster_vec.at(ci)) {
      cluster_result_flattened[w_id] = packed_ci;
    }
    ++packed_ci;
  }
  return cluster_result_flattened;
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
