#ifndef _WORD_LAYER_H
#define _WORD_LAYER_H

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include "buffer_array.h"
#include "dictionary.h"

class WordLayer {
 public:
  inline constexpr static size_t kCluseringIter = 10;

  WordLayer() = delete;
  ~WordLayer() = default;
  WordLayer(const Dictionary& dictionary, const size_t vector_size);

  WordLayer(const WordLayer&) = delete;
  WordLayer& operator=(const WordLayer&) = delete;

  WordLayer(WordLayer&& another);

  std::pair<W2VType::RealT*, W2VType::RealT*> WordVectorIterPair(
      W2VType::WordIndexT w_id) const;

  void SaveVectors(const std::filesystem::path& file_path, size_t n_classes,
                   bool bin_or_not);

  size_t VectorSize() const;

 protected:
  class WordVectorOutput {
   public:
    virtual ~WordVectorOutput() = default;
    virtual void WriteVector(std::ofstream& fstr,
                             const W2VType::WordIndexT w_id) const = 0;
  };

  class PairAccessVectorOutput : public WordVectorOutput {
   public:
    using ItFuncT = std::function<std::pair<W2VType::RealT*, W2VType::RealT*>(
        const W2VType::WordIndexT)>;

    PairAccessVectorOutput(ItFuncT it_func);
    void WriteVector(std::ofstream& fstr, const W2VType::WordIndexT w_id) const;

   protected:
    virtual void WriteElement(std::ofstream& fstr,
                              const W2VType::RealT* p_elem) const = 0;

   private:
    const ItFuncT it_func_;
  };

  class TextVectorOutput : public PairAccessVectorOutput {
   public:
    TextVectorOutput(PairAccessVectorOutput::ItFuncT it_func);

   protected:
    void WriteElement(std::ofstream& fstr, const W2VType::RealT* p_elem) const;
  };

  class BinaryVectorOutput : public PairAccessVectorOutput {
   public:
    BinaryVectorOutput(PairAccessVectorOutput::ItFuncT it_func);

   protected:
    void WriteElement(std::ofstream& fstr, const W2VType::RealT* p_elem) const;
  };

  class ClusterVectorOutput : public WordVectorOutput {
   public:
    ClusterVectorOutput(std::vector<size_t> cluster_vec);
    void WriteVector(std::ofstream& fstr, const W2VType::WordIndexT w_id) const;

   private:
    const std::vector<size_t> cluster_vec_;
  };

  std::vector<size_t> KMeanCluster(const size_t n_clusters) const;

  std::unique_ptr<WordVectorOutput> VectorOutputFactory(size_t n_classes,
                                                        bool bin_or_not) const;

 private:
  const Dictionary& dictionary_;
  Aligned2DBuffer<W2VType::RealT> buffer_;
};

#endif