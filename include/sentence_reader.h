#ifndef _SENTENCE_READER_H
#define _SENTENCE_READER_H

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "dictionary.h"
#include "global_constants.h"
#include "w2v_type.h"
#include "word_reader.h"

class AbstractSentenceReader {
 public:
  using FactoryFnT = std::function<std::unique_ptr<AbstractSentenceReader>(
      const std::filesystem::path&, const Dictionary&, const float)>;
  virtual ~AbstractSentenceReader() = default;
  virtual std::vector<W2VType::WordIndexT> Next() = 0;
};

class SentenceReader : public AbstractSentenceReader {
 public:
  SentenceReader(
      const std::filesystem::path& train_file, const Dictionary& dictionary,
      const float sample,
      WordReader::FactoryFnT&& factory_fn = WordStreamReaderFunctor(),
      const size_t n_words_sen_limit = 1000);

  ~SentenceReader() = default;

  std::vector<W2VType::WordIndexT> Next();

 protected:
  bool SampleOut(W2VType::WordIndexT w_id) const;

 private:
  const Dictionary& dictionary_;
  std::unique_ptr<WordReader> word_reader_;
  const float sample_;
  size_t n_words_sen_limit_;
};

struct DefaultSentenceReaderFunctor {
  DefaultSentenceReaderFunctor() = default;
  std::unique_ptr<AbstractSentenceReader> operator()(
      const std::filesystem::path& train_file, const Dictionary& dictionary,
      const float sample) const {
    return std::make_unique<SentenceReader>(train_file, dictionary, sample);
  }
};

class BufferedDaemonSentenceReader : public AbstractSentenceReader {
 public:
  BufferedDaemonSentenceReader() = delete;
  BufferedDaemonSentenceReader(const std::size_t n_iteration,
                               const size_t n_client_threads,
                               std::filesystem::path train_file,
                               const Dictionary& dictionary, const float sample,
                               AbstractSentenceReader::FactoryFnT factory =
                                   DefaultSentenceReaderFunctor());
  ~BufferedDaemonSentenceReader();

  bool Register();

  std::vector<W2VType::WordIndexT> Next() override;

 protected:
  void WaitIfNotActivated();
  void Fetch();

 private:
  struct alignas(kCacheLineSize) RingCell {
    RingCell(size_t seq);
    ~RingCell() = default;
    std::atomic<size_t> seq_;
    std::vector<W2VType::WordIndexT> sentence_;
  };

  struct alignas(kCacheLineSize) RingBuffer {
    RingBuffer();
    ~RingBuffer();
    alignas(kCacheLineSize) std::atomic<size_t> reader_index_;
    alignas(kCacheLineSize) std::atomic<size_t> writer_index_;
    RingCell* p_ring_cell_;
  };
  size_t iter_idx_{0};
  size_t n_iteration_;

  std::filesystem::path train_file_;
  const Dictionary& dictionary_;
  const float sample_;
  SentenceReader::FactoryFnT sentence_reader_factory_;
  std::unique_ptr<AbstractSentenceReader> p_sen_reader_;

  size_t n_ring_buffers_;
  std::vector<RingBuffer> ring_buffer_headers_;
  std::unordered_map<std::thread::id, size_t> thread_seq_map_;
  std::mutex lock_seq_map_;

  alignas(kCacheLineSize) std::atomic<bool> finished_to_read_{false};
  alignas(kCacheLineSize) std::atomic<bool> activated_{false};

  std::thread fetch_thread_;
};

#endif