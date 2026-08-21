#include "sentence_reader.h"

#include <chrono>
#include <cmath>
#include <random>

SentenceReader::SentenceReader(const std::filesystem::path& train_file,
                               const Dictionary& dictionary, const float sample,
                               WordReader::FactoryFnT&& factory_fn,
                               const size_t n_words_sen_limit)
    : dictionary_(dictionary),
      word_reader_(factory_fn(train_file)),
      sample_(sample),
      n_words_sen_limit_(n_words_sen_limit) {}

std::vector<W2VType::WordIndexT> SentenceReader::Next() {
  std::vector<W2VType::WordIndexT> sentence;
  auto word_op = word_reader_->Next();
  while (word_op && (*word_op == WordReader::NEW_LINE ||
                     !dictionary_.GetIndex(*word_op))) {
    word_op = word_reader_->Next();
  }
  while (word_op && *word_op != WordReader::NEW_LINE &&
         sentence.size() < n_words_sen_limit_) {
    auto index_op = dictionary_.GetIndex(*word_op);
    if (index_op && !SampleOut(*index_op)) {
      sentence.push_back(*index_op);
    }
    if (sentence.size() < n_words_sen_limit_) {
      word_op = word_reader_->Next();
    }
  }
  return sentence;
}

bool SentenceReader::SampleOut(W2VType::WordIndexT w_id) const {
  if (sample_ == 0.0) {
    return false;
  }
  thread_local std::mt19937 gen([] {
    std::random_device rd;
    return std::mt19937::result_type(rd());
  }());
  thread_local std::uniform_real_distribution<float> dist(0, 1);

  const auto freq = dictionary_.WordFreq(w_id);
  const auto total_freqs = dictionary_.NumTotalFreqs();

  const float normalised_freq =
      (std::sqrt(freq / (sample_ * total_freqs)) + 1) *
      (sample_ * total_freqs) / freq;

  return normalised_freq < dist(gen);
}

inline constexpr size_t kRingBufferSize = 128;  // 2 ^ n
inline constexpr size_t kIdleSleepTimeMS = 1;

BufferedDaemonSentenceReader::RingCell::RingCell(size_t seq) : seq_{seq} {}

BufferedDaemonSentenceReader::RingBuffer::RingBuffer()
    : reader_index_(0),
      writer_index_(0),
      p_ring_cell_(static_cast<RingCell*>(::operator new(
          sizeof(BufferedDaemonSentenceReader::RingCell) * kRingBufferSize,
          std::align_val_t(kCacheLineSize)))) {
  for (size_t i = 0; i < kRingBufferSize; ++i) {
    new (&p_ring_cell_[i]) RingCell(i);
  }
}

BufferedDaemonSentenceReader::RingBuffer::~RingBuffer() {
  for (size_t i = 0; i < kRingBufferSize; ++i) {
    p_ring_cell_[i].~RingCell();
  }
  ::operator delete(p_ring_cell_, std::align_val_t(kCacheLineSize));
}

BufferedDaemonSentenceReader::BufferedDaemonSentenceReader(
    const size_t n_iteration, const size_t n_client_threads,
    std::filesystem::path train_file, const Dictionary& dictionary,
    const float sample, SentenceReader::FactoryFnT factory)
    : n_iteration_(n_iteration),
      train_file_(std::move(train_file)),
      dictionary_(dictionary),
      sample_(sample),
      sentence_reader_factory_(std::move(factory)),
      p_sen_reader_(
          sentence_reader_factory_(train_file_, dictionary_, sample_)),
      n_ring_buffers_(n_client_threads),
      ring_buffer_headers_(n_client_threads) {}

BufferedDaemonSentenceReader::~BufferedDaemonSentenceReader() {
  finished_to_read_.store(true, std::memory_order_release);
  if (activated_.load(std::memory_order_relaxed) && fetch_thread_.joinable()) {
    fetch_thread_.join();
  }
}

bool BufferedDaemonSentenceReader::Register() {
  auto thread_id = std::this_thread::get_id();
  std::lock_guard<std::mutex> lg(lock_seq_map_);
  if (thread_seq_map_.size() >= n_ring_buffers_ ||
      thread_seq_map_.find(thread_id) != thread_seq_map_.end()) {
    return false;
  }
  thread_seq_map_[thread_id] = thread_seq_map_.size();
  if (thread_seq_map_.size() == n_ring_buffers_) {
    fetch_thread_ =
        std::move(std::thread(&BufferedDaemonSentenceReader::Fetch, this));
    activated_.store(true, std::memory_order_release);
  }
  return true;
}

std::vector<W2VType::WordIndexT> BufferedDaemonSentenceReader::Next() {
  const auto buf_idx_iter = thread_seq_map_.find(std::this_thread::get_id());
  if (buf_idx_iter == thread_seq_map_.end()) {
    return {};
  }
  WaitIfNotActivated();
  RingBuffer& buf = ring_buffer_headers_.at(buf_idx_iter->second);
  RingCell* cell{nullptr};
  auto r_idx = buf.reader_index_.load(std::memory_order_relaxed);
  while (true) {
    cell = &buf.p_ring_cell_[r_idx & (kRingBufferSize - 1)];
    auto seq = cell->seq_.load(std::memory_order_acquire);
    auto diff = static_cast<int32_t>(r_idx + 1) - static_cast<int32_t>(seq);

    if (diff == 0 && buf.reader_index_.compare_exchange_weak(
                         r_idx, r_idx + 1, std::memory_order_relaxed)) {
      break;
    } else if (diff > 0 &&
               !finished_to_read_.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kIdleSleepTimeMS));
    } else if (diff > 0) {
      return {};
    }
    r_idx = buf.reader_index_.load(std::memory_order_relaxed);
  }
  std::vector<W2VType::WordIndexT> sentence{std::move(cell->sentence_)};
  cell->seq_.store(r_idx + kRingBufferSize, std::memory_order_release);
  return sentence;
}

void BufferedDaemonSentenceReader::Fetch() {
  size_t line_num{0};
  while (true) {
    auto sentence = p_sen_reader_->Next();
    if (sentence.empty() && iter_idx_ + 1 < n_iteration_) {
      line_num = 0;
      ++iter_idx_;
      p_sen_reader_.reset();
      p_sen_reader_ =
          sentence_reader_factory_(train_file_, dictionary_, sample_);
      continue;
    } else if (sentence.empty()) {
      finished_to_read_.store(true, std::memory_order_release);
      break;
    }
    RingBuffer& buf = ring_buffer_headers_.at(line_num++ % n_ring_buffers_);
    RingCell* cell{nullptr};
    auto w_idx = buf.writer_index_.load(std::memory_order_relaxed);
    while (true) {
      cell = &buf.p_ring_cell_[w_idx & (kRingBufferSize - 1)];
      auto seq = cell->seq_.load(std::memory_order_acquire);
      auto diff = static_cast<int32_t>(w_idx) - static_cast<int32_t>(seq);
      if (diff == 0 && buf.writer_index_.compare_exchange_weak(
                           w_idx, w_idx + 1, std::memory_order_relaxed)) {
        break;
      } else if (diff > 0) {  // full
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kIdleSleepTimeMS));
      }
      w_idx = buf.writer_index_.load(std::memory_order_relaxed);
    }
    cell->sentence_ = std::move(sentence);
    cell->seq_.store(w_idx + 1, std::memory_order_release);
  }
}

void BufferedDaemonSentenceReader::WaitIfNotActivated() {
  while (!activated_.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kIdleSleepTimeMS));
  }
}
