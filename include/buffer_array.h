#ifndef _BUFFER_ARRAY_H
#define _BUFFER_ARRAY_H

#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>

#include "global_constants.h"

inline size_t CalcPadding(size_t size) {
  return (size + kCacheLineSize - 1) / kCacheLineSize * kCacheLineSize;
}

template <typename T>
inline T* CreateBuffer(size_t size) {
  T* buf = static_cast<T*>(std::calloc(size, sizeof(T)));
  for (size_t i = 0; i < size; ++i) {
    new (buf + i) T();
  }
  return buf;
}

template <typename T>
class Aligned2DBuffer {
 public:
  Aligned2DBuffer(const size_t n_row, const size_t n_col,
                  const size_t align = kCacheLineSize)
      : n_row_(n_row),
        n_col_(n_col),
        vec_stride_size_(CalcPadding(n_col * sizeof(T)) / sizeof(T)) {
    const auto size = CalcPadding(n_row * n_col * sizeof(T));
    buffer_ = static_cast<T*>(std::aligned_alloc(align, size));
    if (!buffer_) {
      throw std::bad_alloc{};
    }

    for (size_t r = 0; r < n_row; ++r) {
      for (size_t c = 0; c < n_col; ++c) {
        const size_t index = r * vec_stride_size_ + c;
        new (buffer_ + index) T();
      }
    }
  }

  Aligned2DBuffer(const Aligned2DBuffer&) = delete;
  Aligned2DBuffer& operator=(const Aligned2DBuffer&) = delete;

  Aligned2DBuffer(Aligned2DBuffer&& another)
      : n_row_(another.n_row_),
        n_col_(another.n_col_),
        vec_stride_size_(another.vec_stride_size_),
        buffer_(another.buffer_)

  {
    another.buffer_ = nullptr;
  }

  ~Aligned2DBuffer() {
    if (buffer_) {
      std::free(buffer_);
    }
  }

  void FillAll(std::function<T(T)>&& func) {
    CheckIfBufferIsValid();
    std::transform(buffer_, buffer_ + (n_row_ * vec_stride_size_), buffer_,
                   func);
  }

  T& Element(size_t row, size_t col) {
    CheckIfBufferIsValid();
    return static_cast<T&>(*(buffer_ + (row * vec_stride_size_) + col));
  }

  const T Element(size_t row, size_t col) const {
    CheckIfBufferIsValid();
    return static_cast<T>(*(buffer_ + (row * vec_stride_size_) + col));
  }

  std::pair<T*, T*> LayerIterPair(const size_t index) const {
    CheckIfBufferIsValid();
    T* begin = buffer_ + (index * vec_stride_size_);
    T* end = begin + n_col_;
    return {begin, end};
  }

  const size_t VecSize() const { return n_col_; }

 private:
  void CheckIfBufferIsValid() const {
    if (!buffer_) {
      throw std::logic_error("Access to deleted buffer");
    }
  };

  const size_t n_row_;
  const size_t n_col_;
  const size_t vec_stride_size_;
  T* buffer_;
};

#endif
