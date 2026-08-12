#include "buffer_array.h"

#include <gtest/gtest.h>

TEST(PaddingCalculationTest, PaddingCheck) {
  EXPECT_EQ(CalcPadding(0), 0);   // 0 -> 0
  EXPECT_EQ(CalcPadding(1), 64);  // gt(0) & less(64) -> 64
  EXPECT_EQ(CalcPadding(63), 64);
  EXPECT_EQ(CalcPadding(64), 64);    // eq(64) -> 64
  EXPECT_EQ(CalcPadding(120), 128);  // gt(64 * 1) & less(64 * 2) -> 64 * 2
  EXPECT_EQ(CalcPadding(128), 128);
  EXPECT_EQ(CalcPadding(129), 192);
  EXPECT_EQ(CalcPadding(191), 192);
  EXPECT_EQ(CalcPadding(193), 256);
}

TEST(OneDimArrayTest, PrimitiveTypes) {
  // a buffer is returned with initialsing 0
  auto* int_buf = CreateBuffer<int>(10);
  EXPECT_EQ(*(int_buf + 0), 0);
  EXPECT_EQ(*(int_buf + 9), 0);
  std::free(int_buf);

  auto* double_buf = CreateBuffer<double>(1024);
  EXPECT_EQ(*(double_buf + 0), 0.0);
  EXPECT_EQ(*(double_buf + 512), 0.0);
  EXPECT_EQ(*(double_buf + 1023), 0.0);
  std::free(double_buf);
}

TEST(OneDimArrayTest, TriviallyConstructableStruct) {
  struct Trivial {
    Trivial() : a(10) {}
    int a;
    int b;
    std::string c;
  };

  Trivial* buf = CreateBuffer<Trivial>(10);
  // CreateBuffer calls a default constructor for each element
  EXPECT_EQ(static_cast<Trivial>(*(buf + 0)).a, 10);
  EXPECT_EQ(static_cast<Trivial>(*(buf + 0)).b, 0);
  EXPECT_EQ(static_cast<Trivial>(*(buf + 9)).a, 10);
  EXPECT_EQ(static_cast<Trivial>(*(buf + 9)).b, 0);
  EXPECT_EQ(static_cast<Trivial>(*(buf + 9)).c, std::string{});
}

TEST(Align2DBufferFunctionalTest, SetGetElementsWithFillAll) {
  const int row = 256, col = 256;

  Aligned2DBuffer<int> rect(row, col);

  EXPECT_EQ(rect.Element(0, 0), 0);              // int default: 0
  EXPECT_EQ(rect.Element(row - 1, col - 1), 0);  // int default: 0

  rect.FillAll([](const int val) { return -200; });  // rect <- -200

  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      const int val = rect.Element(r, c);
      EXPECT_EQ(val, -200);              // check if every elem is -200
      rect.Element(r, c) = r * col + c;  // set
      EXPECT_EQ(rect.Element(r, c), r * col + c);  // and check
    }
  }
}

TEST(Align2DBufferFunctionalTest, MoveConstructor) {
  struct NonCopyable {
    NonCopyable() : a(10), c("Hello,World") {}

    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;

    int a;
    double b;
    std::string c;
  };

  Aligned2DBuffer<NonCopyable> rect(10, 10);

  EXPECT_EQ(rect.Element(0, 0).a, 10);
  EXPECT_EQ(rect.Element(9, 9).c, "Hello,World");

  // when an object is moved
  Aligned2DBuffer<NonCopyable> moved{std::move(rect)};

  // then, (1) the moved object has the same value
  EXPECT_EQ(moved.Element(0, 0).a, 10);
  EXPECT_EQ(moved.Element(9, 9).c, "Hello,World");

  // then, (2) the original instance isn't accessible
  EXPECT_THROW(rect.Element(0, 0), std::logic_error);
}

TEST(Align2DBufferFunctionalTest, VectorIterator) {
  // when, for a matrix 10x10
  const size_t row = 10, col = 10;
  Aligned2DBuffer<int> rect(row, col);

  for (int r = 0; r < row; ++r) {
    if (r % 2) {
      continue;
    }
    // every row with an even index is updated with 1
    const auto iter = rect.LayerIterPair(r);
    std::transform(iter.first, iter.second, iter.first,
                   [](const int /*unused*/) { return 1; });
  }

  // then, verify whether an even index row has values of 1
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      const int val = rect.Element(r, c);
      EXPECT_EQ(val, (r % 2 ? 0 : 1));
    }
  }
}