#include "word_layer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

using ::testing::Return;

Dictionary kEmptyDictionary;

struct VectorOutFactoryValue {
  size_t n_classes_;
  bool is_binary_;
  const std::string class_name_;
};

struct FactoryTestFixture
    : public WordLayer,
      public testing::TestWithParam<VectorOutFactoryValue> {
 protected:
  FactoryTestFixture() : WordLayer(kEmptyDictionary, 100 /* not important*/) {}
};

INSTANTIATE_TEST_SUITE_P(
    FactoryCreationParameters, FactoryTestFixture,
    testing::Values(VectorOutFactoryValue{0, true, {"BinaryVectorOutput"}},
                    VectorOutFactoryValue{0, false, {"TextVectorOutput"}},
                    VectorOutFactoryValue{1, true, {"ClusterVectorOutput"}},
                    VectorOutFactoryValue{1, false, {"ClusterVectorOutput"}},
                    VectorOutFactoryValue{10, true, {"ClusterVectorOutput"}},
                    VectorOutFactoryValue{10, false, {"ClusterVectorOutput"}},
                    VectorOutFactoryValue{100, true, {"ClusterVectorOutput"}},
                    VectorOutFactoryValue{100, false, {"ClusterVectorOutput"}}

                    ));

TEST_P(FactoryTestFixture, ParamChecking) {
  const VectorOutFactoryValue& param = GetParam();
  std::unique_ptr<WordVectorOutput> p_vector_out =
      VectorOutputFactory(param.n_classes_, param.is_binary_);

  EXPECT_NE(p_vector_out, nullptr);
  EXPECT_TRUE(
      std::string(typeid(*p_vector_out).name()).find(param.class_name_) !=
      std::string::npos);
}

struct RandomVectorVarianceTestValue {
  size_t vocab_size_;     // | n_row |
  size_t vector_length_;  // | n_col |
};


class MockDictionary : public Dictionary {
 public:
  MOCK_METHOD(size_t, VocabSize, (), (override, const));
};

struct VarianceTestFixture
    : public testing::TestWithParam<RandomVectorVarianceTestValue> {};

INSTANTIATE_TEST_SUITE_P(
    WordVectorParameters, VarianceTestFixture,
    testing::Values(RandomVectorVarianceTestValue(100, 100),
                    RandomVectorVarianceTestValue(1000, 200),
                    RandomVectorVarianceTestValue(2000, 300)

                        ));

TEST_P(VarianceTestFixture, VarianceMeasuring) {
  const RandomVectorVarianceTestValue& param = GetParam();
  // given
  MockDictionary mock_dict;
  EXPECT_CALL(mock_dict, VocabSize()).WillRepeatedly(Return(param.vocab_size_));

  // when
  WordLayer word_layer(mock_dict, param.vector_length_);

  // then
  for (size_t i = 1; i < param.vocab_size_; ++i) {  // skip 0
    auto pair = word_layer.WordVectorIterPair(i);
    const auto sum = std::accumulate(pair.first, pair.second, 0.0);
    const auto avg = sum / param.vocab_size_;
    const auto sqsum =
        std::inner_product(pair.first, pair.second, pair.first, 0.0);
    const auto var = (sqsum / param.vocab_size_) - avg * avg;

    const auto expect_var = 1 / (param.vocab_size_ * param.vocab_size_) / 12;

    EXPECT_TRUE(sum <= 0.5);
    EXPECT_TRUE(sum >= -0.5);
    EXPECT_NEAR(avg, 0, 0.001);
    EXPECT_NEAR(var, expect_var, 0.001);
  }
}
