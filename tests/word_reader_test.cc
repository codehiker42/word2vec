#include "word_reader.h"

#include <gtest/gtest.h>

#include <barrier>
#include <future>
#include <random>
#include <sstream>
#include <thread>

TEST(PathConstructorTest, UnexistFileName) {
  EXPECT_THROW(WordReader("UnexistDirectory/UnexistFileName.EXT"),
               std::runtime_error);
}

TEST(PathConstructorTest, EmptyFileReading) {
  // given, an empty temporary file
  std::string filename = "TEST_" + std::to_string(std::random_device{}());
  const auto path = std::filesystem::temp_directory_path() / filename;
  {
    std::ofstream of(path);
    of << "";
  }

  // when, create a WordReader
  WordReader reader(path);

  // then, no error and reads nothing
  EXPECT_FALSE(reader.Next().has_value());
  EXPECT_FALSE(reader.Next().has_value());  // call twice

  std::filesystem::remove(path);
}

struct StreamParamTestValue {
  const std::string content_;
  std::vector<std::string> words_seq_;
};

class WordStreamParamTest
    : public testing::TestWithParam<StreamParamTestValue> {};

INSTANTIATE_TEST_SUITE_P(
    WordStreamParameters, WordStreamParamTest,
    testing::Values(
        StreamParamTestValue{{}, {}},  // empty
        StreamParamTestValue{"  \tWORD\t", {"WORD"}},
        StreamParamTestValue{" \n  \n \tWORD\n\t",
                             {WordReader::NEW_LINE, WordReader::NEW_LINE,
                              "WORD", WordReader::NEW_LINE}},
        StreamParamTestValue{
            " abcd  \t efg  hij kl \tmnop  qrst  \n \n uvw xy z",
            {"abcd", "efg", "hij", "kl", "mnop", "qrst", WordReader::NEW_LINE,
             WordReader::NEW_LINE, "uvw", "xy", "z"}},
        StreamParamTestValue{" 1234  \t 567 , . !#$%^ 89 *()",
                             {}},  // only alphabet characters are counted
        StreamParamTestValue{" 1234  \t 567 wo1rd !#$%^ 89 *()", {"word"}},
        // By design, children’s -> childrens, death-marked -> deathmarked
        StreamParamTestValue{
            "Two households, both alike in dignity "
            "(In fair Verona, where we lay our scene), "
            "From ancient grudge break to new mutiny, "
            "Where civil blood makes civil hands unclean. "
            "From forth the fatal loins of these two foes "
            "A pair of star-crossed lovers take their life; "
            "Whose misadventured piteous overthrows "
            "Doth with their death bury their parents’ strife. "
            "The fearful passage of their death-marked love "
            "And the continuance of their parents’ rage, "
            "Which, but their children’s end, naught could remove, "
            "Is now the two hours’ traffic of our stage; "
            "The which, if you with patient ears attend, "
            "What here shall miss, our toil shall strive to mend.",
            {"Two",
             "households",
             "both",
             "alike",
             "in",
             "dignity",
             "In",
             "fair",
             "Verona",
             "where",
             "we",
             "lay",
             "our",
             "scene",
             "From",
             "ancient",
             "grudge",
             "break",
             "to",
             "new",
             "mutiny",
             "Where",
             "civil",
             "blood",
             "makes",
             "civil",
             "hands",
             "unclean",
             "From",
             "forth",
             "the",
             "fatal",
             "loins",
             "of",
             "these",
             "two",
             "foes",
             "A",
             "pair",
             "of",
             "starcrossed",
             "lovers",
             "take",
             "their",
             "life",
             "Whose",
             "misadventured",
             "piteous",
             "overthrows",
             "Doth",
             "with",
             "their",
             "death",
             "bury",
             "their",
             "parents",
             "strife",
             "The",
             "fearful",
             "passage",
             "of",
             "their",
             "deathmarked",
             "love",
             "And",
             "the",
             "continuance",
             "of",
             "their",
             "parents",
             "rage",
             "Which",
             "but",
             "their",
             "childrens",
             "end",
             "naught",
             "could",
             "remove",
             "Is",
             "now",
             "the",
             "two",
             "hours",
             "traffic",
             "of",
             "our",
             "stage",
             "The",
             "which",
             "if",
             "you",
             "with",
             "patient",
             "ears",
             "attend",
             "What",
             "here",
             "shall",
             "miss",
             "our",
             "toil",
             "shall",
             "strive",
             "to",
             "mend"}}));

TEST_P(WordStreamParamTest, StreamReading) {
  // given, parameter
  const StreamParamTestValue& param = GetParam();
  std::istringstream iss(param.content_);

  // when, for every WordReader intance
  WordReader word_reader(iss);

  // then,
  for (const auto& word : param.words_seq_) {
    EXPECT_EQ(*word_reader.Next(), word);
  }
  EXPECT_FALSE(word_reader.Next().has_value());
}

TEST_P(WordStreamParamTest, ConcurrentStreamReading) {
  // given, parameter, one stream,
  const StreamParamTestValue& param = GetParam();
  std::istringstream iss(param.content_);

  // when, one WordReader, one stream, but my multiple threads
  const size_t n_threads = 5;
  std::barrier sync_point(n_threads);
  WordReader word_reader(iss);
  auto func = [](WordReader& reader, std::barrier<>& sync) {
    std::vector<std::string> words;
    sync.arrive_and_wait();
    for (auto w_op = reader.Next(); w_op; w_op = reader.Next()) {
      words.emplace_back(*w_op);
    }
    return words;
  };
  std::vector<std::future<std::vector<std::string>>> futures;
  for (auto i = 0; i < n_threads; ++i) {
    futures.emplace_back(std::async(
        std::launch::async, func, std::ref(word_reader), std::ref(sync_point)));
  }
  std::vector<std::vector<std::string>> vec_of_words;
  for (auto& fut : futures) {
    fut.wait();
    vec_of_words.emplace_back(std::move(fut.get()));
  }

  // then, (1) | vec_of_words | == param.words_seq_.size()
  auto n_words = std::accumulate(
      vec_of_words.begin(), vec_of_words.end(), 0UL,
      [](const auto sum, const auto& vec) { return vec.size() + sum; });
  EXPECT_EQ(param.words_seq_.size(), n_words);

  //    , (2) every vec is a sequence of param.words_seq
  for (auto i = 0; i < n_threads; ++i) {
    auto whole_seq_iter = param.words_seq_.begin();
    for (const auto& word : vec_of_words.at(i)) {
      auto iter = std::find(whole_seq_iter, param.words_seq_.end(), word);
      EXPECT_TRUE(iter != param.words_seq_.end()) << "Unexpected seq " << word;
      whole_seq_iter = std::next(iter);
    }
  }
}
