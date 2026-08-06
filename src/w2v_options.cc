#include "w2v_options.h"

#include <stdexcept>

struct DefaultOption {
  inline static const std::tuple<bool, std::string> kDebugMode{false,
                                                               "No debug mode"};
  inline static const std::tuple<bool, std::string> kBinaryWriting{
      false, "Text file writing"};
  inline static const std::tuple<bool, std::string> kUseSkipGram{false,
                                                                 "Using CBOW"};
  inline static const std::tuple<bool, std::string> kUseHs{
      false, "Not using Hierarchical Softmax"};
  inline static const std::tuple<uint16_t, std::string> kWriteClasses{
      0, "Number of clusters, default:0"};

  inline static const std::tuple<float, std::string> kLearningRate{0.5, "0.5"};
  inline static const std::tuple<float, std::string> kDownSamplingSize{1e-3,
                                                                       "1e-3"};

  inline static const std::tuple<uint64_t, std::string> kLayerSize{100, "100"};

  inline static const std::tuple<uint8_t, std::string> kWindowSize{5, "5"};
  inline static const std::tuple<uint8_t, std::string> kNegativeSize{5, "5"};
  inline static const std::tuple<uint8_t, std::string> kIterationSize{5, "5"};
  inline static const std::tuple<uint8_t, std::string> kMinCount{5, "5"};
  inline static const std::tuple<uint8_t, std::string> kDefaultThread{12, "12"};
};

struct W2VOpNames {
  inline static const std::string output{"output"};
  inline static const std::string train{"train"};
  inline static const std::string size{"size"};
  inline static const std::string window{"window"};
  inline static const std::string sample{"sample"};
  inline static const std::string hs{"hs"};
  inline static const std::string negative{"negative"};
  inline static const std::string iter{"iter"};
  inline static const std::string min_count{"min-count"};
  inline static const std::string alpha{"alpha"};
  inline static const std::string classes{"classes"};
  inline static const std::string debug{"debug"};
  inline static const std::string binary{"binary"};
  inline static const std::string save_vocab{"save-vocab"};
  inline static const std::string read_vocab{"read-vocab"};
  inline static const std::string skip_gram{"skip-gram"};
  inline static const std::string threads{"threads"};
};

Word2VecOptions::Word2VecOptions(int argc, char** argv)
    : ConsoleOptionValues(
          argc, argv,
          {{W2VOpNames::train, 't', true,
            "<file>;Use text data from <file> to train the model",
            typeid(std::string)},

           {W2VOpNames::output, 'o', true,
            "<file>;Use <file> to save the resulting 'word vectors / word "
            "clusters'",
            typeid(std::string)},

           {W2VOpNames::size, 's', true, "<int>;Set size of word vectors",
            DefaultOption::kLayerSize},

           {W2VOpNames::window, 'w', true,
            "<int>;Set max skip length between words",
            DefaultOption::kWindowSize},

           {W2VOpNames::sample, 0, true,
            "<float>;Set max skip length between words, useful range is (0, "
            "1e-5)",
            DefaultOption::kDownSamplingSize},

           {W2VOpNames::hs, 0, false,
            ";Use Hierarchical Softmax, when this option is used",
            DefaultOption::kUseHs},

           {W2VOpNames::negative, 'n', true,
            "<int>;Number of negative examples common values are 3 - 10 (0 = "
            "not used)",
            DefaultOption::kNegativeSize},

           {W2VOpNames::iter, 'i', true, "<int>;Run more training iterations",
            DefaultOption::kIterationSize},

           {W2VOpNames::min_count, 0, true,
            "<int>;This will discard words that appear less than <int> times",
            DefaultOption::kMinCount},

           {W2VOpNames::alpha, 'a', true,
            "<float>;Set the starting learning rate",
            DefaultOption::kLearningRate},

           {W2VOpNames::classes, 0, true,
            "<int>;Output word classes rather than word vectors",
            DefaultOption::kWriteClasses},

           {W2VOpNames::debug, 'd', false, ";Set the debug mode",
            DefaultOption::kDebugMode},

           {W2VOpNames::binary, 'b', false,
            ";Save the resulting vectors in binary moded;",
            DefaultOption::kBinaryWriting},

           {W2VOpNames::save_vocab, 0, false,
            "<file>;The vocabulary will be saved to <file>",
            typeid(std::string)},

           {W2VOpNames::read_vocab, 0, false,
            "<file>;The vocabulary will be read from <file>, not constructed "
            "from the training data",
            typeid(std::string)},

           {W2VOpNames::skip_gram, 0, false,
            ";Use skip-gram model, otherwise CBOW",
            DefaultOption::kUseSkipGram},

           {W2VOpNames::threads, 0, true,
            "<int>;Number of negative examples; default is 5, common values "
            "are 3 - 10",
            DefaultOption::kDefaultThread}}) {
  if (!HasMandatoryOptions()) {
    throw std::invalid_argument("Not all mandatory options are set.");
  }
}

const std::optional<std::filesystem::path> Word2VecOptions::TrainFileName()
    const {
  return get_op<std::filesystem::path>(W2VOpNames::train);
}

const std::optional<std::filesystem::path>
Word2VecOptions::VocabFileNameToSave() const {
  return get_op<std::filesystem::path>(W2VOpNames::save_vocab);
}

const std::optional<std::filesystem::path>
Word2VecOptions::VocabFileNameToLoad() const {
  return get_op<std::filesystem::path>(W2VOpNames::read_vocab);
}

const std::optional<std::filesystem::path>
Word2VecOptions::OutputVectorFilePath() const {
  return get_op<std::filesystem::path>(W2VOpNames::output);
}

const uint64_t Word2VecOptions::LayerSize() const {
  return get<uint64_t>(W2VOpNames::size);
}

const bool Word2VecOptions::IsDebugMode() const {
  return get<bool>(W2VOpNames::debug);
}

const bool Word2VecOptions::IsBinaryWriting() const {
  return get<bool>(W2VOpNames::binary);
}

const bool Word2VecOptions::IsCBowOrSkip() const {
  return get<bool>(W2VOpNames::skip_gram);
}

const float Word2VecOptions::LearningRate() const {
  return get<float>(W2VOpNames::alpha);
}

const uint8_t Word2VecOptions::WindowSize() const {
  return get<uint8_t>(W2VOpNames::window);
}

const float Word2VecOptions::DownSamplingSize() const {
  return get<float>(W2VOpNames::sample);
}

const bool Word2VecOptions::UseHierachicalSoftmax() const {
  return get<bool>(W2VOpNames::hs);
}

const uint8_t Word2VecOptions::NegativeSize() const {
  return get<uint8_t>(W2VOpNames::negative);
}

const uint8_t Word2VecOptions::IterationSize() const {
  return get<uint8_t>(W2VOpNames::iter);
}

const uint8_t Word2VecOptions::MinCount() const {
  return get<uint8_t>(W2VOpNames::min_count);
}

const uint16_t Word2VecOptions::NumClusters() const {
  return get<uint16_t>(W2VOpNames::classes);
}

const bool Word2VecOptions::HasMandatoryOptions() const {
  return get_op<std::string>(W2VOpNames::train).has_value() &&
         get_op<std::string>(W2VOpNames::output).has_value();
}

const uint8_t Word2VecOptions::ThreadSize() const {
  return get<uint8_t>(W2VOpNames::threads);
}
