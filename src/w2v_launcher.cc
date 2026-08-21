#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>

#include "dictionary.h"
#include "word2vec.h"

std::filesystem::path GetVocabFile(const Word2VecOptions& options) {
  return options.VocabFileNameToLoad().value_or(
      options.TrainFileName().value());
}

int main(int argc, char** argv) {
  Word2VecOptions options{argc, argv};

  if (!options.HasMandatoryOptions()) {
    std::cout << options.HelpMessage();
    return 1;
  }

  auto train_fun_factory =
      [](Word2Vec& word2vec,
         const Word2VecOptions& options) -> std::function<WordLayer()> {
    if (options.IsCBowOrSkip()) {
      if (options.UseHierachicalSoftmax()) {
        return std::bind(
            &Word2Vec::Train<ContinuousBagOfWords, HierachicalSoftMax>,
            word2vec, options);
      } else {
        return std::bind(
            &Word2Vec::Train<ContinuousBagOfWords, NegativeSampling>, word2vec,
            options);
      }

    } else {
      if (options.UseHierachicalSoftmax()) {
        return std::bind(&Word2Vec::Train<SkipGram, HierachicalSoftMax>,
                         word2vec, options);

      } else {
        return std::bind(&Word2Vec::Train<SkipGram, NegativeSampling>, word2vec,
                         options);
      }
    }
    throw std::runtime_error("Unexpected state");
  };

  const Dictionary dictionary(GetVocabFile(options), options.MinCount());
  if (options.VocabFileNameToSave()) {
    dictionary.DumpVocab(*options.VocabFileNameToSave());
  }
  Word2Vec word2vec(dictionary);

  const auto train_func = train_fun_factory(word2vec, options);
  WordLayer word_layer = train_func();
  word_layer.SaveVectors(*options.OutputVectorFilePath(), options.NumClusters(),
                         options.IsBinaryWriting());
  return 0;
}
