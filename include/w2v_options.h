#ifndef _W2V_OPTIONS_H
#define _W2V_OPTIONS_H

#include "kv_options.h"


class Word2VecOptions : public ConsoleOptionValues {
 public:

  Word2VecOptions() = default;
  Word2VecOptions(int argc, char** argv);

  const bool HasMandatoryOptions() const override;

  const std::optional<std::filesystem::path> TrainFileName() const;

  const std::optional<std::filesystem::path> VocabFileNameToSave() const;

  const std::optional<std::filesystem::path> VocabFileNameToLoad() const;

  const std::optional<std::filesystem::path> OutputVectorFilePath() const;

  const uint64_t LayerSize() const;

  const bool IsDebugMode() const;

  const bool IsBinaryWriting() const;

  const bool IsCBowOrSkip() const;

  const float LearningRate() const;

  const uint8_t WindowSize() const;

  const float DownSamplingSize() const;

  const bool UseHierachicalSoftmax() const;

  const uint8_t NegativeSize() const;

  const uint8_t IterationSize() const;

  const uint8_t MinCount() const;

  const uint16_t NumClusters() const;

  const uint8_t ThreadSize() const;
};

#endif
