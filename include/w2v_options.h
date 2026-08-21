#ifndef _W2V_OPTIONS_H
#define _W2V_OPTIONS_H

#include "kv_options.h"


class Word2VecOptions : public ConsoleOptionValues {
 public:

  Word2VecOptions() = default;
  Word2VecOptions(int argc, char** argv);

  bool HasMandatoryOptions() const override;

  const std::optional<std::filesystem::path> TrainFileName() const;

  const std::optional<std::filesystem::path> VocabFileNameToSave() const;

  const std::optional<std::filesystem::path> VocabFileNameToLoad() const;

  const std::optional<std::filesystem::path> OutputVectorFilePath() const;

  uint64_t LayerSize() const;

  bool IsDebugMode() const;

  bool IsBinaryWriting() const;

  bool IsCBowOrSkip() const;

  float LearningRate() const;

  uint16_t WindowSize() const;

  float DownSamplingSize() const;

  bool UseHierachicalSoftmax() const;

  uint16_t NegativeSize() const;

  uint16_t IterationSize() const;

  uint16_t MinCount() const;

  uint16_t NumClusters() const;

  uint16_t ThreadSize() const;
};

#endif
