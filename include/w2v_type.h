#ifndef _W2V_TYPE_H
#define _W2V_TYPE_H

#include <cstddef>
#include <vector>

struct W2VType {
  using RealT = float;
  using WordIndexT = size_t;

  using SenItT = std::vector<WordIndexT>::const_iterator;
  using SenPairT = std::pair<SenItT, SenItT>;
  using VecT = std::vector<RealT>;

  struct Element {
    W2VType::SenItT target_;
    std::vector<SenPairT> neighbours_;
  };
};

#endif