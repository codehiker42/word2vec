#ifndef _UNIGRAM_TABLE_H
#define _UNIGRAM_TABLE_H

#include <vector>

#include "w2v_type.h"

class UnigramTable {
 public:
  /* The original version fixes the table size as 10^8; the number may be larger
   * or smaller than the number of vocabularies. This implementation inherits
   * the table size, but it can have a different number in a static way.
   *
   * The class works under the contract; the argument freq_vec has elements that
   * are sorted in descending order, except index:0, which is for NEW_LINE */
  UnigramTable(const std::vector<size_t>& freq_vec,
               const size_t table_size = 1e8);

  W2VType::WordIndexT RandomeIndex() const;

 private:
  std::vector<W2VType::WordIndexT> table_;
};

#endif