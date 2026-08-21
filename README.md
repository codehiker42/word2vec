## Word2Vec C++23

This is a homage project for the famous Word2Vec. This project is functionally identical to the original project, but it is newly written in C++23. 



## Overview

Train Word2Vec with C++ style
```
// 1. build vocabulary for Word2Vec
Dictionary dictionary(...);
// 2. Create a Word2Vec instance
Word2Vec word2vec(dictionary); 

//3. Specify a model and a train method with templates, 
//   such as Train<SkipGram, NegativeSampling>
WordLayer word_layer = 
  word2vec.Train<ContinuousBagOfWords, HierachicalSoftMax>(
                                                  .../*train file & opts. */);
  
// 4. word_layer has trained vectors.
word_layer.SaveVectors(...) 

```


## Differences from the original project

tbd

