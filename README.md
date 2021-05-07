# Entity_Relation_Extraction
- English Entity Relation Extraction on SemEval2010 task8 datasets.
- Using Bidirectional LSTM neural network and Attention mechanism to predict the class of relation for each entity pair.

# Result
|  Input   | Hidden Size  | Training Precison |Training Recall |Training F1 | Validation Precison| Validation Recall| Validation F1|
| -------- | ------------ | ----------------- | -------------- | ---------- |------------------- |----------------- |------------- |
| 100d word embedding  | 50 | 95% | 97% | 96% | 58% | 59% | 58% |
| 100d word and entities position embedding | 50 | 96% | 97% | 97%| 66% | 68% | 67% |
| 100d word and entities position embedding | 150 | 96% | 97% | 96% | 70% | 71% | 70% |
| 100d word and entities position embedding | 300 | 96% | 97% | 96% | 71% | 69% | 70% |

# Environment 
- python 3.6
- pytorch 1.1

# How to run
- Download Glove word embedding from https://nlp.stanford.edu/projects/glove/
- python train.py pretrained

# Reference

[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://www.aclweb.org/anthology/P16-2034.pdf)
