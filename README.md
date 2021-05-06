# Entity_Relation_Extraction
- English Entity Relation Extraction on SemEval2010 task8 datasets.
- Using Bidirectional LSTM neural network and Attention mechanism to predict the class of relation for each entity pair.
- Achieved 54% Precision, 55% Recall, 54% F1 score with 100 dimension Glove word embedding as input. LSTM hidden size is 50.
- Achieved 70% Precision, 71% Recall, 71% F1 score with 100 dimension Glove word embedding and entities position embedding as input. LSTM hidden size is 50.

# Environment 
- python 3.6
- pytorch 1.1

# How to run
- Download Glove word embedding from https://nlp.stanford.edu/projects/glove/
- python train.py pretrained

# Reference

[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://www.aclweb.org/anthology/P16-2034.pdf)
