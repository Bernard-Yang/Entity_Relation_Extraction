# Entity_Relation_Extraction
- English Entity Relation Extraction 
- Using Bidirectional LSTM neural network and Attention mechanism which taking 100 dimension Glove word embedding and entities positional embeddings as input on SemEval2010 task8 datasets to predict the class of relation for each entity pair.
- Achieved 66% Precision, 67% Recall, 67% F1 score after 300 epochs training.

# Environment 
- python 3.6
- pytorch 1.1

# How to run
- Download Glove word embedding from https://nlp.stanford.edu/projects/glove/
- python train.py pretrained
