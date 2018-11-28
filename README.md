# MnemonicReaderAllennlp
PyTorch implementation of MnemonicReader based on Allennlp

Update:

- version0/output_dir1: hops = 1, no stacked bilstm layer
- version0/output_dir2: hops = 2, no stacked bilstm layer
- version0/output_dir3: hops = 2, add char features rnn
- version0/output_dir4: add ner and pos features embedding
- version0/output_dir5: change the GloVe word vectors of 100 dimension to 300 dimension
- version0/output_dir6: add tf, lemma and exact match features for tokens; remove highway layers of embedded question and passage; add dropout to the embedded question and passage.
- version0/output_dir7: modified the learning rate of adam from 1e-4 to 1e-3
- version0/output_dir8: modified the char embedding to use glove char-embedding; Complete the loss function;
- version0/output_dir9: modified the learning rate of adam from 1e-3 to 1e-2
- version0/output_dir10: modified the learning rate of adam from 1e-2 to 0.002 and the betas to [0.9, 0.999]
- version0/output_dir11: modified the optimizer from adam to adamax and the parameter is as default
- version0/output_dir12: change the reader to smaller span
- version0/output_dir13: change the reader as the f1 reader to choose the input text from span text artificially to get max f1 value with original input text

TODO:
- ~~Add tf and exact match features for token~~
- Add previous answer positional embedding and question turn id embedding
- Add previous answer and question in the form
- ~~Update loss function in RL~~
- Update Gumbel max trick
- Add extract/yes or no classifier
- Pre-train yes or no classifier
- Add bert
- Use optimizer adamax