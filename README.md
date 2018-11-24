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

TODO:
- ~~Add tf and exact match features for token~~
- Add previous answer positional embedding and question turn id embedding
- Add previous answer and question in the form
- Update loss function in RL
- Add bert