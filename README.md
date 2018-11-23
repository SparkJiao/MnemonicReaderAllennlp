# MnemonicReaderAllennlp
PyTorch implementation of MnemonicReader based on Allennlp

Update:

- version0/output_dir1: hops = 1, no stacked bilstm layer
- version0/output_dir2: hops = 2, no stacked bilstm layer
- version0/output_dir3: hops = 2, add char features rnn
- version0/output_dir4: add ner and pos features embedding
- version0/output_dir5: change the GloVe word vectors of 100 dimension to 300 dimension

TODO:
- Add tf and exact match features for token
- Add previous answer positional embedding and question turn id embedding
- Add previous answer and question in the form
- Update loss function in RL