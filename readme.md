## Scripts I've collected for loading datasets

### load_wiki

Loads "cleaned" wikipedia data described at http://mattmahoney.net/dc/textdata
as a dataset object with the following properties/methods:
- vocab_size
- count: list of (word string, count) tuples
- data: list of the data, encoded by vocab index
- idx_to_word: dictionary of vocab index --> word string
- word_to_idx: dictionary of word string --> vocab index
- generate_batch(batch_size, context_window_size)

"Cleaned" data uses only the 26 lowercase English
characters. Data comes in two flavors:
- text9: first 10^9 bytes of 2006 English Wikipedia dump (124 million words)
- text8: first 10^8 bytes of the same dump (17 million words)

This code was adapted from step 1 of Google's word2vec tutorial available at:
https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

```python
from load_datasets import load_wiki
text8 = load_wiki.load_text8(target_dir="data", vocab_size = 20000)
text9 = load_wiki.load_text9(target_dir="data", vocab_size = 50000)
```

### load_mnist

Loads MNIST training, validation and test datasets as objects with the following
properties/methods:

- images
- labels
- num_examples
- epochs_completed
- next_batch(batch_size, fake_data=False)

This code is written by Google and is available at: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py

```python
from load_datasets import load_mnist
mnist = load_mnist.read_data_sets(target_dir="data")

train, val, test = mnist.train, mnist.validation, mnist.test
```
