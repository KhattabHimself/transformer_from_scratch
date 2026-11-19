# Transformer tutorials

Three notebooks that continue the ACENET deep learning workshop series after `tf.keras_rnn.ipynb`. They follow the same layout as the workshop notebooks: Step 0 sets up the environment, Step 1 prepares the data, Step 2 defines the network, Step 3 trains it, Step 4 evaluates it and Step 5 tests robustness. The two Keras notebooks add a Step 6 that looks inside the trained model.

Work through them in order, since each one builds on the one before.

| Notebook | What it covers | Run time on a laptop CPU |
|---|---|---|
| `attention_numpy.ipynb` | Self-attention built by hand with NumPy: queries, keys and values, scaling by √d_k, the causal mask, and multi-head attention | a few seconds |
| `tf.keras_transformer_songs.ipynb` | A Transformer encoder on the 'songs' next-letter task from `tf.keras_rnn.ipynb`, with the same robustness test, so it can be compared with the LSTM | about 15 seconds |
| `tf.keras_transformer_dates.ipynb` | An encoder-decoder Transformer that translates English dates ("the 5th of march 2021") into ISO format ("2021-03-05"), with greedy decoding and cross-attention plots | about a minute |

## Requirements

* Python 3 with NumPy and Matplotlib
* TensorFlow 2 for the two Keras notebooks (tested with TensorFlow 2.15)

No GPU and no downloads are needed: the notebooks create all of their own data.

The notebooks are saved with the outputs of a test run, so you can see the expected results before running them. Training starts from random weights, so your numbers and plots may differ slightly.
