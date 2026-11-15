# Transformer from scratch

An English to Italian translator built on the Transformer from "Attention Is All You Need" (2017), written in PyTorch.

## Files

| File | What it does |
|---|---|
| `model.py` | The Transformer: embeddings, attention, encoder, decoder, and `build_transformer` |
| `config.py` | Every setting: batch size, epochs, languages, file names |
| `dataset.py` | Turns sentence pairs into padded token IDs and masks |
| `decoding.py` | The word-by-word translation loop: greedy decoding and beam search |
| `train.py` | Downloads the data, builds the tokenizers, trains, validates, saves checkpoints |
| `translate.py` | Translates a sentence with a trained checkpoint |
| `attention_visual.ipynb` | Draws what each attention head looks at |

## Run it

```bash
pip install -r requirements.txt

python train.py                      # train; saves weights/tmodel_XX.pt after every epoch
tensorboard --logdir runs            # in a second terminal: watch the loss go down

python translate.py "How are you?"   # translate with the latest checkpoint
python translate.py 34 --beam 3      # sentence 34 from the dataset, using beam search
```

Training picks up from the latest checkpoint automatically (`preload` in `config.py`). Set it to `None` to start from scratch.
