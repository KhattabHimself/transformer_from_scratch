import torch
import torch.nn as nn

import math

# Shape names used in the comments below:
#   Batch   = how many sentences go through at once
#   Seq_len = tokens per sentence (after padding)
#   d_model = numbers per token (512), h = number of heads (8), d_k = d_model // h (64)
#   d_ff    = width inside the feed-forward block (2048)


class InputEmbeddings (nn.Module):
    # Step 2: turn each token ID into a vector of d_model numbers.

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()                                  # set up nn.Module first
        self.d_model = d_model
        self.vocab_size = vocab_size
        # a learnable table: one row per token in the vocabulary, d_model columns
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (Batch, Seq_len) token IDs --> (Batch, Seq_len, d_model)
        # look up each ID's row, then scale by sqrt(d_model) like the paper does
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # Step 3: attention can't tell word order on its own, so we add a fixed
    # "position stamp" (made of sine/cosine waves) to every token's vector.

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len          # longest sentence we'll ever need a stamp for
        self.dropout= nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model): one stamp (row) per position
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1): the positions 0, 1, 2, ... as a column
        position = torch.arange(0, seq_len, dtype= torch.float).unsqueeze(1)
        # one "speed" per pair of columns: early columns wiggle fast, later ones slowly.
        # This is 1 / 10000^(2i / d_model), written with exp and log.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        # sine on even columns (0, 2, 4, ...), cosine on odd columns (1, 3, 5, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, Seq_Len, d_model): extra batch dim so it adds onto any batch

        # buffer = saved with the model and moved to the GPU with it, but NOT a weight to learn
        self.register_buffer('pe', pe)

    def forward (self, x):
        # x: (Batch, Seq_len, d_model)
        # add the stamps for the first Seq_len positions; requires_grad_(False) = never train them
        x = x+ (self.pe[:, :x.shape[1]]).requires_grad_(False)
        return self.dropout(x)

