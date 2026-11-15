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

class LayerNormalization(nn.Module):
    # Step 5: for each token, rescale its d_model numbers to average 0 and spread 1,
    # so values don't blow up or shrink as they pass through many layers.

    def __init__( self, eps: float= 10**-6 ):
        super().__init__()
        self.eps= eps                               # tiny number so we never divide by 0
        self.alpha= nn.Parameter(torch.ones(1))     # learned scale: multiplied
        self.bias= nn.Parameter(torch.zeros(1))     # learned shift: added

    def forward(self, x):
        # x: (Batch, Seq_len, d_model)
        # dim=-1: work along the last axis (one token's d_model numbers)
        # keepdim=True: result is (Batch, Seq_len, 1) so it lines up with x for the subtraction
        mean= x.mean(dim= -1, keepdim= True)
        std = x.std(dim= -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    # Step 6: "desk time". Each token is processed on its own (no looking at other tokens):
    # expand to d_ff, ReLU, dropout, squeeze back to d_model.

    def __init__(self, d_model: int, d_ff: int, dropout: float)-> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1: 512 --> 2048 (expand)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2: 2048 --> 512 (squeeze back)

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        # read it inside out: linear_1 -> relu (negatives become 0) -> dropout -> linear_2
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

