"""
dataset.py: turns raw sentence pairs into the tensors the Transformer needs.

One item = one English sentence + its Italian translation, turned into token IDs:

    encoder_input   [SOS] the cat sat on the mat [EOS] [PAD] [PAD] ...   -> model.encode
    decoder_input   [SOS] il gatto sedeva sul tappeto [PAD] [PAD] ...     -> model.decode
    label           il gatto sedeva sul tappeto [EOS] [PAD] [PAD] ...     -> what the decoder should predict

The label is the decoder input shifted one step left: at every position the decoder
must predict the NEXT word. All three are padded to exactly seq_len tokens, so many
sentences can be stacked into one batch.
"""
import torch
from torch.utils.data import Dataset


def causal_mask(size):
    """
    (1, size, size) mask: True on and below the diagonal.
    Each word may look at itself and the words before it, never the words after it.
    """
    # torch.triu(..., diagonal=1) puts 1s strictly ABOVE the diagonal, i.e. on the future
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0    # flip it: True where looking is allowed


class BilingualDataset(Dataset):
    # A PyTorch Dataset needs two methods:
    #   __len__      how many items there are
    #   __getitem__  give me item number idx
    # The DataLoader in train.py calls __getitem__ for many idx values and stacks the results into a batch.

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds                        # the raw pairs: ds[i]["translation"]["en"] and ["it"]
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # IDs of the special tokens. Each language has its own tokenizer, so look them up in each.
        self.src_sos = tokenizer_src.token_to_id("[SOS]")   # start of sentence
        self.src_eos = tokenizer_src.token_to_id("[EOS]")   # end of sentence
        self.src_pad = tokenizer_src.token_to_id("[PAD]")   # padding (filler)
        self.tgt_sos = tokenizer_tgt.token_to_id("[SOS]")
        self.tgt_eos = tokenizer_tgt.token_to_id("[EOS]")
        self.tgt_pad = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair = self.ds[idx]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        # text --> list of token IDs, e.g. "the cat" --> [5, 912]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # how many [PAD] tokens fill each sequence up to exactly seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2   # encoder input gets [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1   # decoder input gets only [SOS], label only [EOS]

        # a negative number means the sentence doesn't fit
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sentence is too long for seq_len = {self.seq_len}. Raise seq_len in config.py.")

        # [SOS] sentence [EOS] [PAD] ...
        encoder_input = torch.cat(
            [
                torch.tensor([self.src_sos], dtype=torch.int64),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor([self.src_eos], dtype=torch.int64),
                torch.full((enc_num_padding_tokens,), self.src_pad, dtype=torch.int64),
            ],
            dim=0,
        )

        # [SOS] sentence [PAD] ...  (no [EOS]: the decoder never has to read past the last word)
        decoder_input = torch.cat(
            [
                torch.tensor([self.tgt_sos], dtype=torch.int64),
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.full((dec_num_padding_tokens,), self.tgt_pad, dtype=torch.int64),
            ],
            dim=0,
        )

        # sentence [EOS] [PAD] ...  (the answers: decoder_input shifted one step left)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.tgt_eos], dtype=torch.int64),
                torch.full((dec_num_padding_tokens,), self.tgt_pad, dtype=torch.int64),
            ],
            dim=0,
        )

        # double-check that all three are exactly seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,     # (seq_len)
            "decoder_input": decoder_input,     # (seq_len)
            # encoder_mask: 1 for real tokens, 0 for [PAD], so attention ignores the padding.
            # Shape (1, 1, seq_len) so it stretches over (heads, query positions) inside attention.
            "encoder_mask": (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0).int(),
            # decoder_mask: hide [PAD] AND the future.
            # (1, seq_len) & (1, seq_len, seq_len) --> (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.tgt_pad).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,                     # (seq_len)
            "src_text": src_text,               # the original text, for printing
            "tgt_text": tgt_text,
        }
