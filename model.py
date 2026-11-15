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

class MultiHeadAttentionBlock (nn.Module):
    # Step 4: "the meeting". Every token gathers information from the tokens that matter to it.
    # Each token makes a Query (what am I looking for?), a Key (what do I contain?)
    # and a Value (what do I hand over?). h heads do this side by side, each on its own slice.
    # The same class is used for encoder self-attention, decoder self-attention and cross-attention;
    # only the q, k, v and mask you pass in change.

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h                      # number of heads
        # the d_model numbers get cut into h equal slices, so it has to divide evenly
        assert d_model % h == 0, " d_model is not divisible by h"

        self.d_k = d_model // h         # numbers per head (512 // 8 = 64)
        self.w_q = nn.Linear(d_model, d_model) # Wq: makes the Queries
        self.w_k = nn.Linear(d_model, d_model) # Wk: makes the Keys
        self.w_v = nn.Linear(d_model, d_model) # Wv: makes the Values

        self.w_o = nn.Linear(d_model, d_model) # Wo: mixes the heads' results back together
        self.dropout = nn.Dropout(dropout)

    # staticmethod = doesn't use self, so it's called as MultiHeadAttentionBlock.attention(...)
    @staticmethod
    def attention (query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # 1. score: compare every Query with every Key (dot product)   2. scale by sqrt(d_k)
        # (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) --> (Batch, h, seq_len, seq_len)
        # (in cross-attention the two seq_lens can differ: target length x source length)
        attention_scores = (query @ key.transpose (-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # where mask is 0 (padding, or future words in the decoder) write a huge negative
            # number, so softmax turns it into ~0% attention
            attention_scores.masked_fill_(mask == 0, -1e9)
        # 3. softmax: each row becomes percentages that add up to 1
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # 4. blend: mix the Values using those percentages
        # (Batch, h, seq_len, seq_len) @ (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, d_k)
        # also return the scores themselves, so we can look at / plot them later
        return(attention_scores @ value), attention_scores


    def forward (self, q, k, v, mask):
        # q, k, v: (Batch, Seq_len, d_model). In self-attention all three are the same x;
        # in the decoder's cross-attention, k and v are the encoder output.
        query = self.w_q(q) # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)   # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)    # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_model)

        # cut into h heads:
        # (Batch, seq_len, d_model) ---> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        # .view reshapes without changing numbers; .transpose(1, 2) puts heads before tokens
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view (key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # glue the heads back together:
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        # .contiguous() re-lays the numbers in memory after transpose so .view works;
        # -1 means "work this size out yourself"
        x= x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # mix what the heads found: (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    # Step 5: "add, don't replace". Output = x + change, where the change comes from a
    # sublayer (attention or feed-forward). Keeps the original signal and helps training.

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout (dropout)
        self.norm = LayerNormalization()

    def forward (self, x, sub_layer):
        # sub_layer is a FUNCTION we get passed (an attention lambda, or the feed-forward block)
        # pre-norm: normalize first, run the sublayer, dropout, then add the original x back
        return x + self.dropout(sub_layer(self.norm(x)))

class EncoderBlock (nn.Module):
    # One encoder layer = attention ("the meeting") + feed-forward ("desk time"),
    # each wrapped in a residual connection. The full Encoder stacks N = 6 of these.
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)-> None:
        super().__init__()                                  # set up nn.Module first
        self.self_attention_block = self_attention_block    # words look at each other
        self.feed_forward_block = feed_forward_block        # each word thinks on its own
        # 2 residual connections: [0] wraps attention, [1] wraps feed-forward
        # (_ means we don't need the loop number, we just want 2 copies)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # x: (Batch, Seq_len, d_model)
        # 1. self-attention: q, k and v are all x, so the sentence attends to itself.
        #    The residual connection wants a function of one input, so the lambda
        #    takes x and fills in the other three arguments (x, x, src_mask).
        #    src_mask makes sure no word looks at [PAD] tokens.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # 2. feed-forward: takes one input already, so pass the block directly
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x    # (Batch, Seq_len, d_model): same shape in and out, so blocks can be stacked

class Encoder (nn.Module):
    # Step 7: the full encoder = N EncoderBlocks in a row + one final LayerNorm.

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers                # nn.ModuleList of N EncoderBlocks
        self.norm = LayerNormalization ()   # final clean-up (needed because we use pre-norm)

    def forward(self, x, mask):
        # x: (Batch, Seq_len, d_model) --> same shape
        for layer in self.layers:
            x = layer (x, mask)             # each block refines x a bit more
        return self.norm(x)


class DecoderBlock(nn.Module):
    # Step 8: one decoder layer. Like an EncoderBlock with one extra step in the middle:
    #   1. masked self-attention: the target sentence looks at itself, but only at earlier words
    #   2. cross-attention: the target looks at the encoder's output (the source sentence)
    #   3. feed-forward: each token thinks on its own
    # Each step is wrapped in its own residual connection, so we need 3 of them.

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block    # target looks at target
        self.cross_attention_block = cross_attention_block  # target looks at source
        self.feed_forward_block = feed_forward_block
        # nn.ModuleList (not nn.Module) holding 3 ResidualConnections: [0] self-attn, [1] cross-attn, [2] feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)  # not used in forward: each ResidualConnection has its own dropout

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x: the target so far, (Batch, tgt_seq_len, d_model)
        # encoder_output: (Batch, src_seq_len, d_model)
        # src_mask hides [PAD] in the source; tgt_mask hides future words (and [PAD]) in the target
        # 1. masked self-attention: q, k, v all come from the target. tgt_mask stops each word
        #    from peeking at the words after it (those are what it has to predict!)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask ))
        # 2. cross-attention: Query from the target (x), Key and Value from the encoder output.
        #    "Which words of the source sentence help me with this target word?"
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # 3. feed-forward: takes one input, so pass the block directly
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x    # (Batch, tgt_seq_len, d_model)

class Decoder(nn.Module):
    # The full decoder = N DecoderBlocks in a row + one final LayerNorm.
    # Every block gets the same encoder_output.

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers                # nn.ModuleList of N DecoderBlocks
        self.norm = LayerNormalization()    # final clean-up (pre-norm, same as the Encoder)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x: (Batch, tgt_seq_len, d_model) --> same shape
        for layer in self.layers:
            x= layer (x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer (nn.Module):
    # Step 11: turn each decoder output vector (d_model numbers) into a score for
    # every word in the target vocabulary, i.e. "which word comes next?"

    def __init__(self, d_model: int, vocab_size: int)-> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # 512 --> one score per word

    def forward (self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, vocab_size)
        # log_softmax = log of the probabilities (numerically safer than plain softmax);
        # the highest one is still the most likely next word
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    # Step 12: holds every piece. Instead of one forward() it has 3 methods, so that when
    # translating you can run the encoder ONCE and then the decoder many times
    # (one new word per step).

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed          # source language: IDs --> vectors
        self.tgt_embed = tgt_embed          # target language: IDs --> vectors
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer= projection_layer

    def encode (self, src, src_mask):
        # source token IDs (Batch, src_seq_len) --> embeddings --> + positions --> encoder
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)      # (Batch, src_seq_len, d_model)

    def decode (self, encoder_output, src_mask, tgt, tgt_mask):
        # target token IDs (Batch, tgt_seq_len) --> embeddings --> + positions --> decoder
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    # (Batch, tgt_seq_len, d_model)

    def project (self, x):
        # (Batch, tgt_seq_len, d_model) --> (Batch, tgt_seq_len, tgt_vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int =8, dropout: float= 0.1, d_ff: int = 2048) -> Transformer:
    # Step 13: the "factory". Creates every layer with the right sizes and wires them together.
    # It's a plain function at the top level (not inside a class), so other files can do:
    #     from model import build_transformer
    # The default sizes are the ones from the paper.

    # create the embedding layers (source and target languages have different vocabularies)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create N encoder blocks. New blocks on every loop pass, so each layer learns its own weights
    encoder_blocks = []
    for _ in range (N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create N decoder blocks: each has self-attention, cross-attention and feed-forward
    decoder_blocks= []
    for _ in range (N):
        decoder_self_attention_block = MultiHeadAttentionBlock (d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append((decoder_block))

    # create the encoder and the decoder. nn.ModuleList (not a plain Python list)
    # so PyTorch can find, train, move and save every block
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer: d_model --> target vocabulary
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters. Xavier picks random starting values scaled to each layer's
    # size, so signals neither explode nor fade at the start of training.
    # p.dim() > 1 = only weight matrices and embedding tables (skip biases and LayerNorm's alpha/bias).
    # The _ at the end of xavier_uniform_ means "change p in place".
    for p in transformer.parameters():
        if p.dim()> 1:
            nn.init.xavier_uniform_(p)

    return transformer
