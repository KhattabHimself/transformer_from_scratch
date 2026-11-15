"""
decoding.py: turning a trained model into an actual translation.

The model only ever predicts the NEXT word. To get a whole sentence you call it in a loop,
adding one word at a time, until it says [EOS]. There are two ways to pick each word:

    greedy_decode        always take the single most likely word (fast and simple)
    beam_search_decode   keep the best `beam_size` partial sentences and extend all of them
                         (slower, but it can recover from a bad early choice)

Both are used by train.py (to check progress after each epoch) and translate.py.
"""
import torch

from dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """
    source:      (1, seq_len) token IDs of ONE source sentence
    source_mask: (1, 1, 1, seq_len)
    returns:     1-D tensor of target token IDs, starting with [SOS]
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Run the encoder ONCE. Every decoding step reuses its output.
    encoder_output = model.encode(source, source_mask)

    # The translation starts as just [SOS]: shape (1, 1)
    decoder_input = torch.full((1, 1), sos_idx, dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        # causal mask for the words we have so far
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # run the decoder on everything written so far: (1, current_len, d_model)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # only the LAST position predicts the word we don't have yet
        log_probs = model.project(out[:, -1])                   # (1, tgt_vocab_size)
        next_word = log_probs.argmax(dim=-1, keepdim=True)      # (1, 1): ID of the most likely word

        # append it and go again
        decoder_input = torch.cat([decoder_input, next_word], dim=1)

        if next_word.item() == eos_idx:     # the model says the sentence is finished
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, beam_size, source, source_mask, tokenizer_tgt, max_len, device):
    """
    Like greedy_decode, but keeps the `beam_size` best partial translations at every step.

    A candidate's score is the sum of the log-probabilities of its words. model.project returns
    log-probabilities, and adding logs is the same as multiplying probabilities, so a higher
    score means a more likely sentence.
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    start = torch.full((1, 1), sos_idx, dtype=source.dtype, device=device)

    # Each candidate is (token IDs so far, score). An empty sentence has probability 1, and log(1) = 0.
    candidates = [(start, 0.0)]

    # Every extra word adds a negative number to the score, so raw scores favour short sentences.
    # Comparing the average score per word is fairer.
    def average_score(candidate):
        tokens, score = candidate
        return score / tokens.size(1)

    while True:
        # stop once a candidate is as long as allowed
        if any(tokens.size(1) >= max_len for tokens, _ in candidates):
            break

        new_candidates = []
        for tokens, score in candidates:
            # a finished sentence is kept as it is, so it can still win
            if tokens[0, -1].item() == eos_idx:
                new_candidates.append((tokens, score))
                continue

            mask = causal_mask(tokens.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, tokens, mask)
            log_probs = model.project(out[:, -1])                                   # (1, tgt_vocab_size)

            # the beam_size best next words for this candidate
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_size, dim=1)      # (1, beam_size) each
            for i in range(beam_size):
                token = topk_idx[0, i].view(1, 1)
                new_tokens = torch.cat([tokens, token], dim=1)
                new_candidates.append((new_tokens, score + topk_log_probs[0, i].item()))

        # keep only the beam_size best candidates
        candidates = sorted(new_candidates, key=average_score, reverse=True)[:beam_size]

        # stop when every remaining candidate has finished
        if all(tokens[0, -1].item() == eos_idx for tokens, _ in candidates):
            break

    best_tokens, _ = max(candidates, key=average_score)
    return best_tokens.squeeze(0)
