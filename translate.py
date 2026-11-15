"""
translate.py: translate a sentence with your trained model.

    python translate.py "I am not a very good student."
    python translate.py 34                     # sentence number 34 from the dataset, shown next to the real translation
    python translate.py "Good morning" --beam 3  # beam search instead of greedy decoding

It needs a trained checkpoint in the weights folder and the tokenizer files, so run train.py first.
"""
import argparse

import torch
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path, get_tokenizer_path
from decoding import greedy_decode, beam_search_decode
from model import build_transformer


def load_model(config, device):
    """Rebuild the model with the same sizes as in training, then load the trained weights into it."""
    tokenizer_src = Tokenizer.from_file(str(get_tokenizer_path(config, config["lang_src"])))
    tokenizer_tgt = Tokenizer.from_file(str(get_tokenizer_path(config, config["lang_tgt"])))

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                              config["seq_len"], config["seq_len"], d_model=config["d_model"]).to(device)

    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError("No trained weights found in the weights folder. Run `python train.py` first.")
    print("Loading weights:", model_filename)
    # map_location: lets weights saved on a GPU load on a CPU too
    state = torch.load(model_filename, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()    # switch dropout off
    return model, tokenizer_src, tokenizer_tgt


def encode_sentence(sentence, tokenizer_src, seq_len, device):
    """Text --> (1, seq_len) token IDs laid out as [SOS] sentence [EOS] [PAD] ..., plus its mask."""
    ids = tokenizer_src.encode(sentence).ids
    sos = tokenizer_src.token_to_id("[SOS]")
    eos = tokenizer_src.token_to_id("[EOS]")
    pad = tokenizer_src.token_to_id("[PAD]")

    num_padding = seq_len - len(ids) - 2
    if num_padding < 0:
        raise ValueError(f"Sentence is too long: {len(ids)} tokens, the limit is {seq_len - 2}.")

    # .unsqueeze(0) adds the batch dimension: (seq_len) --> (1, seq_len)
    source = torch.tensor([sos] + ids + [eos] + [pad] * num_padding, dtype=torch.int64, device=device).unsqueeze(0)
    source_mask = (source != pad).unsqueeze(1).unsqueeze(1).int()   # (1, 1, 1, seq_len): 0 on [PAD]
    return source, source_mask


def translate(sentence, beam_size=1):
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, tokenizer_src, tokenizer_tgt = load_model(config, device)

    # A number means "take that sentence from the dataset", so we can compare with the real translation
    label = None
    if str(sentence).isdigit():
        from datasets import load_dataset   # only needed in this case
        idx = int(sentence)
        ds = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")
        sentence = ds[idx]["translation"][config["lang_src"]]
        label = ds[idx]["translation"][config["lang_tgt"]]

    source, source_mask = encode_sentence(sentence, tokenizer_src, config["seq_len"], device)

    with torch.no_grad():   # no training here, so no gradients needed
        if beam_size > 1:
            out = beam_search_decode(model, beam_size, source, source_mask, tokenizer_tgt, config["seq_len"], device)
        else:
            out = greedy_decode(model, source, source_mask, tokenizer_tgt, config["seq_len"], device)

    translation = tokenizer_tgt.decode(out.tolist())    # token IDs --> text (drops [SOS] and [EOS])

    print(f"{'SOURCE: ':>12}{sentence}")
    if label is not None:
        print(f"{'TARGET: ':>12}{label}")
    print(f"{'PREDICTED: ':>12}{translation}")
    return translation


if __name__ == "__main__":
    # argparse reads the words you type after `python translate.py`
    parser = argparse.ArgumentParser(description="Translate English to Italian with your trained Transformer.")
    parser.add_argument("sentence", nargs="?", default="I am not a very good student.",
                        help="the text to translate, or a dataset index such as 34")
    parser.add_argument("--beam", type=int, default=1,
                        help="beam size for beam search; 1 means greedy decoding (the default)")
    args = parser.parse_args()
    translate(args.sentence, beam_size=args.beam)
