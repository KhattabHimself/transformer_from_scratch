"""
train.py: teaches the Transformer in model.py to translate English to Italian.

Run it from a terminal in the project folder:

    python train.py

What happens:
  1. download the sentence pairs (opus_books, English-Italian) from the Hugging Face Hub
  2. build a tokenizer for each language (or load the ones saved last time)
  3. split the pairs 90% training / 10% validation and wrap them in BilingualDataset (dataset.py)
  4. build the model with build_transformer (model.py)
  5. every epoch: run all training batches (forward -> loss -> backward -> update),
     translate a few validation sentences to show progress, and save a checkpoint

Watch the loss go down in TensorBoard: run `tensorboard --logdir runs` in a second terminal
and open the link it prints.
"""
import shutil
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm   # progress bars

# Hugging Face libraries: `datasets` downloads the data, `tokenizers` splits text into tokens
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import PROJECT_DIR, get_config, get_weights_file_path, latest_weights_file_path, get_tokenizer_path
from dataset import BilingualDataset
from decoding import greedy_decode
from model import build_transformer

# torchmetrics is optional: without it, training still works but CER / WER / BLEU aren't logged
try:
    import torchmetrics
except ImportError:
    torchmetrics = None

# TensorBoard is optional too. A broken TensorFlow install in the environment can make this
# import fail; training still works then, just without the loss charts.
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as error:
    SummaryWriter = None
    TENSORBOARD_ERROR = error


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    """Use a GPU if there is one: NVIDIA (cuda) or Apple (mps). Otherwise the CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda,", torch.cuda.get_device_name(device))
        print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu. Training will be very slow without a GPU.")
    return device


# ---------------------------------------------------------------------------
# Tokenizers and data
# ---------------------------------------------------------------------------

def get_all_sentences(ds, lang):
    """Yield every sentence in one language. A generator: one sentence at a time, so it saves memory."""
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Load the tokenizer for `lang` from disk, or train a new one from the dataset the first time.

    WordLevel = every distinct word gets its own ID. Simple and easy to read; real systems
    split words into smaller pieces (sub-words) so they never meet an unknown word.
    """
    tokenizer_path = get_tokenizer_path(config, lang)
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))     # words it has never seen become [UNK]
    tokenizer.pre_tokenizer = Whitespace()                   # split on spaces and punctuation
    # The special tokens get the first IDs: [UNK]=0, [PAD]=1, [SOS]=2, [EOS]=3.
    # min_frequency=2: a word must appear at least twice to get its own ID.
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """Download the data, build the tokenizers, and return the training and validation DataLoaders."""
    # opus_books only has a "train" split, so we make our own validation split below
    ds_raw = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training and 10% for validation. The fixed seed gives the same split every run,
    # so resuming training never mixes validation sentences into the training set.
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=torch.Generator().manual_seed(42))

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    # Longest sentence in each language, in tokens. seq_len in config.py must be at least this + 2.
    max_len_src = max(len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids) for item in ds_raw)
    max_len_tgt = max(len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids) for item in ds_raw)
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    # A DataLoader picks items from the dataset and stacks them into batches.
    # shuffle=True: a different order every epoch, so the model can't learn the order.
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)    # one sentence at a time for translating

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """Build the Transformer from model.py with the sizes in config.py."""
    return build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config["d_model"])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Sends numbers (loss, scores) to TensorBoard, and to Weights & Biases too if use_wandb is on."""

    def __init__(self, config):
        self.writer = None
        if SummaryWriter is not None:
            self.writer = SummaryWriter(str(PROJECT_DIR / config["experiment_name"]))
        else:
            print(f"Note: TensorBoard logging is off because it couldn't be imported ({TENSORBOARD_ERROR!r}).")
        self.wandb = None
        if config["use_wandb"]:
            import wandb    # imported only when needed, so wandb stays optional
            wandb.init(project="pytorch-transformer", config=config)
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("validation/*", step_metric="global_step")
            self.wandb = wandb

    def log(self, name, value, step):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
        if self.wandb is not None:
            self.wandb.log({name: value, "global_step": step})

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(model, val_dataloader, tokenizer_tgt, max_len, device, print_msg, global_step, logger=None, num_examples=2):
    """Translate a few validation sentences, print them next to the real translation, and log scores."""
    model.eval()    # evaluation mode: switches dropout off

    source_texts, expected, predicted = [], [], []
    console_width = shutil.get_terminal_size((80, 20)).columns

    with torch.no_grad():   # no gradients needed here (we're not learning), so it's faster and uses less memory
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device)   # (1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)     # (1, 1, 1, seq_len)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            # token IDs --> text. decode() drops the special tokens like [SOS] and [EOS].
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().tolist())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg("-" * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if len(predicted) == num_examples:
                print_msg("-" * console_width)
                break

    if logger is not None and torchmetrics is not None:
        # CER: share of characters that are wrong. WER: share of words that are wrong. Lower is better.
        cer = torchmetrics.CharErrorRate()(predicted, expected)
        wer = torchmetrics.WordErrorRate()(predicted, expected)
        # BLEU: how many word sequences match the real translation, from 0 to 1. Higher is better.
        # BLEU accepts several correct translations per sentence, so each one goes in its own list.
        bleu = torchmetrics.BLEUScore()(predicted, [[text] for text in expected])
        logger.log("validation/cer", cer.item(), global_step)
        logger.log("validation/wer", wer.item(), global_step)
        logger.log("validation/bleu", bleu.item(), global_step)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(config):
    device = get_device()
    if torchmetrics is None:
        print("Note: `pip install torchmetrics` to also log CER / WER / BLEU scores.")

    # make sure the weights folder exists
    (PROJECT_DIR / config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    logger = Logger(config)

    # Adam: the optimizer that nudges every weight a little to lower the loss.
    # eps is a tiny number that stops it from ever dividing by zero.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # Resume from a saved checkpoint if there is one, so training can continue where it stopped
    initial_epoch = 0
    global_step = 0     # counts training batches across all epochs (the x-axis in TensorBoard)
    preload = config["preload"]
    if preload == "latest":
        model_filename = latest_weights_file_path(config)
    elif preload:
        model_filename = get_weights_file_path(config, preload)
    else:
        model_filename = None

    if model_filename:
        print(f"Preloading model {model_filename}")
        # map_location: lets weights saved on a GPU load on a CPU too (and the other way round)
        state = torch.load(model_filename, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    # The loss measures how wrong the predictions are.
    #   ignore_index: [PAD] positions don't count (there's nothing to predict there)
    #   label_smoothing=0.1: give 10% of the credit to other words, so the model doesn't get over-confident
    # CrossEntropyLoss applies log_softmax itself. Our ProjectionLayer already returns log_softmax,
    # and applying it twice gives the same numbers, so this is fine.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.train()   # training mode: switches dropout on

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)   # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)   # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)     # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)     # (B, 1, seq_len, seq_len)
            label = batch["label"].to(device)                   # (B, seq_len)

            # 1. Forward pass: encoder -> decoder -> projection.
            #    Every position is predicted at once; the causal mask stops it from cheating.
            encoder_output = model.encode(encoder_input, encoder_mask)                                # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)                                               # (B, seq_len, tgt_vocab_size)

            # 2. Loss: compare every predicted position with the label.
            #    view(-1, vocab) flattens (B, seq_len, vocab) into (B * seq_len, vocab); label becomes (B * seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            logger.log("train/loss", loss.item(), global_step)

            # 3. Backward pass: work out how much each weight contributed to the loss (the gradients)
            loss.backward()

            # 4. Update the weights, then clear the gradients ready for the next batch
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # After every epoch: translate a few validation sentences to see how it's doing
        run_validation(model, val_dataloader, tokenizer_tgt, config["seq_len"], device,
                       lambda msg: batch_iterator.write(msg), global_step, logger, config["validation_examples"])

        # Save a checkpoint, so training can resume later and translate.py can use it
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

    logger.close()


# This block only runs when you start this file directly (python train.py),
# not when another file does `from train import ...`.
if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # hide library warnings so the progress bar stays readable
    config = get_config()
    train_model(config)
