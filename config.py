"""
config.py: every setting for training and translating, in one place.

Change values here instead of hunting through the other files.
train.py, translate.py and attention_visual.ipynb all read their settings from get_config().
"""
from pathlib import Path

# The folder this file lives in (your project folder). Every path below is built from it,
# so the scripts work no matter which folder you start them from.
PROJECT_DIR = Path(__file__).resolve().parent


def get_config():
    return {
        # --- training ---
        "batch_size": 8,            # sentence pairs per training step (lower it if the GPU runs out of memory)
        "num_epochs": 20,           # how many times to go through the whole training set
        "lr": 10**-4,               # learning rate: how big each weight update is

        # --- model size (see model.py) ---
        "seq_len": 350,             # every sentence is padded to exactly this many tokens
        "d_model": 512,             # numbers per token

        # --- data ---
        "datasource": "Helsinki-NLP/opus_books",   # dataset on the Hugging Face Hub (sentences from books)
        "lang_src": "en",           # translate FROM English
        "lang_tgt": "it",           # translate TO Italian

        # --- saving and loading ---
        "model_folder": "weights",              # checkpoints are saved here
        "model_basename": "tmodel_",            # checkpoint names: tmodel_00.pt, tmodel_01.pt, ...
        "preload": "latest",                    # resume from: "latest", an epoch like "05", or None to start fresh
        "tokenizer_file": "tokenizer_{0}.json", # {0} becomes the language: tokenizer_en.json, tokenizer_it.json
        "experiment_name": "runs/tmodel",       # TensorBoard log folder

        # --- extras ---
        "validation_examples": 2,   # validation sentences to translate and print after every epoch
        "use_wandb": False,         # also log to Weights & Biases (needs `pip install wandb` and `wandb login`)
    }


def get_weights_file_path(config, epoch: str) -> str:
    """Path of the checkpoint for one epoch, e.g. .../weights/tmodel_05.pt"""
    model_folder = PROJECT_DIR / config["model_folder"]
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    """Path of the newest checkpoint in the weights folder, or None if there isn't one yet."""
    model_folder = PROJECT_DIR / config["model_folder"]
    weights_files = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if len(weights_files) == 0:
        return None
    # sort by the epoch number in the name, so tmodel_9 < tmodel_10 < tmodel_100
    weights_files.sort(key=lambda path: int(path.stem.replace(config["model_basename"], "")))
    return str(weights_files[-1])


def get_tokenizer_path(config, lang: str) -> Path:
    """Path of the saved tokenizer for one language, e.g. .../tokenizer_en.json"""
    return PROJECT_DIR / config["tokenizer_file"].format(lang)
