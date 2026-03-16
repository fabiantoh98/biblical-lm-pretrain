"""Train a byte-level BPE tokenizer on the full corpus."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

RAW_DIR = Path("data/raw")
TOKENIZER_DIR = Path("data/tokenizer")

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]


def collect_corpus_files(raw_dir: Path) -> list[str]:
    """Collect all raw text file paths for tokenizer training.

    Args:
        raw_dir: Directory containing asv.txt and matthew_henry/.

    Returns:
        List of absolute file path strings ready for the tokenizers trainer.
    """
    files: list[str] = []

    asv_path = raw_dir / "asv.txt"
    if asv_path.exists():
        files.append(str(asv_path))
    else:
        print(f"Warning: {asv_path} not found — run scripts/download_data.py first.")

    mh_dir = raw_dir / "matthew_henry"
    if mh_dir.exists():
        vol_files = sorted(mh_dir.glob("vol_*.txt"))
        files.extend(str(p) for p in vol_files)
        if not vol_files:
            print(f"Warning: no Matthew Henry volumes found in {mh_dir}.")
    else:
        print(f"Warning: {mh_dir} not found — run scripts/download_data.py first.")

    return files


def train_tokenizer(files: list[str], output_dir: Path) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the provided corpus files.

    Uses byte-level pre-tokenization (GPT-2 style) so the tokenizer handles
    any Unicode without unknown characters, and NFC normalization for
    consistent Unicode representation across the corpus.

    Args:
        files: List of text file paths to train on.
        output_dir: Directory to save tokenizer.json and config.json.

    Returns:
        The trained Tokenizer instance.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE}) on {len(files)} file(s) ...")
    tokenizer.train(files=files, trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "eos_token_id": tokenizer.token_to_id("<eos>"),
        "unk_token_id": tokenizer.token_to_id("<unk>"),
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Config saved to {config_path}")
    print(f"Final vocab size: {config['vocab_size']}")
    print(f"Special token IDs: {config['special_tokens']}")

    return tokenizer


if __name__ == "__main__":
    files = collect_corpus_files(RAW_DIR)
    if not files:
        raise RuntimeError("No corpus files found. Run scripts/download_data.py first.")
    train_tokenizer(files, TOKENIZER_DIR)
