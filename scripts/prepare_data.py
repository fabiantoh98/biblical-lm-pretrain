"""Tokenize the corpus and write train/val binary files.

Pipeline:
  1. Load ASV text split into one document per Bible book.
  2. Load each Matthew Henry volume as one document.
  3. Tokenize each document and append an EOS token.
  4. Shuffle all documents at the document level (reproducible via seed).
  5. Split 90/10 train/val at the document boundary.
  6. Concatenate and write as uint16 numpy memmap binary files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TOKENIZER_PATH = Path("data/tokenizer/tokenizer.json")
TOKENIZER_CONFIG_PATH = Path("data/tokenizer/config.json")


def load_asv_by_book(asv_path: Path) -> list[str]:
    """Load ASV text and return one document string per Bible book.

    download_data.py writes each book as a block of verse lines followed by
    a blank line, so splitting on double newline yields one entry per book.

    Args:
        asv_path: Path to asv.txt.

    Returns:
        List of book text strings (up to 66 entries).
    """
    text = asv_path.read_text(encoding="utf-8")
    books = [block.strip() for block in text.split("\n\n") if block.strip()]
    return books


def load_matthew_henry_volumes(mh_dir: Path) -> list[str]:
    """Load each Matthew Henry volume as a single document string.

    Args:
        mh_dir: Directory containing vol_1.txt through vol_6.txt.

    Returns:
        List of volume text strings.
    """
    docs: list[str] = []
    for vol_path in sorted(mh_dir.glob("vol_*.txt")):
        text = vol_path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs


def tokenize_documents(
    docs: list[str],
    tokenizer: Tokenizer,
    eos_token_id: int,
) -> list[list[int]]:
    """Tokenize each document and append an EOS token to mark document boundaries.

    Args:
        docs: List of plain text documents.
        tokenizer: Trained BPE tokenizer.
        eos_token_id: Token ID to append after each document.

    Returns:
        List of token ID lists, one per document.
    """
    tokenized: list[list[int]] = []
    for doc in tqdm(docs, desc="Tokenizing"):
        ids = tokenizer.encode(doc).ids
        ids.append(eos_token_id)
        tokenized.append(ids)
    return tokenized


def write_bin(token_lists: list[list[int]], output_path: Path) -> int:
    """Concatenate token lists and write to a uint16 memory-mapped binary file.

    Args:
        token_lists: List of token ID lists to concatenate.
        output_path: Destination path for the .bin file.

    Returns:
        Total number of tokens written.
    """
    all_tokens: list[int] = []
    for tokens in token_lists:
        all_tokens.extend(tokens)

    arr = np.array(all_tokens, dtype=np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fp = np.memmap(str(output_path), dtype=np.uint16, mode="w+", shape=(len(arr),))
    fp[:] = arr
    del fp  # flushes and closes the memmap

    return len(arr)


def prepare_data(seed: int = 42) -> None:
    """Run the full tokenization and binary file preparation pipeline.

    Args:
        seed: Random seed for reproducible document-level shuffle.

    Raises:
        FileNotFoundError: If tokenizer or raw corpus files are missing.
    """
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"{TOKENIZER_PATH} not found. Run scripts/train_tokenizer.py first."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    config = json.loads(TOKENIZER_CONFIG_PATH.read_text())
    eos_token_id: int = config["eos_token_id"]
    print(f"Loaded tokenizer — vocab_size={tokenizer.get_vocab_size()}, eos_id={eos_token_id}")

    print("Loading ASV by book ...")
    asv_docs = load_asv_by_book(RAW_DIR / "asv.txt")
    print(f"  {len(asv_docs)} ASV book documents")

    print("Loading Matthew Henry volumes ...")
    mh_docs = load_matthew_henry_volumes(RAW_DIR / "matthew_henry")
    print(f"  {len(mh_docs)} Matthew Henry volume documents")

    all_docs = asv_docs + mh_docs
    print(f"Total documents before shuffle: {len(all_docs)}")

    random.seed(seed)
    random.shuffle(all_docs)

    split_idx = max(1, int(0.9 * len(all_docs)))
    train_docs = all_docs[:split_idx]
    val_docs = all_docs[split_idx:]
    print(f"Split: {len(train_docs)} train docs, {len(val_docs)} val docs")

    print("Tokenizing train split ...")
    train_tokens = tokenize_documents(train_docs, tokenizer, eos_token_id)
    print("Tokenizing val split ...")
    val_tokens = tokenize_documents(val_docs, tokenizer, eos_token_id)

    train_count = write_bin(train_tokens, PROCESSED_DIR / "train.bin")
    val_count = write_bin(val_tokens, PROCESSED_DIR / "val.bin")

    print(f"\nData preparation complete:")
    print(f"  train.bin : {train_count:>12,} tokens")
    print(f"  val.bin   : {val_count:>12,} tokens")
    print(f"  total     : {train_count + val_count:>12,} tokens")


if __name__ == "__main__":
    prepare_data()
