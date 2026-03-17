"""Tokenize the corpus and write train/val binary files.

Which datasets are included is controlled by DataConfig in
src/biblical_lm/config.py. Set the relevant flags there, then re-run this
script to rebuild train.bin and val.bin.

Pipeline:
  1. Load documents for each enabled dataset.
  2. Tokenize each document and append an EOS token.
  3. Shuffle all documents at the document level (reproducible via seed).
  4. Split 90/10 train/val at the document boundary.
  5. Concatenate and write as uint16 numpy memmap binary files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from biblical_lm.config import DataConfig

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
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def load_txt_dir(directory: Path, pattern: str = "*.txt") -> list[str]:
    """Load all .txt files from a directory, one document per file.

    Args:
        directory: Directory to scan for text files.
        pattern: Glob pattern for file matching.

    Returns:
        List of document strings, one per file found.
    """
    docs: list[str] = []
    for path in sorted(directory.glob(pattern)):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs


def load_all_documents(data_config: DataConfig, raw_dir: Path) -> list[str]:
    """Load all enabled corpus documents according to DataConfig flags.

    Args:
        data_config: DataConfig instance controlling which datasets to include.
        raw_dir: Root directory for raw text files (data/raw/).

    Returns:
        Combined list of document strings from all enabled datasets.
    """
    all_docs: list[str] = []

    if data_config.use_asv:
        asv_path = raw_dir / "asv.txt"
        if asv_path.exists():
            docs = load_asv_by_book(asv_path)
            print(f"  ASV: {len(docs)} book documents")
            all_docs.extend(docs)
        else:
            print(f"  ASV: SKIPPED — {asv_path} not found")

    if data_config.use_matthew_henry:
        mh_dir = raw_dir / "matthew_henry"
        if mh_dir.exists():
            docs = load_txt_dir(mh_dir, "vol_*.txt")
            print(f"  Matthew Henry: {len(docs)} volume documents")
            all_docs.extend(docs)
        else:
            print(f"  Matthew Henry: SKIPPED — {mh_dir} not found")

    if data_config.use_calvin:
        calvin_dir = raw_dir / "calvin"
        if calvin_dir.exists():
            docs = load_txt_dir(calvin_dir)
            print(f"  Calvin: {len(docs)} documents")
            all_docs.extend(docs)
        else:
            print(f"  Calvin: SKIPPED — {calvin_dir} not found (run download_data.py)")

    if data_config.use_spurgeon:
        spurgeon_dir = raw_dir / "spurgeon"
        if spurgeon_dir.exists():
            docs = load_txt_dir(spurgeon_dir)
            print(f"  Spurgeon: {len(docs)} documents")
            all_docs.extend(docs)
        else:
            print(f"  Spurgeon: SKIPPED — {spurgeon_dir} not found (run download_data.py)")

    if data_config.use_augustine:
        augustine_dir = raw_dir / "augustine"
        if augustine_dir.exists():
            docs = load_txt_dir(augustine_dir)
            print(f"  Augustine: {len(docs)} documents")
            all_docs.extend(docs)
        else:
            print(f"  Augustine: SKIPPED — {augustine_dir} not found (run download_data.py)")

    return all_docs


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
    del fp

    return len(arr)


def prepare_data(data_config: DataConfig | None = None, seed: int = 42) -> None:
    """Run the full tokenization and binary file preparation pipeline.

    Args:
        data_config: DataConfig controlling which datasets to include.
            Defaults to DataConfig() which enables ASV and Matthew Henry.
        seed: Random seed for reproducible document-level shuffle.

    Raises:
        FileNotFoundError: If the tokenizer is missing.
        RuntimeError: If no documents were loaded from any enabled dataset.
    """
    if data_config is None:
        data_config = DataConfig()

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"{TOKENIZER_PATH} not found. Run scripts/train_tokenizer.py first."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    config = json.loads(TOKENIZER_CONFIG_PATH.read_text())
    eos_token_id: int = config["eos_token_id"]
    print(f"Loaded tokenizer — vocab_size={tokenizer.get_vocab_size()}, eos_id={eos_token_id}")

    print("\nEnabled datasets:")
    all_docs = load_all_documents(data_config, RAW_DIR)

    if not all_docs:
        raise RuntimeError(
            "No documents loaded. Check DataConfig flags and run download_data.py."
        )

    print(f"\nTotal documents before shuffle: {len(all_docs)}")

    random.seed(seed)
    random.shuffle(all_docs)

    split_idx = max(1, int(0.9 * len(all_docs)))
    train_docs = all_docs[:split_idx]
    val_docs = all_docs[split_idx:]
    print(f"Split: {len(train_docs)} train docs, {len(val_docs)} val docs")

    print("\nTokenizing train split ...")
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
