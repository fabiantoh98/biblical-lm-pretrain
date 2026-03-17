# biblical-lm-pretrain

GPT-2 style language model pre-trained from scratch on public domain biblical and theological texts.

## Overview

A complete pre-training pipeline including data download, BPE tokenizer training, model training with checkpoint resume, and an interactive Gradio frontend for generation and evaluation.

**Architecture:** 12-layer decoder-only transformer, 768 hidden dim, 12 heads, 512 context window, ~93M parameters.

**Corpus:** American Standard Version Bible, Matthew Henry's Complete Commentary, Calvin's Institutes and Commentaries, Spurgeon's devotionals, Augustine's works — all public domain via [CCEL](https://ccel.org).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For development tools (Jupyter, matplotlib):

```bash
uv sync --extra dev
```

> GPU with CUDA 12.8 is expected. PyTorch is pulled from the PyTorch CUDA index automatically.

## Pipeline

### 1. Download corpus

```bash
uv run python scripts/download_data.py
```

Downloads all datasets to `data/raw/`. Already-downloaded files are skipped. Takes ~10 minutes on first run (Calvin commentaries are 46 individual volumes).

### 2. Train tokenizer

```bash
uv run python scripts/train_tokenizer.py
```

Trains a byte-level BPE tokenizer (vocab size 10,000) on the full corpus. Saves `data/tokenizer/tokenizer.json` and `data/tokenizer/config.json`.

### 3. Prepare data

```bash
uv run python scripts/prepare_data.py
```

Tokenizes all documents, shuffles at document level, splits 90/10 train/val, and writes `data/processed/train.bin` and `data/processed/val.bin` as uint16 memory-mapped binary files.

Which datasets are included is controlled by `DataConfig` in `src/biblical_lm/config.py`:

```python
class DataConfig(BaseModel):
    use_asv: bool = True
    use_matthew_henry: bool = True
    use_calvin: bool = True
    use_spurgeon: bool = True
    use_augustine: bool = True
```

Set any flag to `False` and re-run `prepare_data.py` to rebuild without re-downloading.

### 4. Train

```bash
uv run python scripts/train.py
```

Trains for 30 epochs with cosine LR decay and linear warmup. Saves two checkpoints:

| File | When saved |
|------|-----------|
| `outputs/checkpoints/best.pt` | When validation loss improves |
| `outputs/checkpoints/latest.pt` | End of every epoch |

**Resume training** from a checkpoint:

```bash
uv run python scripts/train.py --resume latest
uv run python scripts/train.py --resume outputs/checkpoints/best.pt
```

Training logs to W&B in offline mode by default (`wandb_offline=True` in `TrainingConfig`).

### 5. Generate

```bash
uv run python scripts/generate.py --checkpoint best --prompt "In the beginning"
```

Options:

```
--checkpoint   latest | best | path/to/checkpoint.pt
--prompt       seed text (default: "In the beginning God created")
--max_new_tokens  tokens to generate (default: 200)
--temperature  sampling temperature (default: 0.8)
--top_k        top-k filtering, 0 = disabled (default: 50)
--num_samples  number of independent samples (default: 3)
```

## Gradio UI

```bash
uv run python app.py
```

Opens a local web interface for interactive generation. Supports checkpoint switching (best/latest), temperature and top-k controls, and quick prompt buttons.

## Project Structure

```
biblical-lm-pretrain/
├── app.py                      # Gradio web UI
├── pyproject.toml
├── scripts/
│   ├── download_data.py        # Download corpus from CCEL and GitHub
│   ├── train_tokenizer.py      # Train BPE tokenizer
│   ├── prepare_data.py         # Tokenize and write binary files
│   ├── train.py                # Pre-training loop
│   └── generate.py             # CLI text generation
├── src/biblical_lm/
│   ├── config.py               # ModelConfig, TrainingConfig, DataConfig
│   ├── model.py                # GPT transformer (FlashAttention, weight tying)
│   ├── dataset.py              # MemoryMappedDataset (uint16 memmap)
│   └── generate.py             # Sampling utilities
└── data/                       # gitignored
    ├── raw/                    # Downloaded text files
    ├── processed/              # train.bin, val.bin
    └── tokenizer/              # tokenizer.json, config.json
```

## Configuration

All hyperparameters live in `src/biblical_lm/config.py`. Key defaults:

| Parameter | Value |
|-----------|-------|
| `n_layer` | 12 |
| `n_head` | 12 |
| `n_embd` | 768 |
| `block_size` | 512 |
| `vocab_size` | 10,000 |
| `learning_rate` | 3e-4 |
| `batch_size` | 16 |
| `grad_accum_steps` | 4 (effective batch: 64) |
| `max_epochs` | 30 |
| `dtype` | bfloat16 |

## Data Sources

All texts are public domain.

| Dataset | Source |
|---------|--------|
| American Standard Version Bible | [openbibleinfo/American-Standard-Version-Bible](https://github.com/openbibleinfo/American-Standard-Version-Bible) |
| Matthew Henry's Complete Commentary (6 vols) | [CCEL](https://ccel.org/ccel/henry) |
| Calvin's Institutes + Commentaries (46 vols) | [CCEL](https://ccel.org/ccel/calvin) |
| Spurgeon's devotionals and sermons | [CCEL](https://ccel.org/ccel/spurgeon) |
| Augustine's Confessions, City of God, etc. | [CCEL](https://ccel.org/ccel/augustine) |
