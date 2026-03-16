# biblical-lm-pretrain

## Overview

A from-scratch GPT pre-training project on public domain Biblical and theological text.
Demonstrates the **self-supervised pre-training** phase of the LLM training pipeline using
causal language modeling (next-token prediction) — no labels, no human feedback, pure
unsupervised learning from raw text.

This project completes the LLM training trifecta:

| Phase | Repo |
|---|---|
| Self-supervised pre-training | **biblical-lm-pretrain** (this repo) |
| Supervised fine-tuning (SFT) | `finetune-llm` |
| Preference learning / RLHF | `llm-preference-learning` |

---

## Learning Goals

- Understand causal language modeling: what next-token prediction learns and why it works
- Implement a GPT-style transformer from scratch (no HuggingFace Trainer)
- Build a full data pipeline: raw text → BPE tokenizer → tokenized binary files
- Understand training dynamics: loss curves, perplexity, learning rate schedules, overfitting
- Observe how a model transitions from random noise to coherent domain-specific text

---

## Corpus

All sources are fully public domain — no copyright concerns.

| Source | Format | Est. Words | Est. Tokens | Where to get |
|---|---|---|---|---|
| American Standard Version (ASV, 1901) | Plain text | ~800K | ~1M | [openbibleinfo/American-Standard-Version-Bible](https://github.com/openbibleinfo/American-Standard-Version-Bible) |
| Matthew Henry's Complete Commentary | Plain text | ~9M | ~12M | [Project Gutenberg](https://www.gutenberg.org) / [sacred-texts.com](https://sacred-texts.com) |
| **Total** | | **~9.8M** | **~13M** | |

### Why ASV + Matthew Henry

- **ASV**: Literal word-for-word translation directly from Hebrew (OT) and Greek NT Westcott-Hort text. The NASB was built on top of this — it is the closest public domain equivalent to a formal equivalence translation.
- **Matthew Henry**: The commentary directly expounds the same text verse-by-verse, adding theological reasoning and rich prose. This creates thematic consistency across the corpus rather than mixing unrelated domains.

### Optional corpus extensions (if more data is needed)

| Source | Est. Tokens | Notes |
|---|---|---|
| Young's Literal Translation (YLT) | ~1M | Extremely literal, different vocabulary profile |
| Darby Translation | ~1M | Scholarly, public domain |
| Early Church Fathers (CCEL / Gutenberg) | ~5M | Augustine, Chrysostom, etc. |

---

## Architecture

A GPT-2 small equivalent implemented from scratch, not loaded from HuggingFace.

| Hyperparameter | Value | Notes |
|---|---|---|
| `n_layer` | 12 | Transformer blocks |
| `n_head` | 12 | Attention heads |
| `n_embd` | 768 | Hidden dimension |
| `block_size` | 512 | Context length (tokens) |
| `vocab_size` | ~8K–16K | Trained BPE tokenizer on corpus |
| **Total params** | **~124M** | GPT-2 small scale |

### Why build from scratch

The goal is to understand pre-training mechanics. Using HuggingFace `Trainer` abstracts away
exactly what needs to be learned. Building the training loop manually exposes:
- How gradients flow through a transformer
- Why learning rate warmup matters
- How perplexity relates to loss
- When and why overfitting occurs on small corpora

---

## Training Setup

| Setting | Value | Reason |
|---|---|---|
| dtype | bf16 | Halves VRAM vs fp32, RTX 4070 native support |
| Optimizer | AdamW | Standard for transformers |
| Learning rate | 3e-4 | GPT-2 paper baseline |
| LR schedule | Cosine decay with warmup | Prevents early instability |
| Warmup steps | 100 | ~1% of total steps |
| Gradient clipping | 1.0 | Prevents gradient explosion |
| Batch size | 16 | With grad accumulation steps=4, effective=64 |
| Epochs | 20–30 | Monitor val perplexity, stop when plateau |
| Experiment tracking | W&B (offline mode) | Consistent with other repos |

### Checkpoint Strategy

Two checkpoint files are maintained at all times:

| File | When saved | Purpose |
|---|---|---|
| `outputs/checkpoints/best.pt` | When `val_loss` improves | Best generalizing weights — use for inference and SFT |
| `outputs/checkpoints/latest.pt` | End of every epoch | Always-current snapshot — use for training resume |

Both files are overwritten in-place (no epoch-numbered copies) to keep disk usage flat.
Per-epoch numbered copies (`epoch_005.pt`, etc.) are optional and off by default.

#### Checkpoint contents

Each `.pt` file is a dict saved via `torch.save`:

```python
{
    "epoch": int,                  # completed epoch index (0-based)
    "global_step": int,            # total optimizer steps taken
    "model_state_dict": ...,       # model weights
    "optimizer_state_dict": ...,   # AdamW momentum + variance (fp32)
    "scaler_state_dict": ...,      # AMP loss scaler state
    "scheduler_state_dict": ...,   # cosine LR schedule position
    "best_val_loss": float,        # tracked to gate best.pt saves
    "config": dict,                # ModelConfig + TrainingConfig snapshot
    "rng_states": {                # for reproducible resume
        "torch": ...,
        "torch_cuda": ...,
        "numpy": ...,
        "python": ...,
    },
}
```

#### Resume behaviour

Pass `--resume latest` or `--resume path/to/checkpoint.pt` to `scripts/train.py`.

On resume:
1. Load all state dicts (model, optimizer, scaler, scheduler)
2. Restore `epoch` and `global_step` counters
3. Restore RNG states
4. Skip the dataloader forward by `global_step % steps_per_epoch` batches so data ordering is consistent

### VRAM breakdown (bf16, 124M params, batch=16, seq=512)

| Component | Size |
|---|---|
| Parameters | ~250MB |
| Gradients | ~250MB |
| Adam states (fp32) | ~1.5GB |
| Activations | ~1.5–2GB |
| **Total** | **~3.5–4.5GB** |

Fits comfortably within 7GB usable VRAM.

### Expected training time (RTX 4070 Laptop)

| Epochs | Token exposures | Estimated time |
|---|---|---|
| 10 | 130M | ~1.5 hrs |
| 20 | 260M | ~3 hrs |
| 30 | 390M | ~4.5 hrs |

All well within a 2-day window.

---

## Evaluation

### Quantitative

- **Perplexity** on held-out validation set (10% of corpus) at each checkpoint
- Plot train vs val perplexity curves to detect overfitting

### Qualitative

- Generate text samples at regular checkpoints (every 5 epochs) with the same seed prompt
- Observe how coherence improves: random noise → partial words → coherent Biblical prose

### Prompts for generation checkpoints

```
Prompt 1: "In the beginning God created"
Prompt 2: "The LORD said unto Moses"
Prompt 3: "For God so loved"
```

---

## Project Structure

```
biblical-lm-pretrain/
├── pyproject.toml
├── README.md
├── LEARNING_PLAN.md
│
├── data/
│   ├── raw/                    # downloaded plain text files
│   │   ├── asv.txt
│   │   └── matthew_henry/
│   ├── processed/              # tokenized binary files
│   │   ├── train.bin
│   │   └── val.bin
│   └── tokenizer/              # trained BPE tokenizer files
│
├── src/
│   └── biblical_lm/
│       ├── __init__.py
│       ├── config.py           # Pydantic config for model + training
│       ├── model.py            # GPT architecture from scratch
│       ├── dataset.py          # MemoryMappedDataset for binary files
│       └── generate.py         # Text generation / sampling utilities
│
├── scripts/
│   ├── download_data.py        # Download ASV + Matthew Henry
│   ├── train_tokenizer.py      # Train BPE tokenizer on corpus
│   ├── prepare_data.py         # Tokenize corpus → train.bin / val.bin
│   ├── train.py                # Main training loop
│   └── generate.py             # Run inference on a trained checkpoint
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Token counts, vocab analysis
│   ├── 02_architecture_walkthrough.ipynb # GPT internals explained
│   └── 03_training_analysis.ipynb      # Loss curves, perplexity, samples
│
└── outputs/
    ├── checkpoints/
    │   ├── best.pt             # Best val_loss weights (overwritten in-place)
    │   └── latest.pt           # End-of-epoch snapshot for training resume
    └── samples/                # Generated text at each checkpoint
```

---

## Implementation Phases

### Phase 1: Data Pipeline

**Goal**: Go from raw text files to tokenized binary files ready for training.

1. `scripts/download_data.py`
   - Download ASV from GitHub (openbibleinfo)
   - Download Matthew Henry from Project Gutenberg
   - Strip headers, footnotes, verse numbers into clean plain text

2. `scripts/train_tokenizer.py`
   - Train a BPE tokenizer on the full corpus using HuggingFace `tokenizers`
   - Target vocab size: 8K–16K (small corpus, no need for 50K GPT-2 vocab)
   - Save tokenizer to `data/tokenizer/`

3. `scripts/prepare_data.py`
   - Tokenize full corpus with trained tokenizer
   - Concatenate all tokens into a single array
   - Split 90/10 train/val
   - Write to `data/processed/train.bin` and `val.bin` as memory-mapped numpy arrays

4. `notebooks/01_data_exploration.ipynb`
   - Token count distribution
   - Most frequent tokens
   - Average sequence lengths
   - Sample decoded passages

### Phase 2: Model Architecture

**Goal**: Implement a GPT-2 style transformer from scratch.

1. `src/biblical_lm/model.py`
   - `CausalSelfAttention`: multi-head attention with causal mask
   - `MLP`: two-layer feedforward with GELU activation
   - `Block`: attention + MLP with pre-norm (LayerNorm)
   - `GPT`: token embeddings + positional embeddings + N blocks + LM head

2. `src/biblical_lm/config.py`
   - `ModelConfig`: n_layer, n_head, n_embd, block_size, vocab_size
   - `TrainingConfig`: lr, batch_size, epochs, grad_accum, dtype

3. `notebooks/02_architecture_walkthrough.ipynb`
   - Visualise attention patterns on Bible text
   - Explain each component with annotated forward pass

### Phase 3: Training Loop

**Goal**: Train the model and produce checkpoints.

1. `src/biblical_lm/dataset.py`
   - `MemoryMappedDataset`: reads directly from `.bin` files without loading into RAM
   - Yields fixed-length `(x, y)` pairs where `y = x shifted by 1`

2. `scripts/train.py`
   - AdamW optimizer with cosine LR decay + warmup
   - bf16 mixed precision with `torch.amp`
   - Gradient clipping
   - Eval loop every N steps: compute val perplexity
   - Save `latest.pt` at end of every epoch
   - Save `best.pt` whenever `val_loss` improves
   - `--resume latest` or `--resume <path>` flag to continue interrupted runs
   - W&B logging (offline mode)

### Phase 4: Evaluation and Analysis

**Goal**: Understand what the model learned.

1. `scripts/generate.py`
   - Load checkpoint
   - Temperature and top-k sampling
   - Generate fixed-prompt samples for comparison across checkpoints

2. `notebooks/03_training_analysis.ipynb`
   - Train vs val perplexity curves
   - Side-by-side generated samples at epoch 1, 5, 10, 20, 30
   - Identify overfitting onset epoch

### Phase 5: Optional SFT Extension

After pre-training, use the trained weights as a base for supervised fine-tuning
on a Bible Q&A dataset — completing the pre-train → SFT pipeline and connecting
this repo to `finetune-llm`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` (CUDA) | Training backend |
| `tokenizers` | BPE tokenizer training |
| `numpy` | Memory-mapped binary files |
| `pydantic` | Config validation |
| `wandb` | Experiment tracking |
| `datasets` | Optional: HuggingFace datasets for SFT phase |
| `jupyter` | Notebooks |
| `requests` / `beautifulsoup4` | Data download scripts |

---

## Hardware

- **GPU**: RTX 4070 Laptop, 8GB VRAM (~7GB usable)
- **Expected peak VRAM**: ~4–4.5GB
- **Expected training time**: 3–5 hours for 20–30 epochs
- **CUDA**: 12.9, PyTorch cu128
- **Package manager**: uv
