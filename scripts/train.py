"""Main pre-training loop for the Biblical LM.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --resume latest
    uv run python scripts/train.py --resume outputs/checkpoints/best.pt

Checkpoints:
    outputs/checkpoints/latest.pt  — saved at end of every epoch (resume target)
    outputs/checkpoints/best.pt    — saved when val_loss improves (best weights)
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from tokenizers import Tokenizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from biblical_lm.config import ModelConfig, TrainingConfig
from biblical_lm.dataset import MemoryMappedDataset
from biblical_lm.generate import generate
from biblical_lm.model import GPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
SAMPLES_DIR = PROJECT_ROOT / "outputs" / "samples"


# Module-level function (not a closure) so functools.partial produces a
# picklable object that LambdaLR can store and restore via state_dict.
def _lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to zero.

    Args:
        current_step: Current optimizer step count.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (warmup + decay).

    Returns:
        LR multiplier in [0.0, 1.0].
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> dict:
    """Capture current RNG states from all random number generators.

    Returns:
        Dict with keys 'torch', 'torch_cuda' (None if no CUDA), 'numpy', 'python'.
    """
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_states(states: dict) -> None:
    """Restore RNG states from a checkpoint dict.

    Args:
        states: Dict returned by get_rng_states().
    """
    torch.set_rng_state(states["torch"])
    if states["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(states["torch_cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    model_config: ModelConfig,
    train_config: TrainingConfig,
) -> None:
    """Save a full training checkpoint.

    Saves model weights, optimizer state, scaler state, scheduler state,
    training counters, best val loss, configs, and RNG states.
    Both best.pt and latest.pt use this same format.

    Args:
        path: Destination file path (overwritten in-place).
        model: GPT model instance.
        optimizer: AdamW optimizer.
        scaler: AMP GradScaler (may be disabled for bf16).
        scheduler: LambdaLR scheduler.
        epoch: Index of the completed epoch (0-based).
        global_step: Total optimizer steps taken so far.
        best_val_loss: Best validation loss seen so far.
        model_config: ModelConfig used to build the model.
        train_config: TrainingConfig used for this run.
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "model": model_config.to_dict(),
            "training": train_config.to_dict(),
        },
        "rng_states": get_rng_states(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    eval_steps: int,
    autocast_ctx: contextlib.AbstractContextManager,
) -> float:
    """Compute mean validation loss over up to eval_steps batches.

    Args:
        model: GPT model.
        val_loader: Validation DataLoader.
        device: Compute device.
        eval_steps: Maximum number of batches to evaluate.
        autocast_ctx: Mixed precision context (autocast or nullcontext).

    Returns:
        Mean cross-entropy loss over the evaluated batches.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in val_loader:
        if n_batches >= eval_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


def generate_samples(
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    epoch: int,
    samples_dir: Path,
) -> None:
    """Generate text from fixed seed prompts and save to disk.

    Args:
        model: GPT model.
        tokenizer: Trained BPE tokenizer for encoding prompts and decoding output.
        device: Compute device.
        epoch: Epoch number used to name the output file.
        samples_dir: Directory to write sample files into.
    """
    prompts = [
        "In the beginning God created",
        "The LORD said unto Moses",
        "For God so loved",
    ]
    samples_dir.mkdir(parents=True, exist_ok=True)
    output_path = samples_dir / f"epoch_{epoch:03d}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(f"=== Prompt: {prompt!r} ===\n")
            ids = tokenizer.encode(prompt).ids
            idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            output = generate(model, idx, max_new_tokens=200, temperature=0.8, top_k=50)
            decoded = tokenizer.decode(output[0].tolist())
            f.write(decoded + "\n\n")

    print(f"  Samples saved to {output_path}")


def build_scheduler(
    optimizer: torch.optim.AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Build a LambdaLR scheduler with linear warmup + cosine decay.

    Must be called both on initial construction and on resume (before
    loading the state dict) because LambdaLR does not serialize the
    lambda function itself — only last_epoch is restored.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (all epochs combined).

    Returns:
        Configured LambdaLR instance.
    """
    return LambdaLR(
        optimizer,
        lr_lambda=functools.partial(
            _lr_lambda,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )


def train(args: argparse.Namespace) -> None:
    """Run the full pre-training loop.

    Args:
        args: Parsed command-line arguments.
    """
    model_config = ModelConfig()
    train_config = TrainingConfig()

    set_seeds(train_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    # data
    data_dir = PROJECT_ROOT / train_config.data_dir
    train_dataset = MemoryMappedDataset(data_dir / "train.bin", model_config.block_size)
    val_dataset = MemoryMappedDataset(data_dir / "val.bin", model_config.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
    )

    # model
    model = GPT(model_config).to(device)
    print(f"Model parameters: {model.num_parameters:,}")

    # optimizer
    optimizer = model.configure_optimizer(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=train_config.betas,
        device_type=device_type,
    )

    # steps per epoch = batches per epoch / grad_accum_steps
    # drop_last=True ensures len(train_loader) is consistent across epochs
    steps_per_epoch = len(train_loader) // train_config.grad_accum_steps
    total_steps = steps_per_epoch * train_config.max_epochs

    scheduler = build_scheduler(optimizer, train_config.warmup_steps, total_steps)

    # GradScaler: only enabled for float16; bf16 does not need dynamic loss scaling
    use_scaler = train_config.dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # mixed precision context
    if train_config.dtype == "bfloat16":
        autocast_ctx: contextlib.AbstractContextManager = torch.amp.autocast(
            "cuda", dtype=torch.bfloat16
        )
    elif train_config.dtype == "float16":
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    # tokenizer for sample generation
    tokenizer = Tokenizer.from_file(str(PROJECT_ROOT / train_config.tokenizer_path))

    # training state
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # resume from checkpoint
    if args.resume:
        ckpt_path = (
            CHECKPOINT_DIR / "latest.pt" if args.resume == "latest" else Path(args.resume)
        )
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path

        print(f"Resuming from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # rebuild scheduler before loading state_dict — LambdaLR only stores
        # last_epoch in its state_dict, not the lambda function itself
        scheduler = build_scheduler(optimizer, train_config.warmup_steps, total_steps)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]
        restore_rng_states(checkpoint["rng_states"])

        print(
            f"  Resumed at epoch {start_epoch}, "
            f"step {global_step}, "
            f"best_val_loss={best_val_loss:.4f}"
        )

    # W&B — anchored to project root to keep all run data in one place
    wandb.init(
        project=train_config.wandb_project,
        mode="offline" if train_config.wandb_offline else "online",
        dir=str(PROJECT_ROOT),
        config={
            "model": model_config.to_dict(),
            "training": train_config.to_dict(),
        },
        resume="allow" if args.resume else None,
    )

    model.train()

    epoch_bar = tqdm(
        range(start_epoch, train_config.max_epochs),
        desc="Epochs",
        unit="epoch",
        initial=start_epoch,
        total=train_config.max_epochs,
    )

    for epoch in epoch_bar:
        epoch_loss_sum = 0.0
        n_optimizer_steps_this_epoch = 0

        optimizer.zero_grad()

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{train_config.max_epochs}",
            unit="batch",
            leave=False,
        )

        for batch_idx, (x, y) in enumerate(batch_bar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_last_batch = batch_idx == len(train_loader) - 1
            should_step = (
                (batch_idx + 1) % train_config.grad_accum_steps == 0
            ) or is_last_batch

            with autocast_ctx:
                _, loss = model(x, y)
                # divide before backward so accumulated gradient == full-batch gradient
                loss = loss / train_config.grad_accum_steps

            scaler.scale(loss).backward()

            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                n_optimizer_steps_this_epoch += 1

                # unscaled loss for logging
                train_loss = loss.item() * train_config.grad_accum_steps
                epoch_loss_sum += train_loss

                lr = scheduler.get_last_lr()[0]
                batch_bar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.2e}", step=global_step)

                if global_step % train_config.log_interval == 0:
                    wandb.log({"train_loss": train_loss, "lr": lr, "step": global_step})

                if global_step % train_config.eval_interval == 0:
                    val_loss = evaluate(
                        model, val_loader, device, train_config.eval_steps, autocast_ctx
                    )
                    val_perplexity = math.exp(min(val_loss, 20.0))
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_perplexity": val_perplexity,
                            "step": global_step,
                        }
                    )
                    tqdm.write(
                        f"  [eval] step {global_step} |"
                        f" val_loss={val_loss:.4f}"
                        f" | perplexity={val_perplexity:.2f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            CHECKPOINT_DIR / "best.pt",
                            model, optimizer, scaler, scheduler,
                            epoch, global_step, best_val_loss,
                            model_config, train_config,
                        )
                        tqdm.write(
                            f"  [checkpoint] best.pt saved"
                            f" (val_loss={best_val_loss:.4f})"
                        )

                    model.train()

        batch_bar.close()

        # end of epoch: always save latest.pt for resume
        mean_epoch_loss = epoch_loss_sum / max(1, n_optimizer_steps_this_epoch)
        save_checkpoint(
            CHECKPOINT_DIR / "latest.pt",
            model, optimizer, scaler, scheduler,
            epoch, global_step, best_val_loss,
            model_config, train_config,
        )
        epoch_bar.set_postfix(mean_loss=f"{mean_epoch_loss:.4f}", best_val=f"{best_val_loss:.4f}")
        tqdm.write(
            f"  [checkpoint] latest.pt saved"
            f" (epoch {epoch + 1}, mean_loss={mean_epoch_loss:.4f})"
        )
        wandb.log({"epoch": epoch + 1, "epoch_loss": mean_epoch_loss})

        # generate text samples every 5 epochs and at the final epoch
        if (epoch + 1) % 5 == 0 or epoch == train_config.max_epochs - 1:
            generate_samples(model, tokenizer, device, epoch + 1, SAMPLES_DIR)

    wandb.finish()
    print("\nTraining complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with attribute 'resume' (str or None).
    """
    parser = argparse.ArgumentParser(description="Pre-train the Biblical LM from scratch.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Resume training from a checkpoint. "
            "Pass 'latest' to auto-load outputs/checkpoints/latest.pt, "
            "or provide an explicit path."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
