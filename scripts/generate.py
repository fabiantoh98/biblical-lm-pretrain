"""CLI for generating text from a trained Biblical LM checkpoint.

Usage:
    uv run python scripts/generate.py --checkpoint outputs/checkpoints/best.pt
    uv run python scripts/generate.py --checkpoint latest --prompt "The LORD said"
    uv run python scripts/generate.py --checkpoint best --num_samples 5 --top_k 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from biblical_lm.config import ModelConfig
from biblical_lm.generate import generate
from biblical_lm.model import GPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    """Resolve a checkpoint argument to an absolute path.

    Accepts 'latest', 'best', a relative path, or an absolute path.

    Args:
        checkpoint_arg: Checkpoint argument string from CLI.

    Returns:
        Resolved absolute Path to the checkpoint file.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    if checkpoint_arg in ("latest", "best"):
        path = CHECKPOINT_DIR / f"{checkpoint_arg}.pt"
    else:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT, ModelConfig]:
    """Load a GPT model from a training checkpoint.

    Reconstructs ModelConfig from the saved config dict so no external
    config file is needed.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (model in eval mode, model_config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig.from_dict(checkpoint["config"]["model"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_config


def run_generate(args: argparse.Namespace) -> None:
    """Load a checkpoint and generate text samples.

    Args:
        args: Parsed CLI arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    model, model_config = load_model(ckpt_path, device)
    print(f"Model loaded — {model.num_parameters:,} parameters, block_size={model_config.block_size}")

    tokenizer_path = PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    for i in range(args.num_samples):
        print(f"\n--- Sample {i + 1}/{args.num_samples} ---")
        print(f"Prompt: {args.prompt!r}")

        ids = tokenizer.encode(args.prompt).ids
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        output = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )
        print(tokenizer.decode(output[0].tolist()))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with generation configuration.
    """
    parser = argparse.ArgumentParser(description="Generate text from a Biblical LM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path, or 'latest' / 'best' to auto-resolve from outputs/checkpoints/.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In the beginning God created",
        help="Text prompt to seed generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower = more deterministic).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling. Set to 0 to disable.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of independent samples to generate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_generate(parse_args())
