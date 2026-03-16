"""Configuration dataclasses for model architecture and training hyperparameters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    """GPT model architecture hyperparameters.

    Attributes:
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_embd: Hidden dimension size.
        block_size: Maximum context length in tokens.
        vocab_size: Tokenizer vocabulary size.
        dropout: Dropout probability (applied to attention, MLP, and embeddings).
        bias: Whether to use bias in Linear and LayerNorm layers.
    """

    model_config = ConfigDict(frozen=True)

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 512
    vocab_size: int = 10000
    dropout: float = 0.1
    bias: bool = False

    @property
    def padded_vocab_size(self) -> int:
        """Vocab size rounded up to nearest multiple of 64 for CUDA matmul efficiency."""
        return ((self.vocab_size + 63) // 64) * 64

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict for checkpoint storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Reconstruct config from a checkpoint dict."""
        return cls(**data)


class TrainingConfig(BaseModel):
    """Training loop hyperparameters.

    Attributes:
        learning_rate: Peak learning rate for AdamW.
        weight_decay: L2 regularization applied to 2D+ parameters.
        betas: AdamW beta coefficients.
        batch_size: Per-step batch size fed to the model.
        grad_accum_steps: Number of micro-steps before an optimizer step.
            Effective batch size = batch_size * grad_accum_steps.
        max_epochs: Total training epochs.
        warmup_steps: Number of linear LR warmup steps.
        grad_clip: Gradient norm clip threshold.
        dtype: Mixed precision dtype. Use 'bfloat16' on RTX 40xx.
        eval_interval: Run validation every this many optimizer steps.
        eval_steps: Number of val batches to average for loss estimate.
        log_interval: Log training loss every this many optimizer steps.
        checkpoint_dir: Directory for best.pt and latest.pt.
        samples_dir: Directory for generated text samples.
        data_dir: Directory containing train.bin and val.bin.
        tokenizer_path: Path to tokenizer.json.
        wandb_project: W&B project name.
        wandb_offline: Run W&B in offline mode.
        seed: Global random seed.
        num_workers: DataLoader worker processes. Must be 0 on Windows.
    """

    model_config = ConfigDict(frozen=True)

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_epochs: int = 30
    warmup_steps: int = 100
    grad_clip: float = 1.0
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    eval_interval: int = 200
    eval_steps: int = 50
    log_interval: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    samples_dir: str = "outputs/samples"
    data_dir: str = "data/processed"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    wandb_project: str = "biblical-lm-pretrain"
    wandb_offline: bool = True
    seed: int = 42
    num_workers: int = 0  # must be 0 on Windows; increase on Linux

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict for checkpoint storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Reconstruct config from a checkpoint dict."""
        return cls(**data)
