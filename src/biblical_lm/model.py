"""GPT-2 style causal language model implemented from scratch."""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from biblical_lm.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses torch.nn.functional.scaled_dot_product_attention for FlashAttention
    when available (PyTorch >= 2.0), falling back to manual attention otherwise.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj._is_residual_proj = True  # flag for scaled weight init
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal self-attention.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Two-layer feedforward network with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj._is_residual_proj = True  # flag for scaled weight init
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward transformation.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block with pre-norm: LayerNorm → Attention/MLP → residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP sub-layers with residual connections.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style decoder-only transformer for causal language modeling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        # weight tying: output projection shares weights with token embedding
        self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)

        # residual projections use scaled init from GPT-2 paper to control
        # residual stream variance growth with depth
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_is_residual_proj", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters, excluding positional embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.transformer["wpe"].weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the transformer.

        Args:
            idx: Token indices of shape (B, T).
            targets: Optional target token indices of shape (B, T) for loss computation.

        Returns:
            Tuple of (logits, loss). logits has shape (B, T, vocab_size) when targets
            is provided, or (B, 1, vocab_size) for the last position only when targets
            is None. loss is None when targets is None.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer["drop"](
            self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        )

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """Build AdamW optimizer with selective weight decay.

        Weight decay is applied only to 2D+ parameters (weight matrices).
        Biases and LayerNorm parameters are excluded. Parameters are deduplicated
        by identity to correctly handle weight-tied parameters.

        Args:
            weight_decay: L2 regularization coefficient for weight matrices.
            learning_rate: Initial learning rate.
            betas: AdamW beta coefficients (beta1, beta2).
            device_type: 'cuda' or 'cpu'. Enables fused AdamW on CUDA.

        Returns:
            Configured AdamW optimizer.
        """
        seen_ids: set[int] = set()
        decay_params: list[torch.Tensor] = []
        nodecay_params: list[torch.Tensor] = []

        for _name, param in self.named_parameters():
            if not param.requires_grad or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        use_fused = (
            "cuda" in device_type
            and torch.cuda.is_available()
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        extra_kwargs: dict = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_kwargs)
