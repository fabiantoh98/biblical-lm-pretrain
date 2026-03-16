"""Text generation utilities with temperature and top-k sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from biblical_lm.model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Auto-regressively generate tokens from the model.

    Sets the model to eval mode for the duration of generation and uses bfloat16
    autocast for consistency with the training dtype.

    Args:
        model: The GPT model instance.
        idx: Seed token indices of shape (B, T).
        max_new_tokens: Number of new tokens to generate.
        temperature: Softmax temperature. Lower values make the distribution
            more peaked (more deterministic); higher values increase randomness.
        top_k: If set, restrict sampling to the top-k most likely tokens at
            each step, setting all other logits to -inf before softmax.

    Returns:
        Token indices of shape (B, T + max_new_tokens).
    """
    model.eval()
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
