"""Gradio frontend for the Biblical LM — interactive text generation and evaluation.

Usage:
    uv run python app.py
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from tokenizers import Tokenizer

from biblical_lm.config import ModelConfig
from biblical_lm.generate import generate
from biblical_lm.model import GPT

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# module-level cache so the model is not reloaded on every generation call
_model_cache: dict[str, tuple[GPT, ModelConfig]] = {}
_tokenizer: Tokenizer | None = None


def _get_tokenizer() -> Tokenizer:
    """Load and cache the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {TOKENIZER_PATH}. "
                "Run scripts/train_tokenizer.py first."
            )
        _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    return _tokenizer


def _get_model(checkpoint_name: str) -> tuple[GPT, ModelConfig]:
    """Load and cache a model from a checkpoint.

    Args:
        checkpoint_name: 'best' or 'latest'.

    Returns:
        Tuple of (cached GPT model, ModelConfig).
    """
    if checkpoint_name not in _model_cache:
        ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_name}.pt' not found in {CHECKPOINT_DIR}. "
                "Run scripts/train.py first."
            )
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model_config = ModelConfig.from_dict(checkpoint["config"]["model"])
        model = GPT(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        _model_cache[checkpoint_name] = (model, model_config)

    return _model_cache[checkpoint_name]


def _available_checkpoints() -> list[str]:
    """Return checkpoint names that exist on disk."""
    return [
        name for name in ("best", "latest")
        if (CHECKPOINT_DIR / f"{name}.pt").exists()
    ]


def run_generation(
    prompt: str,
    checkpoint_name: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    num_samples: int,
) -> str:
    """Generate text from the model and return formatted output.

    Args:
        prompt: Seed text for generation.
        checkpoint_name: Which checkpoint to load ('best' or 'latest').
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k cutoff (0 = disabled).
        num_samples: Number of independent samples to generate.

    Returns:
        Formatted string with all samples.
    """
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        model, model_config = _get_model(checkpoint_name)
        tokenizer = _get_tokenizer()
    except FileNotFoundError as exc:
        return str(exc)

    ids = tokenizer.encode(prompt).ids
    if not ids:
        return "Prompt could not be tokenized."

    idx = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    top_k_val = top_k if top_k > 0 else None

    outputs: list[str] = []
    for i in range(num_samples):
        output = generate(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_val,
        )
        decoded = tokenizer.decode(output[0].tolist())
        outputs.append(f"--- Sample {i + 1} ---\n{decoded}")

    return "\n\n".join(outputs)


def get_model_info(checkpoint_name: str) -> str:
    """Return a summary of the loaded model and checkpoint.

    Args:
        checkpoint_name: Which checkpoint to inspect.

    Returns:
        Formatted info string.
    """
    try:
        model, config = _get_model(checkpoint_name)
    except FileNotFoundError as exc:
        return str(exc)

    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    lines = [
        f"Checkpoint : {checkpoint_name}.pt",
        f"Epoch      : {checkpoint['epoch'] + 1}",
        f"Global step: {checkpoint['global_step']:,}",
        f"Best val loss: {checkpoint['best_val_loss']:.4f}",
        f"Parameters : {model.num_parameters:,}",
        f"n_layer    : {config.n_layer}",
        f"n_head     : {config.n_head}",
        f"n_embd     : {config.n_embd}",
        f"block_size : {config.block_size}",
        f"vocab_size : {config.vocab_size}",
        f"Device     : {DEVICE}",
    ]
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""
    available = _available_checkpoints()
    default_checkpoint = "best" if "best" in available else ("latest" if available else "best")

    with gr.Blocks(title="Biblical LM") as demo:
        gr.Markdown("# Biblical LM — Text Generation")
        gr.Markdown(
            "Generate text from a GPT model pre-trained on the ASV Bible"
            " and Matthew Henry's Commentary."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="In the beginning God created",
                    lines=3,
                )
                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        choices=available if available else ["best", "latest"],
                        value=default_checkpoint,
                        label="Checkpoint",
                    )
                    num_samples = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Samples",
                    )
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        minimum=50, maximum=500, value=200, step=50,
                        label="Max new tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                        label="Temperature",
                    )
                    top_k = gr.Slider(
                        minimum=0, maximum=200, value=50, step=5,
                        label="Top-k (0 = disabled)",
                    )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=3):
                output_box = gr.Textbox(
                    label="Generated text",
                    lines=20,
                    interactive=False,
                )

        with gr.Accordion("Model info", open=False):
            info_box = gr.Textbox(
                label="",
                lines=11,
                interactive=False,
                value=get_model_info(default_checkpoint),
            )
            refresh_btn = gr.Button("Refresh info")

        gr.Markdown(
            "**Quick prompts:**"
        )
        with gr.Row():
            for quick_prompt in [
                "In the beginning God created",
                "The LORD said unto Moses",
                "For God so loved",
                "And it came to pass",
                "Blessed is the man",
            ]:
                gr.Button(quick_prompt).click(
                    fn=lambda p=quick_prompt: p,
                    outputs=prompt,
                )

        generate_btn.click(
            fn=run_generation,
            inputs=[prompt, checkpoint_dropdown, max_new_tokens, temperature, top_k, num_samples],
            outputs=output_box,
        )
        refresh_btn.click(
            fn=get_model_info,
            inputs=checkpoint_dropdown,
            outputs=info_box,
        )
        checkpoint_dropdown.change(
            fn=get_model_info,
            inputs=checkpoint_dropdown,
            outputs=info_box,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
