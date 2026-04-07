"""LLM abstraction — Claude CLI subprocess or local MLX model."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from engram.config import Config


def query_llm(prompt: str, system: str = "", config: Config | None = None) -> str:
    if config is None:
        config = Config.load()

    backend = config.llm.backend
    if backend == "claude_cli":
        return _claude_cli(prompt, system, config.llm.model)
    elif backend == "mlx":
        return _mlx_generate(prompt, system, config.llm.mlx_model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def _claude_cli(prompt: str, system: str, model: str) -> str:
    cmd = ["claude", "-p", prompt, "--model", model]
    if system:
        cmd.extend(["--system", system])
    cmd.append("--no-input")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr[:500]}")
    return result.stdout.strip()


def _mlx_generate(prompt: str, system: str, model: str) -> str:
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise RuntimeError("mlx-lm not installed. Run: pip install mlx-lm")

    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    model_obj, tokenizer = load(model)
    response = generate(model_obj, tokenizer, prompt=full_prompt, max_tokens=2048)
    return response.strip()


def extract_json_from_response(text: str) -> dict | list:
    text = text.strip()
    # try to find JSON in markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # try finding first [ or {
    for i, c in enumerate(text):
        if c in ("[", "{"):
            bracket = "]" if c == "[" else "}"
            depth = 0
            for j in range(i, len(text)):
                if text[j] == c:
                    depth += 1
                elif text[j] == bracket:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j+1])
                        except json.JSONDecodeError:
                            break
            break

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")
