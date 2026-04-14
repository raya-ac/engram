"""LLM abstraction — Claude CLI, Anthropic API, OpenAI API, or local MLX model."""

from __future__ import annotations

import json
import os
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
    elif backend == "anthropic":
        return _anthropic_api(prompt, system, config.llm.model, config.llm.api_key)
    elif backend == "openai":
        return _openai_api(prompt, system, config.llm.model, config.llm.api_key)
    elif backend == "mlx":
        return _mlx_generate(prompt, system, config.llm.mlx_model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Use: claude_cli, anthropic, openai, mlx")


def _claude_cli(prompt: str, system: str, model: str) -> str:
    cmd = ["claude", "-p", prompt, "--model", model]
    if system:
        cmd.extend(["--system", system])
    cmd.append("--no-input")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr[:500]}")
    return result.stdout.strip()


def _anthropic_api(prompt: str, system: str, model: str, api_key: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic not installed. Run: pip install 'engram-memory-system[anthropic]'")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("No Anthropic API key. Set ANTHROPIC_API_KEY or config llm.api_key")

    client = anthropic.Anthropic(api_key=key)
    kwargs: dict = {"model": model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text.strip()


def _openai_api(prompt: str, system: str, model: str, api_key: str) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai not installed. Run: pip install 'engram-memory-system[openai]'")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY or config llm.api_key")

    client = openai.OpenAI(api_key=key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=4096)
    return response.choices[0].message.content.strip()


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
