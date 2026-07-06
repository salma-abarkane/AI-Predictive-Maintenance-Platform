from __future__ import annotations

import os
from typing import Any

import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def list_models(timeout: int = 10) -> list[str]:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return [model["name"] for model in data.get("models", [])]


def resolve_model(model_name: str | None = None) -> str:
    requested = model_name or DEFAULT_OLLAMA_MODEL
    available_models = list_models()
    if requested in available_models:
        return requested
    if available_models:
        return available_models[0]
    return requested


def ollama_generate(prompt: str, model_name: str | None = None, timeout: int = 180) -> dict[str, Any]:
    selected_model = resolve_model(model_name)
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "available": True,
        "model": selected_model,
        "text": data.get("response", "").strip(),
    }


def safe_ollama_generate(prompt: str, model_name: str | None = None, timeout: int = 45) -> dict[str, Any]:
    selected_model = model_name or DEFAULT_OLLAMA_MODEL
    try:
        return ollama_generate(prompt, selected_model, timeout=timeout)
    except Exception as exc:
        return {
            "available": False,
            "model": selected_model,
            "text": "Ollama is unavailable. Start the Ollama server on the host machine or configure OLLAMA_BASE_URL for deployment.",
            "error": str(exc),
        }
