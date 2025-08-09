"""
Ollama client for LLM integration.

This module provides a thin wrapper around the local Ollama HTTP API.  It
defaults to using the LLaMA 3 model but can be configured via environment
variables.  If the request fails, the client logs an error and returns an
empty string to the caller.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Simple client for the Ollama generate API."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 120.0) -> None:
        # Determine the model name and base URL from environment variables if not provided
        self.model = model or os.getenv("LLM_MODEL", "llama3")
        default_url = "http://localhost:11434/api/generate"
        self.base_url = base_url or os.getenv("OLLAMA_URL", default_url)
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Generate a response for ``prompt`` using the configured model.

        Parameters
        ----------
        prompt:
            The text prompt to send to the LLM.

        Returns
        -------
        str
            The model's response or an empty string on failure.
        """
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            # The API may return different shapes depending on version
            if isinstance(data, dict):
                if "response" in data:
                    return str(data["response"]).strip()
                # compatibility with generic chat API
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    return str(choice.get("text", "")).strip()
            return ""
        except Exception as e:
            logger.error(f"Ollama request failed: {e}", exc_info=True)
            return ""