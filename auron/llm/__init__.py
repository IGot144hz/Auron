"""Language model integration for Auron.

This package currently provides an ``OllamaClient`` for interacting with a
locally hosted Ollama server using the LLaMA 3 model.  Additional backends
can be added by implementing a similar interface with a ``generate`` method.
"""

from .ollama_client import OllamaClient  # noqa: F401

__all__ = ["OllamaClient"]