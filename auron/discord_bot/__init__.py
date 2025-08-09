"""Discord bridge for Auron.

The Discord integration allows the assistant to participate in Discord
channels.  It receives user messages, processes them via the command
router/LLM and returns responses both as text and (optionally) spoken.
"""

from .discord_bridge import DiscordBridge  # noqa: F401

__all__ = ["DiscordBridge"]