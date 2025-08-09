"""
Discord bridge to integrate Auron with Discord servers.

This bot listens for messages, routes them through the assistant controller and
relays responses back to Discord.  Voice support is optional and can be
enabled via configuration.  The implementation uses the ``discord.py``
library and runs in its own asynchronous event loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord

logger = logging.getLogger(__name__)


class DiscordBridge(discord.Client):
    """
    A simple Discord client that forwards messages to the assistant.

    Parameters
    ----------
    assistant:
        Reference to the ``AssistantController`` which handles command routing
        and response generation.
    token:
        The bot's authentication token.
    speak:
        Whether the assistant should speak responses in a voice channel.  If
        enabled, the TTS subsystem will synthesise the response.
    listen:
        Reserved for future use; currently ignored.
    """

    def __init__(self, assistant: "AssistantController", token: str, *, speak: bool = False, listen: bool = False) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.assistant = assistant
        self.token = token
        self.speak_enabled = speak
        self.listen_enabled = listen

    async def on_ready(self) -> None:
        logger.info(f"Discord bot ready: logged in as {self.user} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages sent by the bot itself
        if message.author == self.user:
            return
        text = message.content.strip()
        if not text:
            return
        logger.debug(f"Discord message received: {text}")
        loop = asyncio.get_event_loop()
        # Offload processing of the command to the assistant on a thread pool
        response: Optional[str] = await loop.run_in_executor(None, self.assistant.handle_command, text)
        if response:
            try:
                await message.channel.send(response)
            except Exception as e:
                logger.error(f"Failed to send Discord message: {e}", exc_info=True)
        # Optionally speak the response via TTS
        if response and self.speak_enabled:
            try:
                await loop.run_in_executor(None, self.assistant.tts.speak, response)
            except Exception as e:
                logger.error(f"Failed to speak Discord response: {e}", exc_info=True)

    def run_bot(self) -> None:
        """Start the Discord bot event loop.  This method blocks until closed."""
        try:
            logger.info("Starting Discord bot…")
            self.run(self.token)
        except Exception as e:
            logger.error(f"Error while running Discord bot: {e}", exc_info=True)