"""
Command routing for Auron assistant.

This module defines a simple extensible router that matches user utterances
against a set of registered patterns and dispatches to internal handler
functions.  Patterns are compiled as case‑insensitive regular expressions.

If no pattern matches, the router indicates that the text should be sent to
the LLM. Each handler should accept the full user text and return either a
string response or ``None`` if it performs an action without text output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Pattern, List, Tuple, Optional

CommandHandler = Callable[[str], Optional[str]]


@dataclass
class Command:
    """Represents an internal command and its handler."""

    pattern: Pattern[str]
    handler: CommandHandler


class CommandRouter:
    """
    Routes user utterances to either internal handlers or the LLM.

    Commands are registered with regular expressions. When ``route`` is called,
    the router tests each command in the order it was added. The first match
    wins. If a command matches, its handler is invoked with the full text and
    the router returns a tuple identifying the target as ``"internal"`` and
    containing the handler's return value. If no command matches, the router
    returns ``("llm", text)`` indicating the utterance should be passed to
    the language model unchanged.
    """

    def __init__(self) -> None:
        self._commands: List[Command] = []

    def add_internal(self, pattern: str | Pattern[str], handler: CommandHandler) -> None:
        """
        Register an internal command.

        Parameters
        ----------
        pattern:
            A regular expression or string. If a string is provided it will be
            compiled with the ``re.IGNORECASE`` flag.
        handler:
            A callable invoked with the full user utterance when the pattern
            matches. It should return a response string or ``None``.
        """
        regex = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        self._commands.append(Command(regex, handler))

    def route(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Determine whether ``text`` matches a known command.

        Returns a tuple ``(target, payload)``. ``target`` is ``"internal"`` if
        an internal command matched or ``"llm"`` otherwise. ``payload`` is either
        the handler's return value (for internal commands) or the original text
        (for LLM commands).
        """
        for cmd in self._commands:
            if cmd.pattern.search(text):
                # Matched internal command
                return ("internal", cmd.handler(text))
        return ("llm", text)