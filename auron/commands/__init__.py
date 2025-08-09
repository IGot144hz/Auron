"""Command routing subsystem for Auron.

This package exposes the ``CommandRouter`` class which is used to match
transcribed user utterances against a set of internal commands.  When a
command pattern matches, the corresponding handler is invoked to perform
the action and optionally generate a response.
"""

from .command_router import CommandRouter  # noqa: F401

__all__ = ["CommandRouter"]