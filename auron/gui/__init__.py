"""Graphical user interface for Auron.

The ``GUIApp`` class provides a Tkinter‑based interface with panes for chat,
control buttons and logs.  It interacts with the assistant controller to send
and receive messages.
"""

from .gui_app import GUIApp  # noqa: F401

__all__ = ["GUIApp"]