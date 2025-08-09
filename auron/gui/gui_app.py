"""
Graphical user interface for Auron.

This module defines the ``GUIApp`` class which builds a simple Tkinter
interface.  It presents conversation history, a user input field, a set of
control buttons and a live log view.  Incoming and outgoing messages are
displayed in the conversation pane and logs are streamed into the log pane
via a custom logging handler.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

logger = logging.getLogger(__name__)


class GUIApp:
    """Tkinter GUI for the Auron assistant."""

    def __init__(self, assistant: "AssistantController") -> None:
        self.assistant = assistant
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Auron Assistant")

        # Conversation pane
        self.chat_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=15)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # User input entry and send button
        self.user_entry = ttk.Entry(self.root)
        self.user_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.user_entry.bind("<Return>", self._on_send_click)
        self.send_button = ttk.Button(self.root, text="Send", command=self._on_send_click)
        self.send_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Buttons frame
        self.buttons_frame = ttk.Frame(self.root)
        self.buttons_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Toggle voice recognition
        self.voice_btn = ttk.Button(self.buttons_frame, text="Toggle Voice", command=self._toggle_voice)
        self.voice_btn.grid(row=0, column=0, padx=5, pady=2)
        # Toggle TTS output
        self.tts_btn = ttk.Button(self.buttons_frame, text="Toggle TTS", command=self._toggle_tts)
        self.tts_btn.grid(row=0, column=1, padx=5, pady=2)
        # Toggle Discord integration
        self.discord_btn = ttk.Button(self.buttons_frame, text="Toggle Discord", command=self._toggle_discord)
        self.discord_btn.grid(row=0, column=2, padx=5, pady=2)

        # Log pane
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=10)
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Configure row/column weights for resizing
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        # Attach log handler so logs appear in GUI
        self._attach_log_handler()

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    def _attach_log_handler(self) -> None:
        """Attach a logging handler to direct logs into the GUI log pane."""
        handler = logging.Handler()
        handler.setLevel(logging.DEBUG)

        def emit(record: logging.LogRecord) -> None:
            msg = handler.format(record)
            self.append_log(msg)

        handler.emit = emit  # type: ignore[assignment]
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    # ---------------------------------------------------------------------
    # User input handling
    # ---------------------------------------------------------------------
    def _on_send_click(self, event: Optional[tk.Event] = None) -> None:
        """Called when the user hits return or clicks the send button."""
        text = self.user_entry.get().strip()
        if not text:
            return
        self.user_entry.delete(0, tk.END)
        self.append_chat("User", text)
        # Process the text in a separate thread to avoid blocking the GUI
        threading.Thread(target=self._process_user_input, args=(text,), daemon=True).start()

    def _process_user_input(self, text: str) -> None:
        """Process user input via the assistant and update the chat pane."""
        response = self.assistant.handle_command(text)
        if response:
            self.append_chat("Assistant", response)

    # ---------------------------------------------------------------------
    # Chat and log updates
    # ---------------------------------------------------------------------
    def append_chat(self, role: str, text: str) -> None:
        """Append a line to the chat pane in a thread‑safe manner."""
        def _append() -> None:
            self.chat_text.configure(state=tk.NORMAL)
            self.chat_text.insert(tk.END, f"{role}: {text}\n")
            self.chat_text.see(tk.END)
            self.chat_text.configure(state=tk.DISABLED)

        self.root.after(0, _append)

    def append_log(self, text: str) -> None:
        """Append a log line to the log pane in a thread‑safe manner."""
        def _append() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{text}\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        self.root.after(0, _append)

    # ---------------------------------------------------------------------
    # Button callbacks
    # ---------------------------------------------------------------------
    def _toggle_voice(self) -> None:
        """Toggle voice recognition on or off."""
        if self.assistant.voice_enabled:
            self.assistant.stop_voice_recognition()
        else:
            self.assistant.start_voice_recognition()
        state = "On" if self.assistant.voice_enabled else "Off"
        self.append_log(f"Voice recognition toggled {state}")

    def _toggle_tts(self) -> None:
        """Toggle TTS output on or off."""
        self.assistant.tts_enabled = not self.assistant.tts_enabled
        state = "On" if self.assistant.tts_enabled else "Off"
        self.append_log(f"TTS toggled {state}")

    def _toggle_discord(self) -> None:
        """Toggle Discord bridge on or off."""
        if self.assistant.discord_bridge is None:
            token = os.getenv("DISCORD_TOKEN")
            if token:
                self.assistant.start_discord()
                self.append_log("Discord started")
            else:
                self.append_log("Discord token not configured (.env: DISCORD_TOKEN)")
        else:
            self.assistant.stop_discord()
            self.append_log("Discord stopped")

    # ---------------------------------------------------------------------
    # Run loop
    # ---------------------------------------------------------------------
    def run(self) -> None:
        """Enter the Tkinter main loop.  Returns when the window is closed."""
        self.root.mainloop()