# Auron Assistant

A Python-based desktop assistant. 
Features voice recognition, speech-to-text, text-to-speech, LLM command routing, and a real-time web UI.  

## Features
- Wake word detection (Picovoice Porcupine)
- Always-on STT with WebRTC VAD and faster-whisper
- TTS via Chatterbox with adjustable voice and model
- Local web UI with logs, chat, subsystem restart (TTS/LLM)
- Unified logging system for CLI and web
- Discord integration *(currently broken / WIP)*

## Requirements
- Python 3.13+
- See `requirements.txt` for dependencies

## Quick Start
1. Clone this repo  
2. Install dependencies:  
   pip install -r requirements.txt

3. Run the assistant:  
   python -m auron.main  
   The web UI will open automatically.

## Community
Join our Discord for updates, help, and discussion:  
https://discord.gg/xNGM9sfvy4

---
*Work in progress — expect bugs. Feedback welcome!*


⚠ **Early Development Notice**  
This project is in a very early stage — most features are not functional yet.  
I’m aware that the base structure was generated with ChatGPT, and that’s intentional.  
The real work — custom adjustments, stability improvements, and making everything run smoothly — is still ahead.  
Stay tuned!
