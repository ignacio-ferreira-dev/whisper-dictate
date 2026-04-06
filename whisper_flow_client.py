#!/usr/bin/env python3
"""
Whisper Flow Client
===================

Global voice-to-keyboard tool.

  F9        → start recording  (plays a rising beep)
  F9 again  → stop recording, transcribe, type text at cursor  (plays a falling beep)
  ESC       → quit

Audio is captured from the default microphone, sent to OpenAI Whisper,
and the transcription is typed directly into whichever application has
keyboard focus.

Usage:
    conda activate whisper-flow
    python whisper_flow_client.py

Optional args:
    --hotkey     KEY        Pynput key name for record toggle (default: f9)
    --volume     FLOAT      Alert volume 0.0-1.0 (default: 0.4)
    --no-alerts             Disable audio alerts
    --add-space             Prepend a space before each transcription
    --char-delay FLOAT      Seconds between typed characters (default: 0.01)
    --language   CODE       Whisper language code, e.g. 'es', 'en', 'auto' (default: auto)
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
import wave
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("ALSA_PCM_CARD", "0")
os.environ.setdefault("ALSA_PCM_DEVICE", "0")

from pynput import keyboard as kb_module
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from whisperflow.audio.alerts import AudioAlertsManager
from whisperflow.audio.recorder import AudioRecorder
from whisperflow.typing.text_typer import TextTyper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper Flow Client - voice to keyboard via F9 hotkey"
    )
    parser.add_argument(
        "--hotkey",
        default="f9",
        help="Pynput key name for the record toggle (default: f9)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.4,
        help="Alert beep volume 0.0-1.0 (default: 0.4)",
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable audio alerts",
    )
    parser.add_argument(
        "--add-space",
        action="store_true",
        help="Prepend a space before each typed transcription",
    )
    parser.add_argument(
        "--char-delay",
        type=float,
        default=0.01,
        help="Delay between typed characters in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Whisper language code: 'auto', 'es', 'en', etc. (default: auto)",
    )
    return parser.parse_args()


class WhisperFlowClient:
    """
    Orchestrates recording → transcription → typing.

    Lifecycle:
        client = WhisperFlowClient(args)
        await client.run()
    """

    MIN_RECORDING_DURATION: float = 0.5  # seconds

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._loop: asyncio.AbstractEventLoop = None
        self._should_quit: bool = False

        # Resolve hotkey
        self._hotkey = self._resolve_hotkey(args.hotkey)

        # Sub-components
        self._alerts = AudioAlertsManager(
            volume=args.volume,
            enabled=not args.no_alerts,
        )
        self._recorder = AudioRecorder(alerts=self._alerts, verbose=True)
        self._typer = TextTyper(
            char_delay=args.char_delay,
            add_space_before=args.add_space,
        )
        self._openai: AsyncOpenAI = None
        self._language: str = args.language

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> int:
        """
        Main async loop. Sets up components, starts hotkey listener, then
        waits until ESC is pressed.
        """
        self._print_banner()
        self._loop = asyncio.get_running_loop()

        if not self._setup_openai():
            return 1

        if not self._recorder.setup():
            self._log("Could not initialize audio device")
            return 1

        self._log(f"Ready. Press {self.args.hotkey.upper()} to record, ESC to quit.")

        with kb_module.Listener(
            on_press=self._on_key_press,
        ) as listener:
            while not self._should_quit:
                await asyncio.sleep(0.05)
            listener.stop()

        self._recorder.teardown()
        return 0

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _on_key_press(self, key) -> None:
        """Handle key press events from pynput listener (runs in a thread)."""
        try:
            if key == self._hotkey:
                # Schedule the toggle on the asyncio event loop
                asyncio.run_coroutine_threadsafe(
                    self._toggle_recording(), self._loop
                )
            elif key == kb_module.Key.esc:
                self._log("ESC pressed - quitting...")
                self._should_quit = True
        except Exception:
            pass

    async def _toggle_recording(self) -> None:
        """Start or stop recording depending on current state."""
        if self._recorder.is_recording:
            frames = self._recorder.stop_recording()
            if frames:
                await self._transcribe_and_type(frames)
            else:
                self._log("No audio captured")
        else:
            self._recorder.start_recording()

    # ------------------------------------------------------------------
    # Transcription + typing
    # ------------------------------------------------------------------

    async def _transcribe_and_type(self, frames: list) -> None:
        """Send recorded PCM frames to Whisper and type the result."""
        duration = len(frames) * self._recorder.CHUNK_SIZE / self._recorder.sample_rate
        if duration < self.MIN_RECORDING_DURATION:
            self._log(f"Recording too short ({duration:.1f}s) - speak for at least {self.MIN_RECORDING_DURATION}s")
            self._alerts.play_error()
            return

        self._log("Transcribing...")
        self._alerts.play_processing()

        try:
            text = await self._call_whisper(frames)
        except Exception as e:
            self._log(f"Transcription error: {e}")
            self._alerts.play_error()
            return

        if not text or not text.strip():
            self._log("No speech detected")
            self._alerts.play_error()
            return

        self._log(f"Transcription: '{text.strip()}'")
        typed = self._typer.type_text(text.strip())
        if typed:
            self._log("Text typed at cursor position")
        else:
            self._log("Warning: text typing failed")

    async def _call_whisper(self, frames: list) -> str:
        """
        Convert PCM frames to a WAV temp file and call OpenAI Whisper API.

        Returns:
            Transcription string (may be empty if no speech detected).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            self._write_wav(tmp_path, frames)
            lang = None if self._language == "auto" else self._language

            with open(tmp_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "text",
                }
                if lang:
                    kwargs["language"] = lang

                result = await self._openai.audio.transcriptions.create(**kwargs)

            return result if isinstance(result, str) else str(result)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _write_wav(self, path: str, frames: list) -> None:
        """Write PCM frame list to a WAV file."""
        import pyaudio
        pa_tmp = None
        try:
            pa_tmp = __import__("pyaudio").PyAudio()
            sample_width = pa_tmp.get_sample_size(AudioRecorder.FORMAT)
        finally:
            if pa_tmp:
                pa_tmp.terminate()

        with wave.open(path, "wb") as wf:
            wf.setnchannels(self._recorder.CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(self._recorder.sample_rate)
            wf.writeframes(b"".join(frames))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_openai(self) -> bool:
        """Initialize OpenAI async client from config."""
        try:
            config = get_config()
            api_key = config.get_openai_api_key()
            if not api_key:
                self._log("OPENAI_API_KEY not found in .env - cannot continue")
                return False
            self._openai = AsyncOpenAI(api_key=api_key)
            self._log("OpenAI client ready")
            return True
        except Exception as e:
            self._log(f"OpenAI setup failed: {e}")
            return False

    @staticmethod
    def _resolve_hotkey(key_name: str):
        """
        Convert a string like 'f9' or 'f10' to the corresponding pynput Key.

        Falls back to None (which will never match any key) on unknown names.
        """
        try:
            return getattr(kb_module.Key, key_name.lower())
        except AttributeError:
            print(f"Warning: unknown key '{key_name}', defaulting to f9")
            return kb_module.Key.f9

    def _log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _print_banner(self) -> None:
        hotkey = self.args.hotkey.upper()
        print()
        print("=" * 45)
        print("  WHISPER FLOW - Voice to Keyboard")
        print("=" * 45)
        print(f"  {hotkey}    : Start / Stop recording")
        print(f"  ESC  : Quit")
        print(f"  Language : {self.args.language}")
        print(f"  Alerts   : {'off' if self.args.no_alerts else 'on'}")
        print("=" * 45)
        print()


async def main() -> int:
    args = parse_args()
    client = WhisperFlowClient(args)
    try:
        return await client.run()
    except KeyboardInterrupt:
        print("\nInterrupted - goodbye!")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
