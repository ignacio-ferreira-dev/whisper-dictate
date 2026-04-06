"""
Whisper Dictate Client
======================

Orchestrates the full voice-to-keyboard flow:
  hotkey press → record audio → transcribe → type at cursor

The transcription provider is injected as a TranscriptionBackend,
making it straightforward to swap OpenAI for any other backend.
"""

import asyncio
import os
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("ALSA_PCM_CARD", "0")
os.environ.setdefault("ALSA_PCM_DEVICE", "0")

from pynput import keyboard as kb_module

from whisper_dictate.audio.alerts import AudioAlertsManager
from whisper_dictate.audio.recorder import AudioRecorder
from whisper_dictate.transcription.base import TranscriptionBackend
from whisper_dictate.typing.text_typer import TextTyper


class WhisperDictateClient:
    """
    Coordinates recording, transcription, and text injection.

    Lifecycle:
        client = WhisperDictateClient(backend, ...)
        await client.run()
    """

    MIN_RECORDING_DURATION: float = 0.5  # seconds

    def __init__(
        self,
        backend: TranscriptionBackend,
        hotkey: str = "f9",
        language: str = "auto",
        alerts: Optional[AudioAlertsManager] = None,
        typer: Optional[TextTyper] = None,
        verbose: bool = True,
    ):
        """
        Args:
            backend:  TranscriptionBackend implementation to use.
            hotkey:   Pynput key name for the record toggle (e.g. 'f9', 'f10').
            language: ISO 639-1 language code or 'auto' for auto-detection.
            alerts:   AudioAlertsManager instance (created with defaults if None).
            typer:    TextTyper instance (created with defaults if None).
            verbose:  When True, print status messages.
        """
        self._backend = backend
        self._language = language
        self._verbose = verbose
        self._hotkey = self._resolve_hotkey(hotkey)
        self._hotkey_name = hotkey.upper()

        self._alerts = alerts or AudioAlertsManager()
        self._recorder = AudioRecorder(
            alerts=self._alerts,
            verbose=verbose,
            on_auto_stop=self._on_recorder_auto_stop,
        )
        self._typer = typer or TextTyper()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._should_quit: bool = False
        self._transcribing: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> int:
        """
        Start the client: set up audio, listen for hotkeys, and block
        until ESC is pressed.

        Returns:
            0 on clean exit, 1 on initialization failure.
        """
        self._print_banner()
        self._loop = asyncio.get_running_loop()

        if not self._recorder.setup():
            self._log("Could not initialize audio device")
            return 1

        self._log(f"Ready. Press {self._hotkey_name} to record, ESC to quit.")

        with kb_module.Listener(on_press=self._on_key_press) as listener:
            while not self._should_quit:
                await asyncio.sleep(0.05)
            listener.stop()

        self._recorder.teardown()
        return 0

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _on_key_press(self, key) -> None:
        """Handle key press events from pynput (runs in a background thread)."""
        try:
            if key == self._hotkey:
                if self._transcribing:
                    return  # ignore hotkey while transcription is in progress
                asyncio.run_coroutine_threadsafe(
                    self._toggle_recording(), self._loop
                )
            elif key == kb_module.Key.esc:
                self._log("ESC pressed - quitting...")
                self._should_quit = True
        except Exception:
            pass

    def _on_recorder_auto_stop(self) -> None:
        """
        Called from the recorder's capture thread when recording stops automatically
        (max duration or stream error). Schedules transcription on the event loop.
        """
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._transcribe_pending(), self._loop
            )

    async def _toggle_recording(self) -> None:
        """Start or stop recording depending on the current state."""
        if self._recorder.is_recording:
            frames = self._recorder.stop_recording()
            if frames:
                await self._transcribe_and_type(frames)
            else:
                self._log("No audio captured")
        else:
            self._recorder.start_recording()

    async def _transcribe_pending(self) -> None:
        """Transcribe frames that were buffered by an auto-stopped recording."""
        if self._recorder.is_recording:
            # A new recording started before the callback ran — leave it alone.
            return
        frames = self._recorder.stop_recording()
        if frames:
            await self._transcribe_and_type(frames)

    # ------------------------------------------------------------------
    # Transcription + typing
    # ------------------------------------------------------------------

    async def _transcribe_and_type(self, frames: list) -> None:
        """Send recorded frames to the backend, then type the result."""
        duration = (
            len(frames) * self._recorder.CHUNK_SIZE / self._recorder.sample_rate
        )
        if duration < self.MIN_RECORDING_DURATION:
            self._log(
                f"Recording too short ({duration:.1f}s) - "
                f"speak for at least {self.MIN_RECORDING_DURATION}s"
            )
            self._alerts.play_error()
            return

        self._log("Transcribing...")
        self._transcribing = True

        try:
            text = await self._backend.transcribe(
                frames=frames,
                sample_rate=self._recorder.sample_rate,
                language=None if self._language == "auto" else self._language,
            )
        except Exception as e:
            self._log(f"Transcription error: {e}")
            self._alerts.play_error()
            return
        finally:
            self._transcribing = False

        if not text or not text.strip():
            self._log("No speech detected")
            self._alerts.play_error()
            return

        self._log(f"Transcription: '{text.strip()}'")
        if self._typer.type_text(text.strip()):
            self._alerts.play_done()
            self._log("Text typed at cursor position")
        else:
            self._alerts.play_error()
            self._log("Warning: text typing failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_hotkey(key_name: str):
        """Convert a key name string to the corresponding pynput Key."""
        try:
            return getattr(kb_module.Key, key_name.lower())
        except AttributeError:
            print(f"Warning: unknown key '{key_name}', defaulting to f9")
            return kb_module.Key.f9

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _print_banner(self) -> None:
        print()
        print("=" * 45)
        print("  WHISPER DICTATE - Voice to Keyboard")
        print("=" * 45)
        print(f"  {self._hotkey_name:<4} : Start / Stop recording")
        print(f"  ESC  : Quit")
        print(f"  Language : {self._language}")
        print("=" * 45)
        print()
