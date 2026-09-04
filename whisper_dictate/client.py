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
from typing import List, Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("ALSA_PCM_CARD", "0")
os.environ.setdefault("ALSA_PCM_DEVICE", "0")

from pynput import keyboard as kb_module

from whisper_dictate.audio.alerts import AudioAlertsManager
from whisper_dictate.audio.recorder import AudioRecorder
from whisper_dictate.transcription.base import TranscriptionBackend
from whisper_dictate.typing.text_typer import TextTyper


def resolve_key(key_name: str):
    """
    Convert a key name to the matching pynput key object.

    Accepts a special-key name ('f9', 'home', 'esc') or a single printable
    character ('q'), which becomes a KeyCode so it compares equal to the
    events pynput reports for that character.

    Raises:
        ValueError: if the name matches neither form.
    """
    name = key_name.strip().lower()
    special = getattr(kb_module.Key, name, None)
    if isinstance(special, kb_module.Key):
        return special
    if len(name) == 1:
        return kb_module.KeyCode.from_char(name)
    raise ValueError(f"unknown key name: {key_name!r}")


class WhisperDictateClient:
    """
    Coordinates recording, transcription, and text injection.

    Lifecycle:
        client = WhisperDictateClient(backend, ...)
        await client.run()
    """

    MIN_RECORDING_DURATION: float = 0.5  # seconds

    DEFAULT_HOTKEY: str = "f9"
    #: Quitting is bound to a key that is awkward to hit by accident. ESC is
    #: deliberately *not* used: it is pressed constantly while working and used
    #: to kill the app mid-dictation.
    DEFAULT_QUIT_KEY: str = "home"

    def __init__(
        self,
        backend: TranscriptionBackend,
        hotkey: str = DEFAULT_HOTKEY,
        quit_key: str = DEFAULT_QUIT_KEY,
        language: str = "auto",
        alerts: Optional[AudioAlertsManager] = None,
        typer: Optional[TextTyper] = None,
        verbose: bool = True,
    ):
        """
        Args:
            backend:  TranscriptionBackend implementation to use.
            hotkey:   Pynput key name for the record toggle (e.g. 'f9', 'f10').
            quit_key: Pynput key name that exits the application (e.g. 'home').
            language: ISO 639-1 language code or 'auto' for auto-detection.
            alerts:   AudioAlertsManager instance (created with defaults if None).
            typer:    TextTyper instance (created with defaults if None).
            verbose:  When True, print status messages.
        """
        self._backend = backend
        self._language = language
        self._verbose = verbose

        self._hotkey, self._hotkey_name = self._resolve_or_default(
            hotkey, self.DEFAULT_HOTKEY
        )
        self._quit_key, self._quit_key_name = self._resolve_or_default(
            quit_key, self.DEFAULT_QUIT_KEY
        )

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
        until the quit key is pressed.

        Returns:
            0 on clean exit, 1 on initialization failure.
        """
        self._print_banner()
        self._loop = asyncio.get_running_loop()

        if not self._recorder.setup():
            self._log("Could not initialize audio device")
            return 1

        self._log(
            f"Ready. Press {self._hotkey_name} to record, "
            f"{self._quit_key_name} to quit."
        )

        try:
            with kb_module.Listener(on_press=self._on_key_press) as listener:
                while not self._should_quit:
                    await asyncio.sleep(0.05)
                listener.stop()
        finally:
            self._shutdown()

        return 0

    def _shutdown(self) -> None:
        """
        Release audio resources and signal the exit audibly.

        The recorder is torn down *before* the shutdown sound so the player
        subprocess never overlaps with an open PortAudio stream — that overlap
        is what used to corrupt the heap on Linux/PulseAudio.
        """
        self._recorder.teardown()
        self._alerts.play_shutdown()

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
            elif key == self._quit_key:
                self._log(f"{self._quit_key_name} pressed - quitting...")
                self._should_quit = True
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Never let a listener callback raise: pynput would stop the
            # listener and the app would silently go deaf to the hotkey.
            self._log(f"Key handling error: {e}")

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

    async def _transcribe_and_type(self, frames: List[bytes]) -> None:
        """Send recorded frames to the backend, then type the result."""
        duration = self._recorder.frames_duration(frames)
        if duration < self.MIN_RECORDING_DURATION:
            self._log(
                f"Recording too short ({duration:.1f}s) - "
                f"speak for at least {self.MIN_RECORDING_DURATION}s"
            )
            self._alerts.play_error()
            return

        self._log("Transcribing...")

        # Held until the text has been typed: a hotkey press in the middle of
        # typing would otherwise start a recording that swallows the keystrokes.
        self._transcribing = True
        try:
            await self._transcribe_and_type_unguarded(frames)
        finally:
            self._transcribing = False

    async def _transcribe_and_type_unguarded(self, frames: List[bytes]) -> None:
        """Transcribe and type, assuming the _transcribing guard is already held."""
        try:
            text = await self._backend.transcribe(
                frames=frames,
                sample_rate=self._recorder.sample_rate,
                language=None if self._language == "auto" else self._language,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._log(f"Transcription error: {e}")
            self._alerts.play_error()
            return

        text = (text or "").strip()
        if not text:
            self._log("No speech detected")
            self._alerts.play_error()
            return

        self._log(f"Transcription: '{text}'")
        if self._typer.type_text(text):
            self._alerts.play_done()
            self._log("Text typed at cursor position")
        else:
            self._alerts.play_error()
            self._log("Warning: text typing failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_or_default(self, key_name: str, fallback: str):
        """
        Resolve a key name, falling back to a known-good default.

        Returns:
            (key object, display name) — the display name always matches the
            key that was actually bound, so the banner never lies.
        """
        try:
            return resolve_key(key_name), key_name.upper()
        except ValueError as e:
            print(f"Warning: {e}, defaulting to {fallback}")
            return resolve_key(fallback), fallback.upper()

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _print_banner(self) -> None:
        print()
        print("=" * 45)
        print("  WHISPER DICTATE - Voice to Keyboard")
        print("=" * 45)
        print(f"  {self._hotkey_name:<5}: Start / Stop recording")
        print(f"  {self._quit_key_name:<5}: Quit")
        print(f"  Language : {self._language}")
        print("=" * 45)
        print()
