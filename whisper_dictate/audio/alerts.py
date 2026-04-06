"""
Audio Alerts Manager
====================

Plays sound files to signal recording state changes.
Uses pygame.mixer for MP3/WAV playback with a numpy+pyaudio sine-wave
fallback when the sound files are not found.

Sound files live in whisper_dictate/sounds/:
    recording_start.mp3           -> play_start()
    recording_end.mp3             -> play_stop()
    transcription_end_success.mp3 -> play_done()
    transcription_end_error.mp3   -> play_error()

Usage:
    alerts = AudioAlertsManager()
    alerts.play_start()   # recording started
    alerts.play_stop()    # recording stopped
    alerts.play_done()    # transcription succeeded
    alerts.play_error()   # something went wrong
"""

import os
import threading
import time
from typing import List, Optional


_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "..", "sounds")


def _sound_path(filename: str) -> str:
    return os.path.normpath(os.path.join(_SOUNDS_DIR, filename))


class AudioAlertsManager:
    """
    Manages audible feedback for recording state changes.

    Uses pygame.mixer for MP3 playback exclusively. PyAudio is intentionally
    NOT used here — creating a second PyAudio instance while the recorder's
    instance is alive causes heap corruption (malloc crash) on Linux.

    Sound files live in whisper_dictate/sounds/:
        recording_start.mp3           -> play_start()
        recording_end.mp3             -> play_stop()
        transcription_end_success.mp3 -> play_done()
        transcription_end_error.mp3   -> play_error()
    """

    _SOUND_MAP = {
        "start": "recording_start.mp3",
        "stop":  "recording_end.mp3",
        "done":  "transcription_end_success.mp3",
        "error": "transcription_end_error.mp3",
    }

    def __init__(self, volume: float = 0.8, enabled: bool = True):
        """
        Args:
            volume:  Playback volume in [0.0, 1.0].
            enabled: When False, all play calls are silent no-ops.
        """
        self.volume = max(0.0, min(1.0, volume))
        self.enabled = enabled
        self._pygame_ok = self._init_pygame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_start(self) -> None:
        """Play the recording-start sound."""
        self._play_async("start")

    def play_stop(self) -> None:
        """Play the recording-end sound."""
        self._play_async("stop")

    def play_done(self) -> None:
        """Play the transcription-success sound."""
        self._play_async("done")

    def play_error(self) -> None:
        """Play the error sound."""
        self._play_async("error")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_async(self, event: str) -> None:
        """Spawn a daemon thread so playback never blocks the caller."""
        if not self.enabled:
            return
        threading.Thread(target=self._play_event, args=(event,), daemon=True).start()

    def _play_event(self, event: str) -> None:
        """Play the sound file for the given event via pygame."""
        if not self._pygame_ok:
            return
        filename = self._SOUND_MAP[event]
        path = _sound_path(filename)
        if not os.path.isfile(path):
            return
        self._play_file(path)

    def _play_file(self, path: str) -> None:
        """Play an audio file using pygame.mixer (supports MP3 and WAV)."""
        try:
            import pygame
            sound = pygame.mixer.Sound(path)
            sound.set_volume(self.volume)
            channel = sound.play()
            while channel and channel.get_busy():
                time.sleep(0.02)
        except Exception:
            pass

    @staticmethod
    def _init_pygame() -> bool:
        """Initialize pygame.mixer; return True on success."""
        try:
            import pygame
            pygame.mixer.init()
            return True
        except Exception:
            return False
