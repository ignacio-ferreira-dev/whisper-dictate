"""
Audio Alerts Manager
====================

Plays sound files to signal recording state changes.
Uses a subprocess (paplay/aplay) to play MP3 files so that audio playback
runs in a completely separate process from PyAudio. This avoids the malloc
heap corruption crash that occurs when pygame/SDL and PortAudio both access
the audio subsystem in the same process on Linux.

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
import shutil
import subprocess
import threading


_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "..", "sounds")


def _sound_path(filename: str) -> str:
    return os.path.normpath(os.path.join(_SOUNDS_DIR, filename))


def _find_player() -> list:
    """
    Return the command prefix to play an audio file via subprocess.

    Preference order:
      1. paplay  — PulseAudio native player, handles MP3 via GStreamer
      2. aplay   — ALSA player (WAV only, so we convert MP3 path as-is;
                   works if alsa-plugins-pulse is installed)
      3. ffplay  — ffmpeg player, handles everything

    Returns an empty list if no player is found.
    """
    for player in ("paplay", "ffplay", "aplay"):
        if shutil.which(player):
            if player == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            return [player]
    return []


_PLAYER_CMD = _find_player()


class AudioAlertsManager:
    """
    Manages audible feedback for recording state changes.

    Each sound is played by spawning a subprocess so that the audio
    player runs completely outside the Python process. This guarantees
    no shared memory or library state with PyAudio/PortAudio.
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
            volume:  Playback volume in [0.0, 1.0]. Currently informational
                     (subprocess players use system volume).
            enabled: When False, all play calls are silent no-ops.
        """
        self.volume = max(0.0, min(1.0, volume))
        self.enabled = enabled

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
        if not self.enabled or not _PLAYER_CMD:
            return
        threading.Thread(target=self._play_event, args=(event,), daemon=True).start()

    def _play_event(self, event: str) -> None:
        """Play the sound file for the given event via subprocess."""
        filename = self._SOUND_MAP[event]
        path = _sound_path(filename)
        if not os.path.isfile(path):
            return
        self._play_file(path)

    def _play_file(self, path: str) -> None:
        """Invoke the system audio player as a subprocess."""
        try:
            subprocess.run(
                _PLAYER_CMD + [path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass
