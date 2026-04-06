"""
Audio Alerts Manager
====================

Plays sound files to signal recording state changes.
Uses pygame.mixer for MP3/WAV playback with a numpy+pyaudio sine-wave
fallback when the sound files are not found.

Sound files expected in <project_root>/sounds/:
    recording_start.mp3       -> play_start()
    recording_end.mp3         -> play_stop()
    transcription_end_success.mp3  -> play_done()
    transcription_end_error.mp3    -> play_error()

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

import numpy as np
import pyaudio


# ---------------------------------------------------------------------------
# Resolve the sounds/ folder relative to the project root (two levels up
# from this file: whisperflow/audio/alerts.py -> project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SOUNDS_DIR = os.path.join(_PROJECT_ROOT, "sounds")


def _sound_path(filename: str) -> str:
    return os.path.join(_SOUNDS_DIR, filename)


class AudioAlertsManager:
    """
    Manages audible feedback for recording state changes.

    Tries to play the custom MP3 files via pygame.mixer first.
    Falls back to synthesized sine-wave tones if pygame is unavailable
    or the file is missing. Every play call is non-blocking (daemon thread).
    """

    # Fallback sine-wave settings
    _SAMPLE_RATE: int = 44100
    _FORMAT = pyaudio.paFloat32
    _CHANNELS: int = 1

    # Map each event to its sound file and fallback tones [(freq_hz, duration_s)]
    _SOUND_MAP = {
        "start":   ("recording_start.mp3",           [(880, 0.08), (1320, 0.12)]),
        "stop":    ("recording_end.mp3",              [(660, 0.15)]),
        "done":    ("transcription_end_success.mp3",  [(880, 0.10), (1100, 0.14)]),
        "error":   ("transcription_end_error.mp3",    [(220, 0.10), (180, 0.10)]),
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
        thread = threading.Thread(
            target=self._play_event,
            args=(event,),
            daemon=True,
        )
        thread.start()

    def _play_event(self, event: str) -> None:
        """Play a sound file or fall back to a synthesized tone."""
        filename, fallback_tones = self._SOUND_MAP[event]
        path = _sound_path(filename)

        if self._pygame_ok and os.path.isfile(path):
            self._play_file(path)
        else:
            self._play_tones(fallback_tones)

    def _play_file(self, path: str) -> None:
        """Play an audio file using pygame.mixer (supports MP3 and WAV)."""
        try:
            import pygame
            sound = pygame.mixer.Sound(path)
            sound.set_volume(self.volume)
            channel = sound.play()
            # Wait until playback finishes so the daemon thread stays alive
            while channel and channel.get_busy():
                time.sleep(0.02)
        except Exception:
            pass  # Best-effort; never crash the main flow

    def _play_tones(self, tones: List[tuple], gap: float = 0.02) -> None:
        """Synthesize and play a list of (frequency_hz, duration_s) tones."""
        pa = pyaudio.PyAudio()
        try:
            output_device = self._find_output_device(pa)
            stream = pa.open(
                format=self._FORMAT,
                channels=self._CHANNELS,
                rate=self._SAMPLE_RATE,
                output=True,
                output_device_index=output_device,
            )
            for idx, (freq, duration) in enumerate(tones):
                samples = self._generate_tone(freq, duration)
                stream.write(samples.tobytes())
                if gap > 0 and idx < len(tones) - 1:
                    time.sleep(gap)
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        finally:
            pa.terminate()

    def _generate_tone(self, frequency: float, duration: float) -> np.ndarray:
        """Generate a sine-wave tone with fade-in/out envelope."""
        num_samples = int(self._SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
        wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        fade_len = min(int(self._SAMPLE_RATE * 0.01), num_samples // 4)
        if fade_len > 0:
            wave[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)
            wave[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)
        return (wave * self.volume).astype(np.float32)

    @staticmethod
    def _find_output_device(pa: pyaudio.PyAudio) -> Optional[int]:
        """Return the index of a pulse/default output device, or None."""
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info["name"].lower()
            if info["maxOutputChannels"] > 0 and ("pulse" in name or "default" in name):
                return i
        return None

    @staticmethod
    def _init_pygame() -> bool:
        """Initialize pygame.mixer; return True on success."""
        try:
            import pygame
            pygame.mixer.init()
            return True
        except Exception:
            return False
