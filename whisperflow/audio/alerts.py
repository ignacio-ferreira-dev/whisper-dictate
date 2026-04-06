"""
Audio Alerts Manager
====================

Generates audible tones to signal recording start/stop events.
Tones are synthesized with numpy+pyaudio - no external audio files needed.

Usage:
    alerts = AudioAlertsManager()
    alerts.play_start()   # rising double-beep  -> recording started
    alerts.play_stop()    # falling single-beep -> recording stopped
    alerts.play_error()   # low short buzz       -> something went wrong
"""

import math
import struct
import threading
import time
from typing import List

import numpy as np
import pyaudio


class AudioAlertsManager:
    """
    Manages audible feedback tones for recording state changes.

    Tones are generated programmatically using sine waves so no
    external audio assets are required. Each play call is non-blocking
    (runs in a daemon thread) so it never delays the main recording flow.
    """

    SAMPLE_RATE: int = 44100
    FORMAT = pyaudio.paFloat32
    CHANNELS: int = 1

    def __init__(self, volume: float = 0.4, enabled: bool = True):
        """
        Args:
            volume: Playback amplitude in [0.0, 1.0].
            enabled: When False all play calls are no-ops (silent mode).
        """
        self.volume = max(0.0, min(1.0, volume))
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_start(self) -> None:
        """Play a rising two-tone beep indicating recording has started."""
        tones = [
            (880, 0.08),   # A5 short
            (1320, 0.12),  # E6 slightly longer
        ]
        self._play_async(tones, gap=0.03)

    def play_stop(self) -> None:
        """Play a single falling tone indicating recording has stopped."""
        tones = [
            (660, 0.15),   # E5
        ]
        self._play_async(tones)

    def play_error(self) -> None:
        """Play a low buzz indicating an error occurred."""
        tones = [
            (220, 0.10),   # A3
            (180, 0.10),   # ~F#3
        ]
        self._play_async(tones, gap=0.02)

    def play_processing(self) -> None:
        """Play a soft single mid-tone while transcription is in progress."""
        tones = [(440, 0.06)]  # A4
        self._play_async(tones)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_async(self, tones: List[tuple], gap: float = 0.0) -> None:
        """Spawn a daemon thread to play tones without blocking the caller."""
        if not self.enabled:
            return
        thread = threading.Thread(
            target=self._play_tones,
            args=(tones, gap),
            daemon=True,
        )
        thread.start()

    def _play_tones(self, tones: List[tuple], gap: float = 0.0) -> None:
        """
        Play a sequence of (frequency_hz, duration_s) tones.

        Args:
            tones: List of (frequency, duration) pairs.
            gap: Silent pause (seconds) between consecutive tones.
        """
        pa = pyaudio.PyAudio()
        try:
            output_device = self._find_output_device(pa)
            stream = pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
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
            pass  # Audio alerts are best-effort; never crash the main flow
        finally:
            pa.terminate()

    def _generate_tone(self, frequency: float, duration: float) -> np.ndarray:
        """
        Generate a sine-wave tone with smooth attack/release envelope.

        Args:
            frequency: Tone frequency in Hz.
            duration:  Tone duration in seconds.

        Returns:
            float32 numpy array of audio samples.
        """
        num_samples = int(self.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
        wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Apply a short fade-in / fade-out to avoid clicks
        fade_len = min(int(self.SAMPLE_RATE * 0.01), num_samples // 4)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
            wave[:fade_len] *= fade_in
            wave[-fade_len:] *= fade_out

        return (wave * self.volume).astype(np.float32)

    @staticmethod
    def _find_output_device(pa: pyaudio.PyAudio):
        """
        Return the index of the best available output device (prefer pulse/default).
        Returns None to fall back to PyAudio's default.
        """
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info["name"].lower()
            if info["maxOutputChannels"] > 0 and ("pulse" in name or "default" in name):
                return i
        return None
