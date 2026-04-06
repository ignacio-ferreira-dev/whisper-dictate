"""
Audio Recorder
==============

Captures microphone input into an in-memory buffer.
Designed to be controlled by an external hotkey manager.

Usage:
    recorder = AudioRecorder()
    recorder.setup()
    recorder.start_recording()
    frames = recorder.stop_recording()   # list of raw PCM bytes
    recorder.teardown()
"""

import contextlib
import os
import threading
import time
from typing import Callable, List, Optional

import pyaudio


@contextlib.contextmanager
def _suppress_alsa_errors():
    """
    Silence the ALSA/Jack error messages that PortAudio prints to stderr
    when probing unavailable audio devices during PyAudio initialisation.
    These are harmless — PyAudio falls back to PulseAudio automatically.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

from whisper_dictate.audio.alerts import AudioAlertsManager


class AudioRecorder:
    """
    Manages microphone capture with start/stop control.

    Audio data is accumulated as a list of raw PCM byte chunks
    (paInt16, mono) ready to be passed to a TranscriptionBackend.
    """

    PREFERRED_SAMPLE_RATES: List[int] = [16000, 22050, 44100, 8000]
    FORMAT = pyaudio.paInt16
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024
    MAX_RECORDING_SECONDS: int = 600  # 10 minutes; auto-stops and transcribes normally

    def __init__(
        self,
        alerts: Optional[AudioAlertsManager] = None,
        verbose: bool = True,
        on_auto_stop: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            alerts:       AudioAlertsManager instance. If None a default one is created.
            verbose:      When True, print status messages to stdout.
            on_auto_stop: Optional callback invoked when recording stops automatically
                          (max duration reached or stream error). Called from the
                          capture thread — must be thread-safe (e.g. schedule a coroutine
                          with run_coroutine_threadsafe).
        """
        self.alerts = alerts or AudioAlertsManager()
        self.verbose = verbose
        self._on_auto_stop = on_auto_stop

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._buffer: List[bytes] = []
        self._recording: bool = False
        self._record_thread: Optional[threading.Thread] = None

        self.sample_rate: Optional[int] = None
        self.device_index: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """
        Initialize PyAudio and discover a working input device.

        Returns:
            True on success, False if no usable device is found.
        """
        with _suppress_alsa_errors():
            self._pa = pyaudio.PyAudio()
        if not self._find_working_device():
            self._pa.terminate()
            self._pa = None
            return False
        return True

    def teardown(self) -> None:
        """Release all PyAudio resources."""
        if self._recording:
            self._stop_stream()
        if self._pa:
            self._pa.terminate()
            self._pa = None

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self) -> bool:
        """
        Begin capturing audio from the microphone.

        Plays a start alert and launches a background capture thread.

        Returns:
            True if recording started successfully.
        """
        if self._recording:
            return False
        if not self._pa or self.device_index is None:
            self._log("Audio not initialized - call setup() first")
            return False

        # Play the alert BEFORE opening the PyAudio stream.
        # Opening a PortAudio input stream while pygame/SDL is also accessing
        # the audio subsystem from a background thread causes heap corruption
        # (malloc crash) on Linux. Playing first ensures pygame finishes before
        # PortAudio acquires the device.
        self.alerts.play_start()
        time.sleep(0.15)  # let the alert thread start before PortAudio opens

        try:
            self._stream = self._pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK_SIZE,
            )
            self._buffer = []
            self._recording = True
            self._log("Recording started - press hotkey again to stop")
            self._record_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._record_thread.start()
            return True
        except Exception as e:
            self._log(f"Failed to start recording: {e}")
            self.alerts.play_error()
            return False

    def stop_recording(self) -> List[bytes]:
        """
        Stop audio capture and return the recorded PCM frames.

        Safe to call even if the capture loop already stopped itself (e.g. due
        to a stream error or hitting the max duration limit). In that case the
        buffered frames are still returned for transcription.

        Plays a stop alert and waits for the capture thread to finish.

        Returns:
            List of raw PCM byte chunks. Empty list if nothing was recorded.
        """
        has_frames = bool(self._buffer)
        if not self._recording and not has_frames:
            return []

        self._stop_stream()
        self.alerts.play_stop()

        if self._record_thread:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None

        frames = list(self._buffer)
        self._buffer = []
        if not frames:
            return []
        duration = len(frames) * self.CHUNK_SIZE / self.sample_rate
        self._log(f"Recording stopped - {duration:.1f}s captured ({len(frames)} chunks)")
        return frames

    @property
    def is_recording(self) -> bool:
        """True while audio capture is active."""
        return self._recording

    @property
    def has_pending_frames(self) -> bool:
        """True if there are buffered frames ready to transcribe but not yet consumed."""
        return bool(self._buffer) and not self._recording

    def get_duration(self) -> float:
        """Return duration in seconds of currently buffered audio."""
        if not self.sample_rate:
            return 0.0
        return len(self._buffer) * self.CHUNK_SIZE / self.sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Background thread: reads chunks from the stream into the buffer.

        PortAudio writes ALSA/Jack probe errors directly to stderr (C level),
        so we redirect fd 2 to /dev/null for the duration of the capture loop.
        Python-level output (our _log calls) is on stdout and is unaffected.
        """
        auto_stopped = False
        with _suppress_alsa_errors():
            while self._recording and self._stream:
                if self.get_duration() >= self.MAX_RECORDING_SECONDS:
                    self._log(
                        f"Maximum recording duration reached "
                        f"({self.MAX_RECORDING_SECONDS // 60} min) — stopping and transcribing"
                    )
                    self._stop_stream()
                    auto_stopped = True
                    break
                try:
                    chunk = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    self._buffer.append(chunk)
                except Exception as e:
                    self._log(f"Capture error: {e}")
                    self._stop_stream()  # close stream cleanly to avoid heap corruption
                    auto_stopped = True
                    break

        if auto_stopped and self._on_auto_stop:
            self.alerts.play_stop()
            self._on_auto_stop()

    def _stop_stream(self) -> None:
        """Safely stop and close the PyAudio stream."""
        self._recording = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _find_working_device(self) -> bool:
        """
        Iterate over available devices and sample rates to find a working combo.
        Prefers pulse/default devices over raw hardware.
        """
        input_devices = [
            i for i in range(self._pa.get_device_count())
            if self._pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        ]

        if not input_devices:
            self._log("No audio input devices found")
            return False

        pulse = [d for d in input_devices
                 if "pulse" in self._pa.get_device_info_by_index(d)["name"].lower()]
        default = [d for d in input_devices
                   if "default" in self._pa.get_device_info_by_index(d)["name"].lower()]
        ordered = pulse + default or input_devices

        for device_id in ordered:
            for rate in self.PREFERRED_SAMPLE_RATES:
                try:
                    stream = self._pa.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=self.CHUNK_SIZE,
                        start=False,
                    )
                    stream.close()
                    self.device_index = device_id
                    self.sample_rate = rate
                    name = self._pa.get_device_info_by_index(device_id)["name"]
                    self._log(f"Audio device ready: '{name}' @ {rate} Hz")
                    return True
                except Exception:
                    continue

        self._log("No working audio configuration found")
        return False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] [Recorder] {msg}")
