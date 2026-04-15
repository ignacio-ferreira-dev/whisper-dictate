"""
Audio Recorder
==============

Captures microphone input into an in-memory buffer.
Designed to be controlled by an external hotkey manager.

The PyAudio stream is opened once during setup() and kept alive for the
entire session. start_recording() / stop_recording() only toggle capture
without touching the stream — this avoids the PortAudio heap corruption
that occurs when repeatedly opening and closing streams on Linux/PulseAudio.

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

from whisper_dictate.audio.alerts import AudioAlertsManager


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


class AudioRecorder:
    """
    Manages microphone capture with start/stop control.

    The PyAudio stream is opened once and kept alive to avoid PortAudio
    heap corruption from repeated open/close cycles on Linux/PulseAudio.
    Capture is controlled by a flag read by the background thread.

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
        self._capture_thread: Optional[threading.Thread] = None

        self.sample_rate: Optional[int] = None
        self.device_index: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """
        Initialize PyAudio, find a working device, and open the stream once.

        The stream stays open for the entire session to avoid PortAudio
        heap corruption from repeated open/close cycles.

        Returns:
            True on success, False if no usable device is found.
        """
        with _suppress_alsa_errors():
            self._pa = pyaudio.PyAudio()

        if not self._find_working_device():
            self._pa.terminate()
            self._pa = None
            return False

        try:
            with _suppress_alsa_errors():
                self._stream = self._pa.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.CHUNK_SIZE,
                    stream_callback=self._stream_callback,
                    start=False,
                )
        except Exception as e:
            self._log(f"Failed to open audio stream: {e}")
            self._pa.terminate()
            self._pa = None
            return False

        return True

    def teardown(self) -> None:
        """Release all PyAudio resources."""
        self._recording = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self) -> bool:
        """
        Begin capturing audio from the microphone.

        Starts the persistent stream (if not already running) and sets the
        capture flag. Plays a start alert synchronously first.

        Returns:
            True if recording started successfully.
        """
        if self._recording:
            return False
        if not self._stream:
            self._log("Audio not initialized - call setup() first")
            return False

        self.alerts.play_start()  # blocking — finishes before stream starts

        self._buffer = []
        self._recording = True

        try:
            self._stream.start_stream()
        except Exception as e:
            self._log(f"Failed to start stream: {e}")
            self._recording = False
            self.alerts.play_error()
            return False

        self._log("Recording started - press hotkey again to stop")

        # Watchdog thread: monitors duration limit and stream errors
        self._capture_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        self._capture_thread.start()
        return True

    def stop_recording(self) -> List[bytes]:
        """
        Stop audio capture and return the recorded PCM frames.

        Safe to call even if the capture stopped automatically.

        Returns:
            List of raw PCM byte chunks. Empty list if nothing was recorded.
        """
        has_frames = bool(self._buffer)
        if not self._recording and not has_frames:
            return []

        self._recording = False

        try:
            self._stream.stop_stream()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self.alerts.play_stop()

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

    def _stream_callback(self, in_data, frame_count, time_info, status):  # pylint: disable=unused-argument
        """
        PyAudio callback: called from PortAudio's audio thread for each chunk.
        Appends data to the buffer only while _recording is True.
        """
        if self._recording:
            self._buffer.append(in_data)
        return (None, pyaudio.paContinue)

    def _watchdog_loop(self) -> None:
        """
        Background thread: monitors recording duration limit.
        Does not read from the stream — PortAudio calls _stream_callback directly.
        """
        auto_stopped = False

        while self._recording:
            if self.get_duration() >= self.MAX_RECORDING_SECONDS:
                self._log(
                    f"Maximum recording duration reached "
                    f"({self.MAX_RECORDING_SECONDS // 60} min) — stopping and transcribing"
                )
                self._recording = False
                try:
                    self._stream.stop_stream()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                auto_stopped = True
                break
            time.sleep(0.5)

        if auto_stopped and self._on_auto_stop:
            self.alerts.play_stop()
            self._on_auto_stop()

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
                    # Test the device/rate combo with a temporary stream
                    with _suppress_alsa_errors():
                        test_stream = self._pa.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=rate,
                            input=True,
                            input_device_index=device_id,
                            frames_per_buffer=self.CHUNK_SIZE,
                            start=False,
                        )
                    test_stream.close()
                    self.device_index = device_id
                    self.sample_rate = rate
                    name = self._pa.get_device_info_by_index(device_id)["name"]
                    self._log(f"Audio device ready: '{name}' @ {rate} Hz")
                    return True
                except Exception:  # pylint: disable=broad-exception-caught
                    continue

        self._log("No working audio configuration found")
        return False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] [Recorder] {msg}")
