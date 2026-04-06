"""
Audio Recorder
==============

Captures microphone input into an in-memory buffer.
Designed to be controlled by an external hotkey manager.

Responsibilities:
  - Find a working PyAudio input device + sample rate
  - Start/stop recording on demand
  - Expose recorded PCM frames for downstream processing
  - Play audio alerts (via AudioAlertsManager) on state transitions

Usage:
    recorder = AudioRecorder()
    recorder.setup()          # find device, open PyAudio
    recorder.start_recording()
    # ... user speaks ...
    frames = recorder.stop_recording()   # returns raw PCM bytes list
    recorder.teardown()
"""

import threading
import time
from typing import List, Optional

import pyaudio

from whisperflow.audio.alerts import AudioAlertsManager


class AudioRecorder:
    """
    Manages microphone capture with start/stop control.

    Audio data is accumulated in a list of raw PCM byte chunks
    (paInt16, mono) which can be passed directly to the transcriber.
    """

    PREFERRED_SAMPLE_RATES: List[int] = [16000, 22050, 44100, 8000]
    FORMAT = pyaudio.paInt16
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024

    def __init__(
        self,
        alerts: Optional[AudioAlertsManager] = None,
        verbose: bool = True,
    ):
        """
        Args:
            alerts:  AudioAlertsManager instance. If None a default one is created.
            verbose: When True, print status messages to stdout.
        """
        self.alerts = alerts or AudioAlertsManager()
        self.verbose = verbose

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
        self._pa = pyaudio.PyAudio()
        result = self._find_working_device()
        if not result:
            self._pa.terminate()
            self._pa = None
        return result

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
            self.alerts.play_start()
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

        Plays a stop alert and waits for the capture thread to finish.

        Returns:
            List of raw PCM byte chunks (paInt16 mono at self.sample_rate).
            Empty list if nothing was recorded.
        """
        if not self._recording:
            return []

        self._stop_stream()
        self.alerts.play_stop()

        if self._record_thread:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None

        frames = list(self._buffer)
        duration = len(frames) * self.CHUNK_SIZE / self.sample_rate
        self._log(f"Recording stopped - {duration:.1f}s captured ({len(frames)} chunks)")
        return frames

    @property
    def is_recording(self) -> bool:
        """True while audio capture is active."""
        return self._recording

    def get_duration(self) -> float:
        """Return duration in seconds of currently buffered audio."""
        if not self.sample_rate:
            return 0.0
        return len(self._buffer) * self.CHUNK_SIZE / self.sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: reads chunks from the stream into the buffer."""
        while self._recording and self._stream:
            try:
                chunk = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                self._buffer.append(chunk)
            except Exception as e:
                self._log(f"Capture error: {e}")
                self._recording = False
                break

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
        Iterate over available devices and sample rates to find a combo
        that opens without error. Prefers pulse/default devices.
        """
        device_count = self._pa.get_device_count()
        input_devices = []

        for i in range(device_count):
            info = self._pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                input_devices.append(i)

        if not input_devices:
            self._log("No audio input devices found")
            return False

        # Prioritize pulse/default devices
        pulse_devices = [
            d for d in input_devices
            if "pulse" in self._pa.get_device_info_by_index(d)["name"].lower()
        ]
        default_devices = [
            d for d in input_devices
            if "default" in self._pa.get_device_info_by_index(d)["name"].lower()
        ]
        ordered = pulse_devices + default_devices
        if not ordered:
            ordered = input_devices  # fallback: try everything

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
        """Print a timestamped log line when verbose mode is on."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] [Recorder] {msg}")
