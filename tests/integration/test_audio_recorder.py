"""
Integration tests for AudioRecorder.

These tests capture real microphone audio and verify timing, frame
format, and start/stop state transitions against actual hardware.

Requirements: microphone input device.

Run with:
    pytest tests/integration/test_audio_recorder.py -v
"""

import time
import pytest

from whisper_dictate.audio.recorder import AudioRecorder

pytestmark = pytest.mark.integration


class TestDeviceDiscovery:
    """setup() finds a usable input device and stores a valid configuration."""

    def test_device_index_is_set(self, recorder):
        assert recorder.device_index is not None

    def test_sample_rate_is_a_known_value(self, recorder):
        assert recorder.sample_rate in AudioRecorder.PREFERRED_SAMPLE_RATES

    def test_setup_returns_true(self, silent_alerts):
        rec = AudioRecorder(alerts=silent_alerts, verbose=False)
        result = rec.setup()
        rec.teardown()
        assert result is True


class TestRecordingStateTransitions:
    """start_recording() and stop_recording() manage is_recording correctly."""

    def test_initially_not_recording(self, recorder):
        assert recorder.is_recording is False

    def test_is_recording_after_start(self, recorder):
        recorder.start_recording()
        assert recorder.is_recording is True
        recorder.stop_recording()

    def test_not_recording_after_stop(self, recorder):
        recorder.start_recording()
        recorder.stop_recording()
        assert recorder.is_recording is False

    def test_start_is_idempotent(self, recorder):
        """Calling start_recording() twice does not raise or corrupt state."""
        recorder.start_recording()
        result = recorder.start_recording()   # second call should return False
        assert result is False
        recorder.stop_recording()


class TestAudioCapture:
    """Audio frames are captured with the expected format and timing."""

    def test_frames_accumulate_during_recording(self, recorder):
        recorder.start_recording()
        time.sleep(1.0)
        duration = recorder.get_duration()
        recorder.stop_recording()
        assert duration > 0.5, f"Expected >0.5s buffered, got {duration:.2f}s"

    def test_stop_returns_list_of_bytes(self, recorder):
        recorder.start_recording()
        time.sleep(0.5)
        frames = recorder.stop_recording()
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert all(isinstance(f, bytes) for f in frames)

    def test_each_chunk_has_expected_size(self, recorder):
        recorder.start_recording()
        time.sleep(0.5)
        frames = recorder.stop_recording()
        expected = AudioRecorder.CHUNK_SIZE * 2  # paInt16 = 2 bytes per sample
        assert all(len(f) == expected for f in frames)

    def test_stop_when_idle_returns_empty_list(self, recorder):
        frames = recorder.stop_recording()
        assert frames == []

    def test_get_duration_is_zero_before_recording(self, recorder):
        assert recorder.get_duration() == pytest.approx(0.0)


class TestTeardown:
    """teardown() releases resources cleanly and is safe to call multiple times."""

    def test_teardown_after_recording_does_not_raise(self, silent_alerts):
        rec = AudioRecorder(alerts=silent_alerts, verbose=False)
        rec.setup()
        rec.start_recording()
        time.sleep(0.2)
        rec.teardown()

    def test_teardown_is_idempotent(self, silent_alerts):
        rec = AudioRecorder(alerts=silent_alerts, verbose=False)
        rec.setup()
        rec.teardown()
        rec.teardown()  # second call must not raise
