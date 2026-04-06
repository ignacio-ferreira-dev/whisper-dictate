"""
Integration tests for AudioRecorder.

Verifies device discovery, start/stop control, and that recorded
frames accumulate correctly over time. Requires a real microphone.
"""

import time
import pytest

from whisper_dictate.audio.alerts import AudioAlertsManager
from whisper_dictate.audio.recorder import AudioRecorder


@pytest.fixture
def silent_recorder():
    """AudioRecorder with alerts disabled to avoid noise during tests."""
    alerts = AudioAlertsManager(enabled=False)
    recorder = AudioRecorder(alerts=alerts, verbose=False)
    recorder.setup()
    yield recorder
    recorder.teardown()


def test_setup_finds_device(silent_recorder):
    assert silent_recorder.device_index is not None
    assert silent_recorder.sample_rate in AudioRecorder.PREFERRED_SAMPLE_RATES


def test_start_sets_is_recording(silent_recorder):
    assert not silent_recorder.is_recording
    silent_recorder.start_recording()
    assert silent_recorder.is_recording
    silent_recorder.stop_recording()


def test_recording_accumulates_chunks(silent_recorder):
    silent_recorder.start_recording()
    time.sleep(1.0)
    duration = silent_recorder.get_duration()
    assert duration > 0.5, f"Expected >0.5s, got {duration:.2f}s"
    silent_recorder.stop_recording()


def test_stop_returns_pcm_frames(silent_recorder):
    silent_recorder.start_recording()
    time.sleep(1.0)
    frames = silent_recorder.stop_recording()
    assert isinstance(frames, list)
    assert len(frames) > 0
    assert all(isinstance(f, bytes) for f in frames)


def test_stop_when_idle_returns_empty_list(silent_recorder):
    frames = silent_recorder.stop_recording()
    assert frames == []


def test_teardown_is_idempotent():
    alerts = AudioAlertsManager(enabled=False)
    recorder = AudioRecorder(alerts=alerts, verbose=False)
    recorder.setup()
    recorder.start_recording()
    time.sleep(0.2)
    recorder.teardown()
    recorder.teardown()  # second call must not crash
