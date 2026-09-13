"""
Unit tests for whisper_dictate.audio.recorder.AudioRecorder.

No microphone is involved: the recorder's state is driven directly so the
duration maths and the watchdog (segment beep, cancel at the cap) can be checked in isolation.
Hardware behaviour is covered by tests/integration/test_audio_recorder.py.
"""

import threading

import pytest
from unittest.mock import MagicMock

from whisper_dictate.audio.recorder import AudioRecorder

pytestmark = pytest.mark.unit


@pytest.fixture
def recorder() -> AudioRecorder:
    """An AudioRecorder wired to a mock alerts manager, with no stream open."""
    rec = AudioRecorder(alerts=MagicMock(), verbose=False)
    rec.sample_rate = 16000
    return rec


def _chunks(count: int) -> list:
    """Return `count` silent PCM chunks of CHUNK_SIZE frames each."""
    return [b"\x00" * (AudioRecorder.CHUNK_SIZE * 2)] * count


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


class TestDuration:
    """frames_duration() converts a chunk count into seconds."""

    def test_duration_of_a_known_chunk_count(self):
        rec = AudioRecorder(alerts=MagicMock(), verbose=False)
        rec.sample_rate = 16000
        # 16000 Hz / 1024 frames per chunk ≈ 15.6 chunks per second
        assert rec.frames_duration(_chunks(1000)) == pytest.approx(64.0)

    def test_duration_is_zero_before_setup(self):
        rec = AudioRecorder(alerts=MagicMock(), verbose=False)
        assert rec.frames_duration(_chunks(100)) == 0.0

    def test_get_duration_reflects_the_live_buffer(self, recorder):
        recorder._buffer = _chunks(1000)
        assert recorder.get_duration() == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# stop_recording
# ---------------------------------------------------------------------------


class TestStopRecording:
    """stop_recording() drains the buffer and owns the stop alert."""

    def test_returns_buffered_frames(self, recorder):
        recorder._recording = True
        recorder._buffer = _chunks(20)
        assert len(recorder.stop_recording()) == 20

    def test_clears_the_buffer(self, recorder):
        recorder._recording = True
        recorder._buffer = _chunks(20)
        recorder.stop_recording()
        assert recorder._buffer == []

    def test_plays_the_stop_alert_once(self, recorder):
        recorder._recording = True
        recorder._buffer = _chunks(20)
        recorder.stop_recording()
        recorder.alerts.play_stop.assert_called_once_with()

    def test_is_a_no_op_when_idle_with_an_empty_buffer(self, recorder):
        assert recorder.stop_recording() == []
        recorder.alerts.play_stop.assert_not_called()

    def test_tolerates_a_missing_stream(self, recorder):
        """setup() may never have run; stopping must not raise."""
        recorder._recording = True
        recorder._buffer = _chunks(5)
        recorder.stop_recording()  # _stream is None


def _recording(seconds: float, max_recording_seconds: float = 600,
               segment_seconds=None) -> AudioRecorder:
    """A recorder that is 'recording' and already holds `seconds` of audio."""
    rec = AudioRecorder(
        alerts=MagicMock(), verbose=False,
        max_recording_seconds=max_recording_seconds, segment_seconds=segment_seconds,
    )
    rec.sample_rate = 16000
    rec._buffer = _chunks(int(seconds * rec.sample_rate / AudioRecorder.CHUNK_SIZE))
    rec._recording = True
    return rec


# ---------------------------------------------------------------------------
# Cancel at the cap
# ---------------------------------------------------------------------------


class TestCancelAtCap:
    """A recording nobody stopped is discarded at the cap, never transcribed."""

    @pytest.fixture
    def cancelled(self, monkeypatch) -> AudioRecorder:
        """Run the watchdog with the cap already exceeded (64 s against 1 s)."""
        monkeypatch.setattr(AudioRecorder, "WATCHDOG_INTERVAL_SECONDS", 0)
        rec = _recording(64, max_recording_seconds=1)
        rec._watchdog_loop()
        return rec

    def test_the_watchdog_stops_with_the_recording(self, monkeypatch):
        """
        A watchdog that sleeps on after cancelling would still be running when
        the next hotkey press starts a recording, and would then watch that one
        too: two beep schedules and two cancels for one recording.
        """
        monkeypatch.setattr(AudioRecorder, "WATCHDOG_INTERVAL_SECONDS", 0.05)
        rec = _recording(64, max_recording_seconds=1)
        watchdog = threading.Thread(target=rec._watchdog_loop, daemon=True)
        watchdog.start()
        watchdog.join(timeout=1.0)
        assert watchdog.is_alive() is False

    def test_stops_recording(self, cancelled):
        assert cancelled.is_recording is False

    def test_discards_the_audio(self, cancelled):
        assert cancelled.get_duration() == 0.0

    def test_plays_the_error_beep_once(self, cancelled):
        cancelled.alerts.play_error.assert_called_once_with()

    def test_a_later_stop_returns_nothing_and_stays_quiet(self, cancelled):
        """F9 after a cancel must not transcribe the discarded audio or beep 'stop'."""
        assert cancelled.stop_recording() == []
        cancelled.alerts.play_stop.assert_not_called()

    def test_a_recording_under_the_cap_is_left_alone(self):
        rec = _recording(20, max_recording_seconds=30)
        rec._watchdog_tick(None)
        assert rec.is_recording is True
        rec.alerts.play_error.assert_not_called()


# ---------------------------------------------------------------------------
# Segment beep
# ---------------------------------------------------------------------------


class TestSegmentBeep:
    """Each new segment gets one short beep, and the recording keeps going."""

    def test_beeps_when_a_segment_boundary_is_crossed(self):
        rec = _recording(40, segment_seconds=30)
        assert rec._watchdog_tick(30) == 60
        rec.alerts.play_segment.assert_called_once_with()

    def test_beeps_once_per_boundary(self):
        rec = _recording(40, segment_seconds=30)
        rec._watchdog_tick(rec._watchdog_tick(30))
        rec.alerts.play_segment.assert_called_once_with()

    def test_keeps_recording_after_the_beep(self):
        rec = _recording(40, segment_seconds=30)
        rec._watchdog_tick(30)
        assert rec.is_recording is True
        assert rec.get_duration() == pytest.approx(40, abs=0.1)

    def test_no_beep_before_the_first_boundary(self):
        rec = _recording(20, segment_seconds=30)
        assert rec._watchdog_tick(30) == 30
        rec.alerts.play_segment.assert_not_called()

    def test_no_beep_without_segment_seconds(self):
        rec = _recording(40)
        assert rec._watchdog_tick(None) is None
        rec.alerts.play_segment.assert_not_called()

    def test_the_cap_cancels_instead_of_beeping(self):
        rec = _recording(40, max_recording_seconds=30, segment_seconds=30)
        rec._watchdog_tick(30)
        rec.alerts.play_segment.assert_not_called()
        rec.alerts.play_error.assert_called_once_with()
