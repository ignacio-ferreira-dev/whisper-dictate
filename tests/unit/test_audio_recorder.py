"""
Unit tests for whisper_dictate.audio.recorder.AudioRecorder.

No microphone is involved: the recorder's state is driven directly so the
duration maths and the auto-stop handshake can be checked in isolation.
Hardware behaviour is covered by tests/integration/test_audio_recorder.py.
"""

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


# ---------------------------------------------------------------------------
# Auto-stop
# ---------------------------------------------------------------------------


class TestAutoStop:
    """The watchdog hands the recording over without duplicating the alert."""

    @pytest.fixture
    def auto_stopped(self, monkeypatch):
        """Run the watchdog once with the duration limit already exceeded."""
        on_auto_stop = MagicMock()
        rec = AudioRecorder(alerts=MagicMock(), verbose=False, on_auto_stop=on_auto_stop)
        rec.sample_rate = 16000
        monkeypatch.setattr(AudioRecorder, "MAX_RECORDING_SECONDS", 1)
        rec._buffer = _chunks(1000)  # 64 s, well past the patched limit
        rec._recording = True
        rec._watchdog_loop()
        return rec, on_auto_stop

    def test_notifies_the_client(self, auto_stopped):
        _, on_auto_stop = auto_stopped
        on_auto_stop.assert_called_once_with()

    def test_clears_the_recording_flag(self, auto_stopped):
        rec, _ = auto_stopped
        assert rec.is_recording is False

    def test_does_not_play_the_stop_alert(self, auto_stopped):
        """
        stop_recording() plays it when the client picks the frames up. Playing
        it here too produced a double beep on every auto-stop.
        """
        rec, _ = auto_stopped
        rec.alerts.play_stop.assert_not_called()

    def test_the_handover_plays_exactly_one_alert(self, auto_stopped):
        rec, _ = auto_stopped
        rec.stop_recording()
        rec.alerts.play_stop.assert_called_once_with()
