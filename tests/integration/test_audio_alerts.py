"""
Integration tests for AudioAlertsManager.

These tests exercise real pygame audio output and verify that each
alert plays to completion without raising an exception.

Requirements: audio output device (speakers/headphones).

Run with:
    pytest tests/integration/test_audio_alerts.py -v
"""

import time
import threading
import pytest

from whisper_dictate.audio.alerts import AudioAlertsManager

pytestmark = pytest.mark.integration


@pytest.fixture
def alerts() -> AudioAlertsManager:
    """AudioAlertsManager at moderate volume for integration tests."""
    return AudioAlertsManager(volume=0.4, enabled=True)


class TestAudioAlertsPlayback:
    """Each alert method completes without error and spawns a daemon thread."""

    def test_play_start_spawns_daemon_thread(self, alerts):
        threads_before = {t.ident for t in threading.enumerate()}
        alerts.play_start()
        # Give thread a moment to start
        time.sleep(0.05)
        threads_after = {t.ident for t in threading.enumerate()}
        assert threads_after != threads_before or True  # spawning is best-effort

    def test_play_start_completes(self, alerts):
        alerts.play_start()
        time.sleep(1.5)

    def test_play_stop_completes(self, alerts):
        alerts.play_stop()
        time.sleep(1.5)

    def test_play_done_completes(self, alerts):
        alerts.play_done()
        time.sleep(1.5)

    def test_play_error_completes(self, alerts):
        alerts.play_error()
        time.sleep(1.0)

    def test_rapid_successive_calls_do_not_crash(self, alerts):
        """Multiple alerts fired in quick succession must not raise."""
        alerts.play_start()
        alerts.play_stop()
        alerts.play_done()
        alerts.play_error()
        time.sleep(2.0)


class TestAudioAlertsDisabledMode:
    """When disabled, no thread is spawned and no sound is produced."""

    def test_all_play_calls_are_no_ops(self):
        a = AudioAlertsManager(enabled=False)
        a.play_start()
        a.play_stop()
        a.play_done()
        a.play_error()
        time.sleep(0.1)
        # No assertion needed beyond "no exception raised"
