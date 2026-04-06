"""
Integration tests for AudioAlertsManager.

Verifies that each alert type plays without error and that disabled
mode is a clean no-op. Tests with enabled=True produce real audio
output (MP3 files via pygame when available, sine-wave fallback otherwise).
"""

import time
import pytest

from whisper_dictate.audio.alerts import AudioAlertsManager


def test_instantiation():
    alerts = AudioAlertsManager(volume=0.4, enabled=True)
    assert alerts.volume == 0.4
    assert alerts.enabled is True


def test_play_start_completes_without_error():
    alerts = AudioAlertsManager(volume=0.5)
    alerts.play_start()
    time.sleep(1.5)


def test_play_stop_completes_without_error():
    alerts = AudioAlertsManager(volume=0.5)
    alerts.play_stop()
    time.sleep(1.5)


def test_play_done_completes_without_error():
    alerts = AudioAlertsManager(volume=0.5)
    alerts.play_done()
    time.sleep(1.5)


def test_play_error_completes_without_error():
    alerts = AudioAlertsManager(volume=0.5)
    alerts.play_error()
    time.sleep(1.0)


def test_disabled_mode_is_silent_noop():
    alerts = AudioAlertsManager(volume=0.9, enabled=False)
    alerts.play_start()
    alerts.play_stop()
    alerts.play_done()
    alerts.play_error()
    time.sleep(0.3)
