"""
Shared pytest configuration, fixtures and markers.

Markers
-------
unit        Fast tests with no I/O dependencies (no microphone, no network).
integration Tests that require real audio hardware (microphone / speakers).

Usage
-----
Run only unit tests:
    pytest -m unit

Run only integration tests (requires microphone):
    pytest -m integration

Run everything:
    pytest
"""

import pytest
from unittest.mock import MagicMock

from whisper_dictate.audio.alerts import AudioAlertsManager
from whisper_dictate.audio.recorder import AudioRecorder


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast tests, no hardware or network required")
    config.addinivalue_line("markers", "integration: require real audio hardware")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silent_alerts() -> AudioAlertsManager:
    """AudioAlertsManager with playback disabled (no sound during tests)."""
    return AudioAlertsManager(enabled=False)


@pytest.fixture
def recorder(silent_alerts) -> AudioRecorder:
    """
    Fully initialised AudioRecorder with no audio alerts.
    Automatically torn down after the test.
    """
    rec = AudioRecorder(alerts=silent_alerts, verbose=False)
    rec.setup()
    yield rec
    rec.teardown()
