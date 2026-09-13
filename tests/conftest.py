"""
Shared pytest fixtures.

Markers (declared in pyproject.toml)
-------
unit        Fast tests with no I/O dependencies (no microphone, no network).
integration Tests that need real audio hardware (microphone / speakers) or
            the paid Whisper API — API tests also need RUN_API_TESTS=1.

Usage
-----
Run only unit tests:
    pytest -m unit

Run only integration tests (microphone; API tests are skipped unless asked):
    pytest -m integration

Run everything:
    pytest
"""

import pytest
from unittest.mock import MagicMock

from whisper_dictate.audio.alerts import AudioAlertsManager
from whisper_dictate.audio.recorder import AudioRecorder


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
