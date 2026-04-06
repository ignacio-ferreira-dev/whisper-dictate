"""
Legacy test configuration.

These tests were written for the original whisperflow package with a local
Whisper model (tiny.en.pt). They are preserved for historical reference but
are skipped automatically because:

  - whisperflow package has been superseded by whisper_dictate
  - The local model file (whisperflow/models/tiny.en.pt) is not in the repo
  - Tests require real audio files from the LibriSpeech dataset

To run them manually (requires the original setup):
    pytest tests/legacy/ --no-header -v
"""

import pytest


def pytest_collection_modifyitems(items):
    """Skip all tests in this directory automatically."""
    skip_legacy = pytest.mark.skip(
        reason=(
            "Legacy whisperflow tests: require local Whisper model and "
            "LibriSpeech audio files. See tests/legacy/conftest.py."
        )
    )
    for item in items:
        if "legacy" in str(item.fspath):
            item.add_marker(skip_legacy)
