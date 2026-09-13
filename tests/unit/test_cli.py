"""
Unit tests for the whisper_dictate CLI argument parser.

The parser takes its defaults from Settings, so these tests pin down the
resolution order that the README documents: command-line flag > .env > default.
"""

import pytest

from whisper_dictate.__main__ import build_backend, build_parser
from whisper_dictate.config import Settings
from whisper_dictate.transcription.chunked import ChunkedTranscriptionBackend
from whisper_dictate.transcription.openai_backend import OpenAIWhisperBackend

pytestmark = pytest.mark.unit


@pytest.fixture
def settings(monkeypatch) -> Settings:
    """Settings built from an environment with nothing configured."""
    for name in ("HOTKEY", "QUIT_KEY", "DEFAULT_LANGUAGE", "ALERT_VOLUME",
                 "ALERTS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    return Settings()


def _parse(settings: Settings, *argv):
    return build_parser(settings).parse_args(list(argv))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """With nothing configured, the built-in defaults apply."""

    def test_hotkey_defaults_to_f9(self, settings):
        assert _parse(settings).hotkey == "f9"

    def test_quit_key_defaults_to_home(self, settings):
        """ESC is deliberately not the default — it is pressed far too often."""
        assert _parse(settings).quit_key == "home"


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    """Flag beats .env beats default, for every setting the CLI exposes."""

    @pytest.mark.parametrize("env_name,attribute,value", [
        ("HOTKEY", "hotkey", "f10"),
        ("QUIT_KEY", "quit_key", "end"),
        ("DEFAULT_LANGUAGE", "language", "es"),
    ])
    def test_env_overrides_the_default(self, monkeypatch, env_name, attribute, value):
        monkeypatch.setenv(env_name, value)
        assert getattr(_parse(Settings()), attribute) == value

    @pytest.mark.parametrize("env_name,flag,attribute", [
        ("HOTKEY", "--hotkey", "hotkey"),
        ("QUIT_KEY", "--quit-key", "quit_key"),
        ("DEFAULT_LANGUAGE", "--language", "language"),
    ])
    def test_flag_overrides_the_env(self, monkeypatch, env_name, flag, attribute):
        monkeypatch.setenv(env_name, "from_env")
        args = _parse(Settings(), flag, "from_flag")
        assert getattr(args, attribute) == "from_flag"

    def test_alert_volume_comes_from_the_env(self, monkeypatch):
        monkeypatch.setenv("ALERT_VOLUME", "0.3")
        assert _parse(Settings()).volume == pytest.approx(0.3)

    def test_alerts_can_be_disabled_from_the_env(self, monkeypatch):
        """ALERTS_ENABLED=false must silence the alerts without passing a flag."""
        monkeypatch.setenv("ALERTS_ENABLED", "false")
        assert _parse(Settings()).no_alerts is True

    def test_alerts_are_enabled_by_default(self, settings):
        assert _parse(settings).no_alerts is False


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------


class TestBuildBackend:
    """The app transcribes through the chunking wrapper, configured from Settings."""

    @pytest.fixture
    def backend(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("TRANSCRIPTION_CHUNK_SECONDS", "120")
        monkeypatch.setenv("TRANSCRIPTION_MAX_PARALLEL", "3")
        return build_backend(Settings())

    def test_long_recordings_are_split(self, backend):
        assert isinstance(backend, ChunkedTranscriptionBackend)

    def test_segments_go_to_whisper(self, backend):
        assert isinstance(backend._inner, OpenAIWhisperBackend)

    def test_split_and_parallelism_come_from_settings(self, backend):
        assert backend._max_segment_seconds == 120
        assert backend._max_parallel == 3
