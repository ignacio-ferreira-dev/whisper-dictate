"""
Unit tests for the whisper_dictate CLI argument parser.

The parser takes its defaults from Settings, so these tests pin down the
resolution order that the README documents: command-line flag > .env > default.
"""

import asyncio

import pytest

from whisper_dictate.__main__ import build_backend, build_parser
from whisper_dictate.config import Settings
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


RATE = 16000


def _silence(seconds: int) -> list:
    return [b"\x00\x00" * RATE * seconds]


class TestBuildBackend:
    """The selected backend transcribes through the chunking wrapper, configured from Settings."""

    @pytest.fixture
    def whisper_requests(self, monkeypatch):
        """Replace the Whisper request with a fake that counts calls and concurrency."""
        stats = {"count": 0, "in_flight": 0, "peak": 0}

        async def fake_transcribe(self, frames, sample_rate, language=None):
            stats["count"] += 1
            stats["in_flight"] += 1
            stats["peak"] = max(stats["peak"], stats["in_flight"])
            await asyncio.sleep(0.01)
            stats["in_flight"] -= 1
            return "text"

        monkeypatch.setattr(OpenAIWhisperBackend, "transcribe", fake_transcribe)
        return stats

    @pytest.fixture
    def split_settings(self, monkeypatch) -> Settings:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("TRANSCRIPTION_CHUNK_SECONDS", "120")
        monkeypatch.setenv("TRANSCRIPTION_MAX_PARALLEL", "2")
        return Settings()

    async def test_long_recordings_become_several_whisper_requests(
        self, whisper_requests, split_settings
    ):
        await build_backend(split_settings, "openai").transcribe(_silence(250), RATE)
        assert whisper_requests["count"] == 3

    async def test_parallelism_comes_from_settings(self, whisper_requests, split_settings):
        await build_backend(split_settings, "openai").transcribe(_silence(250), RATE)
        assert whisper_requests["peak"] == 2

    async def test_short_recordings_stay_a_single_request(self, whisper_requests, split_settings):
        await build_backend(split_settings, "openai").transcribe(_silence(10), RATE)
        assert whisper_requests["count"] == 1

    def test_unknown_backend_is_rejected(self, split_settings):
        with pytest.raises(ValueError, match="'bogus'"):
            build_backend(split_settings, "bogus")

    def test_backend_flag_offers_the_registered_backends(self, settings):
        with pytest.raises(SystemExit):
            _parse(settings, "--backend", "bogus")
        assert _parse(settings, "--backend", "openai").backend == "openai"

    def test_backend_defaults_to_the_env(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIPTION_BACKEND", "openai")
        assert _parse(Settings()).backend == "openai"
