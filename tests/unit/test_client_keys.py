"""
Unit tests for the key bindings of whisper_dictate.client.

Covers key-name resolution, the quit binding (HOME, not ESC), and the
shutdown sequence that plays the closing sound after audio is released.
"""

import pytest
from unittest.mock import MagicMock, call

from pynput import keyboard as kb_module

from whisper_dictate.client import WhisperDictateClient, resolve_key

pytestmark = pytest.mark.unit


def _client(**kwargs) -> WhisperDictateClient:
    """Build a client with every collaborator mocked out."""
    return WhisperDictateClient(
        backend=MagicMock(),
        alerts=MagicMock(),
        typer=MagicMock(),
        verbose=False,
        **kwargs,
    )


class TestRecordingLimits:
    """The recording cap and the segment beep interval reach the recorder's watchdog."""

    def test_cap_is_forwarded_to_the_recorder(self):
        assert _client(max_recording_seconds=90)._recorder.max_recording_seconds == 90

    def test_default_cap_is_half_an_hour(self):
        assert _client()._recorder.max_recording_seconds == 1800

    def test_segment_interval_is_forwarded_to_the_recorder(self):
        assert _client(segment_seconds=600)._recorder.segment_seconds == 600

    def test_no_segment_beep_unless_asked(self):
        assert _client()._recorder.segment_seconds is None


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


class TestResolveKey:
    """resolve_key() maps names to the pynput objects that events compare against."""

    @pytest.mark.parametrize("name,expected", [
        ("f9", kb_module.Key.f9),
        ("home", kb_module.Key.home),
        ("HOME", kb_module.Key.home),
        ("esc", kb_module.Key.esc),
    ])
    def test_resolves_special_keys(self, name, expected):
        assert resolve_key(name) == expected

    def test_resolves_single_character_to_keycode(self):
        assert resolve_key("q") == kb_module.KeyCode.from_char("q")

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            resolve_key("not_a_key")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestKeyDefaults:
    """The client binds F9 to recording and HOME to quitting by default."""

    def test_default_hotkey_is_f9(self):
        assert _client()._hotkey == kb_module.Key.f9

    def test_default_quit_key_is_home(self):
        assert _client()._quit_key == kb_module.Key.home

    def test_quit_key_is_configurable(self):
        assert _client(quit_key="f12")._quit_key == kb_module.Key.f12

    def test_unknown_quit_key_falls_back_to_default(self):
        client = _client(quit_key="nonsense")
        assert client._quit_key == kb_module.Key.home

    def test_display_name_matches_the_bound_key(self):
        """A rejected name must not be shown in the banner as if it were bound."""
        client = _client(quit_key="nonsense")
        assert client._quit_key_name == "HOME"


# ---------------------------------------------------------------------------
# Quit behaviour
# ---------------------------------------------------------------------------


class TestQuitBinding:
    """ESC no longer quits; only the configured quit key does."""

    def test_escape_does_not_quit(self):
        client = _client()
        client._on_key_press(kb_module.Key.esc)
        assert client._should_quit is False

    def test_quit_key_sets_the_quit_flag(self):
        client = _client()
        client._on_key_press(kb_module.Key.home)
        assert client._should_quit is True

    def test_unrelated_key_does_not_quit(self):
        client = _client()
        client._on_key_press(kb_module.KeyCode.from_char("a"))
        assert client._should_quit is False

    def test_escape_quits_when_explicitly_bound(self):
        client = _client(quit_key="esc")
        client._on_key_press(kb_module.Key.esc)
        assert client._should_quit is True


# ---------------------------------------------------------------------------
# Shutdown sequence
# ---------------------------------------------------------------------------


class TestShutdownSequence:
    """_shutdown() releases audio first, then plays the closing sound."""

    def test_plays_the_shutdown_sound(self):
        client = _client()
        client._shutdown()
        client._alerts.play_shutdown.assert_called_once_with()

    def test_recorder_is_torn_down_before_the_sound(self):
        """
        The player subprocess must not overlap an open PortAudio stream —
        that overlap is what corrupted the heap on Linux/PulseAudio.
        """
        client = _client()
        order = MagicMock()
        client._recorder = MagicMock()
        order.attach_mock(client._recorder.teardown, "teardown")
        order.attach_mock(client._alerts.play_shutdown, "play_shutdown")

        client._shutdown()

        assert order.mock_calls == [call.teardown(), call.play_shutdown()]
