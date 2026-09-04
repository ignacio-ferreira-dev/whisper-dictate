"""
Unit tests for whisper_dictate.audio.alerts.AudioAlertsManager.

subprocess.run is patched so no real audio hardware is required.
Tests verify the internal routing logic, volume clamping,
enabled/disabled behaviour, and the sound-file path resolution.
"""

import os
import pytest
from subprocess import CompletedProcess
from unittest.mock import patch

from whisper_dictate.audio.alerts import AudioAlertsManager, _Player, _find_player, _sound_path

pytestmark = pytest.mark.unit


def _player_named(name: str) -> _Player:
    """Return the configured _Player entry for a given command name."""
    from whisper_dictate.audio.alerts import _PLAYERS
    return next(p for p in _PLAYERS if p.name == name)


_FFPLAY = _player_named("ffplay")


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


class TestAudioAlertsManagerDefaults:
    """AudioAlertsManager stores volume and enabled flag correctly."""

    def test_volume_is_stored(self):
        a = AudioAlertsManager(volume=0.6, enabled=True)
        assert a.volume == pytest.approx(0.6)

    def test_enabled_flag_is_stored(self):
        a = AudioAlertsManager(enabled=False)
        assert a.enabled is False

    def test_volume_is_clamped_above_1(self):
        a = AudioAlertsManager(volume=1.5)
        assert a.volume == pytest.approx(1.0)

    def test_volume_is_clamped_below_0(self):
        a = AudioAlertsManager(volume=-0.5)
        assert a.volume == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """When enabled=False every play call is a silent no-op."""

    def test_play_start_does_not_spawn_thread(self):
        a = AudioAlertsManager(enabled=False)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread:
            a.play_start()
            mock_thread.assert_not_called()

    def test_play_stop_does_not_spawn_thread(self):
        a = AudioAlertsManager(enabled=False)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread:
            a.play_stop()
            mock_thread.assert_not_called()

    def test_play_done_does_not_spawn_thread(self):
        a = AudioAlertsManager(enabled=False)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread:
            a.play_done()
            mock_thread.assert_not_called()

    def test_play_error_does_not_spawn_thread(self):
        a = AudioAlertsManager(enabled=False)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread:
            a.play_error()
            mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# File routing
# ---------------------------------------------------------------------------


class TestSoundFileRouting:
    """_play_event() plays the correct MP3 file via subprocess."""

    @pytest.mark.parametrize("event,filename", [
        ("start", "recording_start.mp3"),
        ("stop",  "recording_end.mp3"),
        ("done",  "transcription_end_success.mp3"),
        ("error", "transcription_end_error.mp3"),
    ])
    def test_uses_correct_sound_file_per_event(self, event, filename):
        a = AudioAlertsManager(volume=0.5)
        with patch.object(a, "_play_file") as mock_play_file, \
             patch("os.path.isfile", return_value=True):
            a._play_event(event)
            called_path = mock_play_file.call_args[0][0]
            assert called_path.endswith(filename)

    def test_does_nothing_when_file_missing(self):
        """When the sound file doesn't exist, _play_event returns silently."""
        a = AudioAlertsManager(volume=0.5)
        with patch.object(a, "_play_file") as mock_play_file, \
             patch("os.path.isfile", return_value=False):
            a._play_event("start")
            mock_play_file.assert_not_called()

    def test_play_file_calls_subprocess(self):
        """_play_file uses subprocess.run to invoke the system player."""
        a = AudioAlertsManager(volume=0.5)
        with patch("whisper_dictate.audio.alerts.subprocess.run") as mock_run:
            a._play_file("/tmp/fake.mp3")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "/tmp/fake.mp3" in cmd


# ---------------------------------------------------------------------------
# Sound path resolution
# ---------------------------------------------------------------------------


class TestSoundPathResolution:
    """_sound_path() resolves files relative to the package sounds/ directory."""

    def test_returns_absolute_path(self):
        path = _sound_path("recording_start.mp3")
        assert os.path.isabs(path)

    def test_path_ends_with_correct_filename(self):
        path = _sound_path("recording_start.mp3")
        assert path.endswith("recording_start.mp3")

    def test_sounds_directory_exists(self):
        path = _sound_path("recording_start.mp3")
        sounds_dir = os.path.dirname(path)
        assert os.path.isdir(sounds_dir)

    @pytest.mark.parametrize("filename", [
        "recording_start.mp3",
        "recording_end.mp3",
        "transcription_end_success.mp3",
        "transcription_end_error.mp3",
    ])
    def test_all_sound_files_exist_in_package(self, filename):
        assert os.path.isfile(_sound_path(filename)), \
            f"Missing sound file: {filename}"


# ---------------------------------------------------------------------------
# Shutdown alert
# ---------------------------------------------------------------------------


class TestShutdownAlert:
    """play_shutdown() plays the stop sound twice, blocking until it finishes."""

    def test_plays_stop_sound_twice(self):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_play_file") as mock_play_file, \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
            assert mock_play_file.call_count == AudioAlertsManager.SHUTDOWN_REPEATS
            played = {c[0][0] for c in mock_play_file.call_args_list}
            assert all(p.endswith("recording_end.mp3") for p in played)

    def test_does_not_spawn_a_thread(self):
        """
        Playback must be synchronous: the process exits right afterwards and
        a daemon thread would be killed before the sound is audible.
        """
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread, \
             patch.object(a, "_play_file"), \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
            mock_thread.assert_not_called()

    def test_is_a_no_op_when_disabled(self):
        a = AudioAlertsManager(enabled=False)
        with patch.object(a, "_play_file") as mock_play_file:
            a.play_shutdown()
            mock_play_file.assert_not_called()

    def test_reports_when_no_player_can_decode_the_format(self, capsys):
        """A missing decoder must be visible, not silent — see _warn_once."""
        a = AudioAlertsManager()
        with patch("whisper_dictate.audio.alerts._find_player", return_value=None):
            a.play_shutdown()
        assert "audio alerts disabled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestPlayerSelection:
    """The player is chosen by format, not merely by availability."""

    def test_mp3_never_selects_paplay(self):
        """
        paplay only decodes what libsndfile supports. Selecting it for the MP3
        alerts made every sound fail silently.
        """
        with patch("whisper_dictate.audio.alerts.shutil.which", lambda name: True):
            _find_player.cache_clear()
            player = _find_player(".mp3")
        _find_player.cache_clear()
        assert player.name != "paplay"

    def test_mp3_prefers_ffplay_when_available(self):
        with patch("whisper_dictate.audio.alerts.shutil.which", lambda name: True):
            _find_player.cache_clear()
            player = _find_player(".mp3")
        _find_player.cache_clear()
        assert player.name == "ffplay"

    def test_falls_back_to_an_mp3_capable_player(self):
        """With no ffmpeg installed, mpg123 is used rather than a WAV-only player."""
        with patch("whisper_dictate.audio.alerts.shutil.which",
                   lambda name: name != "ffplay"):
            _find_player.cache_clear()
            player = _find_player(".mp3")
        _find_player.cache_clear()
        assert player.name == "mpg123"

    def test_returns_none_when_nothing_can_decode_the_format(self):
        with patch("whisper_dictate.audio.alerts.shutil.which", lambda name: False):
            _find_player.cache_clear()
            player = _find_player(".mp3")
        _find_player.cache_clear()
        assert player is None

    def test_wav_can_use_paplay(self):
        with patch("whisper_dictate.audio.alerts.shutil.which",
                   lambda name: name == "paplay"):
            _find_player.cache_clear()
            player = _find_player(".wav")
        _find_player.cache_clear()
        assert player.name == "paplay"


class TestVolumeArguments:
    """The volume setting reaches the players that support it."""

    def test_ffplay_receives_percentage_volume(self):
        a = AudioAlertsManager(volume=0.5, player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.run") as mock_run:
            a._play_file("/tmp/fake.mp3")
            cmd = mock_run.call_args[0][0]
            assert cmd == [
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-volume", "50", "/tmp/fake.mp3",
            ]

    def test_paplay_receives_linear_pulseaudio_volume(self):
        """paplay's --volume is linear, where 65536 (PA_VOLUME_NORM) is 100%."""
        paplay = _player_named("paplay")
        assert paplay.command("/tmp/fake.wav", 0.5) == [
            "paplay", "--volume", "32768", "/tmp/fake.wav",
        ]

    def test_aplay_receives_no_volume_flag(self):
        """aplay has no volume flag; passing one would make playback fail."""
        aplay = _player_named("aplay")
        assert aplay.command("/tmp/fake.wav", 0.5) == ["aplay", "-q", "/tmp/fake.wav"]


# ---------------------------------------------------------------------------
# Playback failures
# ---------------------------------------------------------------------------


class TestPlaybackFailureReporting:
    """A player that errors out is reported once, not swallowed and not repeated."""

    def test_non_zero_exit_is_reported(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.run",
                   return_value=CompletedProcess([], 1, b"", b"Failed to open audio file.")):
            a._play_file("/tmp/fake.mp3")
        assert "Failed to open audio file." in capsys.readouterr().out

    def test_repeated_failures_are_reported_only_once(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.run",
                   return_value=CompletedProcess([], 1, b"", b"boom")):
            a._play_file("/tmp/fake.mp3")
            a._play_file("/tmp/fake.mp3")
        assert capsys.readouterr().out.count("Warning") == 1

    def test_successful_playback_is_quiet(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.run",
                   return_value=CompletedProcess([], 0, b"", b"")):
            a._play_file("/tmp/fake.mp3")
        assert capsys.readouterr().out == ""


class TestShutdownBeepLength:
    """The closing beeps are trimmed so quitting is quick, not a 6 s stall."""

    def test_each_beep_is_trimmed(self):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.run",
                   return_value=CompletedProcess([], 0, b"", b"")) as mock_run, \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
            for call_args in mock_run.call_args_list:
                cmd = call_args[0][0]
                assert "-t" in cmd
                assert cmd[cmd.index("-t") + 1] == "0.9"

    def test_regular_stop_alert_is_not_trimmed(self):
        """Only the shutdown beeps are cut short; a normal stop plays in full."""
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.run",
                   return_value=CompletedProcess([], 0, b"", b"")) as mock_run, \
             patch("os.path.isfile", return_value=True):
            a._play_event("stop")
            assert "-t" not in mock_run.call_args[0][0]

    def test_players_without_a_trim_flag_play_the_whole_file(self):
        """mpg123 has no seconds flag; it must still receive a valid command."""
        mpg123 = _player_named("mpg123")
        assert mpg123.command("/tmp/fake.mp3", 0.5, 0.9) == [
            "mpg123", "-q", "-f", "16384", "/tmp/fake.mp3",
        ]
