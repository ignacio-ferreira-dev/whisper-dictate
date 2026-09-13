"""
Unit tests for whisper_dictate.audio.alerts.AudioAlertsManager.

The player subprocess is patched so no real audio hardware is required.
Tests verify the internal routing logic, volume clamping,
enabled/disabled behaviour, and the sound-file path resolution.
"""

import os
import subprocess
import time

import pytest
from unittest.mock import MagicMock, patch

from whisper_dictate.audio.alerts import AudioAlertsManager, _Player, _find_player, _sound_path

pytestmark = pytest.mark.unit


def _player_named(name: str) -> _Player:
    """Return the configured _Player entry for a given command name."""
    from whisper_dictate.audio.alerts import _PLAYERS
    return next(p for p in _PLAYERS if p.name == name)


_FFPLAY = _player_named("ffplay")


def _fake_process(returncode: int = 0, stderr: bytes = b""):
    """A stand-in for a Popen object that finishes immediately."""
    process = MagicMock()
    process.communicate.return_value = (b"", stderr)
    process.returncode = returncode
    return process


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
# Segment beep
# ---------------------------------------------------------------------------


class TestSegmentBeep:
    """play_segment() is a short, non-blocking tick made from the start sound."""

    def test_plays_the_start_sound_cut_short(self):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_play_file") as play_file:
            a._play_event("start", a.SEGMENT_BEEP_SECONDS)
        play_file.assert_called_once_with(_sound_path("recording_start.mp3"), 0.7)

    def test_is_played_in_a_background_thread(self):
        """The watchdog calls it and must keep checking the recording meanwhile."""
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread:
            a.play_segment()
        mock_thread.assert_called_once_with(
            target=a._play_event, args=("start", a.SEGMENT_BEEP_SECONDS), daemon=True
        )
        mock_thread.return_value.start.assert_called_once_with()

    def test_the_player_is_told_to_stop_early(self):
        a = AudioAlertsManager(player=_FFPLAY)
        command = a._command_for(_sound_path("recording_start.mp3"), a.SEGMENT_BEEP_SECONDS)
        assert "0.7" in command

    def test_is_a_no_op_when_disabled(self):
        a = AudioAlertsManager(enabled=False)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread:
            a.play_segment()
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

    def test_play_file_invokes_the_system_player(self):
        """_play_file starts the player as a subprocess and waits for it."""
        a = AudioAlertsManager(volume=0.5, player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _fake_process()
            a._play_file("/tmp/fake.mp3")
            mock_popen.assert_called_once()
            assert "/tmp/fake.mp3" in mock_popen.call_args[0][0]


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
    """play_shutdown() plays two overlapping beeps and blocks until both end."""

    def test_plays_the_stop_sound_twice(self):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process()) as mock_spawn, \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
        assert mock_spawn.call_count == AudioAlertsManager.SHUTDOWN_REPEATS
        for call_args in mock_spawn.call_args_list:
            assert call_args[0][0][-1].endswith("recording_end.mp3")

    def test_the_beeps_overlap(self, monkeypatch):
        """
        Every player is started before any of them is waited on, so the second
        beep begins while the first is still sounding. Playing them back to
        back was too easy to mistake for the ordinary stop beep.
        """
        monkeypatch.setattr(AudioAlertsManager, "SHUTDOWN_BEEP_OFFSET_SECONDS", 0.01)
        a = AudioAlertsManager(player=_FFPLAY)
        order = MagicMock()
        with patch.object(a, "_spawn", return_value=_fake_process()) as spawn, \
             patch.object(a, "_await") as await_, \
             patch("os.path.isfile", return_value=True):
            order.attach_mock(spawn, "spawn")
            order.attach_mock(await_, "wait")
            a.play_shutdown()
        assert [c[0] for c in order.mock_calls] == ["spawn", "spawn", "wait", "wait"]

    def test_the_second_beep_waits_for_the_offset(self, monkeypatch):
        monkeypatch.setattr(AudioAlertsManager, "SHUTDOWN_BEEP_OFFSET_SECONDS", 0.2)
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process()), \
             patch("os.path.isfile", return_value=True):
            started = time.monotonic()
            a.play_shutdown()
            elapsed = time.monotonic() - started
        assert elapsed >= 0.2

    def test_waits_for_every_beep_to_finish(self):
        """The process exits right after; an unawaited player would be killed."""
        process = _fake_process()
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=process), \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
        assert process.communicate.call_count == AudioAlertsManager.SHUTDOWN_REPEATS

    def test_does_not_spawn_a_thread(self):
        """
        Playback must be synchronous: the process exits right afterwards and
        a daemon thread would be killed before the sound is audible.
        """
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.threading.Thread") as mock_thread, \
             patch.object(a, "_spawn", return_value=_fake_process()), \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
        mock_thread.assert_not_called()

    def test_is_a_no_op_when_disabled(self):
        a = AudioAlertsManager(enabled=False)
        with patch.object(a, "_spawn") as mock_spawn:
            a.play_shutdown()
        mock_spawn.assert_not_called()

    def test_stops_after_a_player_fails_to_start(self):
        """No point staggering a second beep once the first could not start."""
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=None) as mock_spawn, \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
        assert mock_spawn.call_count == 1

    def test_reports_when_no_player_can_decode_the_format(self, capsys):
        """A missing decoder must be visible, not silent — see _warn_once."""
        a = AudioAlertsManager()
        with patch("whisper_dictate.audio.alerts._find_player", return_value=None):
            a.play_shutdown()
        assert "audio alerts disabled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Player selection
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


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestVolumeArguments:
    """The volume setting reaches the players that support it."""

    def test_ffplay_receives_percentage_volume(self):
        a = AudioAlertsManager(volume=0.5, player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process()) as mock_spawn:
            a._play_file("/tmp/fake.mp3")
            cmd = mock_spawn.call_args[0][0]
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
        with patch.object(a, "_spawn",
                          return_value=_fake_process(1, b"Failed to open audio file.")):
            a._play_file("/tmp/fake.mp3")
        assert "Failed to open audio file." in capsys.readouterr().out

    def test_repeated_failures_are_reported_only_once(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process(1, b"boom")):
            a._play_file("/tmp/fake.mp3")
            a._play_file("/tmp/fake.mp3")
        assert capsys.readouterr().out.count("Warning") == 1

    def test_successful_playback_is_quiet(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process()):
            a._play_file("/tmp/fake.mp3")
        assert capsys.readouterr().out == ""

    def test_a_player_that_cannot_be_started_is_reported(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch("whisper_dictate.audio.alerts.subprocess.Popen",
                   side_effect=OSError("no such file")):
            a._play_file("/tmp/fake.mp3")
        assert "could not be run" in capsys.readouterr().out

    def test_a_hung_player_is_killed_and_reported(self, capsys):
        a = AudioAlertsManager(player=_FFPLAY)
        process = _fake_process()
        process.communicate.side_effect = subprocess.TimeoutExpired("ffplay", 10)
        with patch.object(a, "_spawn", return_value=process):
            a._play_file("/tmp/fake.mp3")
        process.kill.assert_called_once()
        assert "timed out" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Beep length
# ---------------------------------------------------------------------------


class TestShutdownBeepLength:
    """The closing beeps are trimmed so quitting is quick, not a 6 s stall."""

    def test_each_beep_is_trimmed(self):
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process()) as mock_spawn, \
             patch("os.path.isfile", return_value=True):
            a.play_shutdown()
        for call_args in mock_spawn.call_args_list:
            cmd = call_args[0][0]
            assert cmd[cmd.index("-t") + 1] == "0.9"

    def test_regular_stop_alert_is_not_trimmed(self):
        """Only the shutdown beeps are cut short; a normal stop plays in full."""
        a = AudioAlertsManager(player=_FFPLAY)
        with patch.object(a, "_spawn", return_value=_fake_process()) as mock_spawn, \
             patch("os.path.isfile", return_value=True):
            a._play_event("stop")
        assert "-t" not in mock_spawn.call_args[0][0]

    def test_players_without_a_trim_flag_play_the_whole_file(self):
        """mpg123 has no seconds flag; it must still receive a valid command."""
        mpg123 = _player_named("mpg123")
        assert mpg123.command("/tmp/fake.mp3", 0.5, 0.9) == [
            "mpg123", "-q", "-f", "16384", "/tmp/fake.mp3",
        ]
