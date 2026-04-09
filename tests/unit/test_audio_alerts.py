"""
Unit tests for whisper_dictate.audio.alerts.AudioAlertsManager.

subprocess.run is patched so no real audio hardware is required.
Tests verify the internal routing logic, volume clamping,
enabled/disabled behaviour, and the sound-file path resolution.
"""

import os
import pytest
from unittest.mock import patch

from whisper_dictate.audio.alerts import AudioAlertsManager, _sound_path


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
