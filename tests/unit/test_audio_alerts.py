"""
Unit tests for whisper_dictate.audio.alerts.AudioAlertsManager.

pygame and PyAudio are mocked so no real audio hardware is required.
Tests verify the internal routing logic (file vs fallback), volume clamping,
enabled/disabled behaviour, and the sound-file path resolution.
"""

import os
import threading
import pytest
from unittest.mock import MagicMock, patch, call

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
    """_play_event() uses the MP3 file when pygame is available and the file exists."""

    def _alerts_with_mocked_pygame(self) -> AudioAlertsManager:
        with patch("whisper_dictate.audio.alerts.AudioAlertsManager._init_pygame", return_value=True):
            return AudioAlertsManager(volume=0.5)

    @pytest.mark.parametrize("event,filename", [
        ("start", "recording_start.mp3"),
        ("stop",  "recording_end.mp3"),
        ("done",  "transcription_end_success.mp3"),
        ("error", "transcription_end_error.mp3"),
    ])
    def test_uses_correct_sound_file_per_event(self, event, filename):
        a = self._alerts_with_mocked_pygame()
        with patch.object(a, "_play_file") as mock_play_file, \
             patch("os.path.isfile", return_value=True):
            a._play_event(event)
            called_path = mock_play_file.call_args[0][0]
            assert called_path.endswith(filename)

    def test_falls_back_to_tones_when_file_missing(self):
        a = self._alerts_with_mocked_pygame()
        with patch.object(a, "_play_tones") as mock_tones, \
             patch("os.path.isfile", return_value=False):
            a._play_event("start")
            mock_tones.assert_called_once()

    def test_falls_back_to_tones_when_pygame_unavailable(self):
        with patch("whisper_dictate.audio.alerts.AudioAlertsManager._init_pygame", return_value=False):
            a = AudioAlertsManager(volume=0.5)
        with patch.object(a, "_play_tones") as mock_tones, \
             patch("os.path.isfile", return_value=True):
            a._play_event("start")
            mock_tones.assert_called_once()


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
