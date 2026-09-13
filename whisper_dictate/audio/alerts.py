"""
Audio Alerts Manager
====================

Plays sound files to signal recording state changes.
Uses a subprocess to play the sound files so that audio playback runs in a
completely separate process from PyAudio. This avoids the malloc heap
corruption crash that occurs when pygame/SDL and PortAudio both access the
audio subsystem in the same process on Linux.

The player is picked by file format, not merely by availability: paplay and
aplay cannot decode MP3 (paplay only handles what libsndfile supports), so
choosing them for the MP3 alerts made every sound fail silently.

Sound files live in whisper_dictate/sounds/:
    recording_start.mp3           -> play_start()
    recording_end.mp3             -> play_stop() / play_shutdown()
    transcription_end_success.mp3 -> play_done()
    transcription_end_error.mp3   -> play_error()

Usage:
    alerts = AudioAlertsManager()
    alerts.play_start()     # recording started
    alerts.play_stop()      # recording stopped
    alerts.play_done()      # transcription succeeded
    alerts.play_error()     # something went wrong
    alerts.play_shutdown()  # application is quitting (overlapping double beep)
"""

import functools
import os
import shutil
import subprocess
import threading
from typing import List, Optional


_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "..", "sounds")

#: Hard cap on how long a single playback subprocess may run.
_PLAYBACK_TIMEOUT_SECONDS = 10


def _sound_path(filename: str) -> str:
    return os.path.normpath(os.path.join(_SOUNDS_DIR, filename))


class _Player:
    """A command-line audio player, the formats it decodes, and its volume flag."""

    def __init__(
        self,
        name: str,
        args: tuple,
        formats: frozenset,
        volume_flag=None,
        trim_flag: Optional[str] = None,
    ):
        self.name = name
        self.args = args
        self.formats = formats
        self._volume_flag = volume_flag
        self._trim_flag = trim_flag

    def volume_args(self, volume: float) -> List[str]:
        """Return the flags that set `volume` (0.0-1.0), or [] if unsupported."""
        if self._volume_flag is None:
            return []
        flag, scale = self._volume_flag
        return [flag, str(int(volume * scale))]

    def trim_args(self, max_seconds: Optional[float]) -> List[str]:
        """Return the flags that stop playback early, or [] if unsupported."""
        if max_seconds is None or self._trim_flag is None:
            return []
        return [self._trim_flag, f"{max_seconds:g}"]

    def command(
        self, path: str, volume: float, max_seconds: Optional[float] = None
    ) -> List[str]:
        """Return the full argv used to play `path`."""
        return [
            self.name,
            *self.args,
            *self.volume_args(volume),
            *self.trim_args(max_seconds),
            path,
        ]


#: Candidate players in preference order. Only players that can actually decode
#: the requested format are considered, so a missing ffmpeg degrades to "no
#: sound" rather than to a player that errors out on every file.
_PLAYERS = (
    # ffplay takes a 0-100 percentage.
    _Player("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet"),
            frozenset({".mp3", ".wav", ".ogg", ".flac"}), ("-volume", 100),
            trim_flag="-t"),
    # mpg123 takes a 0-32768 scale factor.
    _Player("mpg123", ("-q",), frozenset({".mp3"}), ("-f", 32768)),
    # paplay takes a linear PulseAudio volume where 65536 (PA_VOLUME_NORM) is 100%.
    _Player("paplay", (), frozenset({".wav", ".ogg", ".flac"}), ("--volume", 65536)),
    # aplay has no volume flag of its own.
    _Player("aplay", ("-q",), frozenset({".wav"})),
)


@functools.lru_cache(maxsize=None)
def _find_player(extension: str) -> Optional[_Player]:
    """
    Return the preferred available player able to decode `extension`.

    Returns None when nothing on PATH can play that format, which turns every
    play call into a no-op.
    """
    for player in _PLAYERS:
        if extension in player.formats and shutil.which(player.name):
            return player
    return None


class AudioAlertsManager:
    """
    Manages audible feedback for recording state changes.

    Each sound is played by spawning a subprocess so that the audio
    player runs completely outside the Python process. This guarantees
    no shared memory or library state with PyAudio/PortAudio.
    """

    _SOUND_MAP = {
        "start": "recording_start.mp3",
        "stop":  "recording_end.mp3",
        "done":  "transcription_end_success.mp3",
        "error": "transcription_end_error.mp3",
    }

    #: Number of times play_shutdown() repeats the stop sound. Two beeps are
    #: what distinguishes "app quitting" from "recording stopped".
    SHUTDOWN_REPEATS: int = 2

    #: How long after one shutdown beep starts the next one starts. Shorter
    #: than SHUTDOWN_BEEP_SECONDS on purpose: the beeps overlap, so the pair
    #: is heard as a single stuttered sound rather than as the ordinary stop
    #: beep played twice, which is ambiguous with "the recording stopped".
    SHUTDOWN_BEEP_OFFSET_SECONDS: float = 0.5

    #: Each shutdown beep is cut to this length. recording_end.mp3 is 3.2 s
    #: long but only its first 0.8 s carry sound, so playing it whole twice
    #: would stall the exit for ~6.5 s. Players without a trim flag play it
    #: whole; the overlap still happens, the exit is just slower.
    SHUTDOWN_BEEP_SECONDS: float = 0.9

    #: The segment beep is the start sound cut to its audible part
    #: (recording_start.mp3 goes silent at ~0.67 s), so it reads as a short
    #: "still recording" tick rather than as a new recording starting.
    SEGMENT_BEEP_SECONDS: float = 0.7

    def __init__(
        self,
        volume: float = 0.8,
        enabled: bool = True,
        player: Optional[_Player] = None,
    ):
        """
        Args:
            volume:  Playback volume in [0.0, 1.0]. Applied by the players that
                     support it (ffplay, mpg123, paplay); ignored by aplay.
            enabled: When False, all play calls are silent no-ops.
            player:  Explicit player to use instead of auto-detection. Mainly
                     useful for tests.
        """
        self.volume = max(0.0, min(1.0, volume))
        self.enabled = enabled
        self._player_override = player
        self._playback_failed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_start(self) -> None:
        """
        Play the recording-start sound synchronously.

        Blocks until the subprocess finishes so that PyAudio does not open
        the capture stream while the audio player is still running.
        """
        self._play_sync("start")

    def play_stop(self) -> None:
        """Play the recording-end sound asynchronously."""
        self._play_async("stop")

    def play_done(self) -> None:
        """Play the transcription-success sound asynchronously."""
        self._play_async("done")

    def play_error(self) -> None:
        """Play the error sound asynchronously."""
        self._play_async("error")

    def play_segment(self) -> None:
        """
        Play a short beep, asynchronously, while a recording keeps going.

        Marks that a long recording has started a new transcription segment.
        Asynchronous because it is called from the recorder's watchdog thread,
        which must keep checking the recording's length while it plays.
        """
        self._play_async("start", max_seconds=self.SEGMENT_BEEP_SECONDS)

    def play_shutdown(self) -> None:
        """
        Play the recording-end sound twice, overlapping, synchronously.

        The second beep starts SHUTDOWN_BEEP_OFFSET_SECONDS after the first,
        before the first has finished, so the two run together as one
        stuttered sound. Playing them back to back instead was too easy to
        confuse with an ordinary "recording stopped" beep.

        Blocking is deliberate: the process exits right after this returns,
        and a background thread would be killed before the sound is audible.
        """
        if not self.enabled:
            return
        self._play_overlapping(
            "stop",
            repeats=self.SHUTDOWN_REPEATS,
            offset=self.SHUTDOWN_BEEP_OFFSET_SECONDS,
            max_seconds=self.SHUTDOWN_BEEP_SECONDS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_sync(self, event: str) -> None:
        """Play a sound and block until the subprocess finishes."""
        if not self.enabled:
            return
        self._play_event(event)

    def _play_async(self, event: str, max_seconds: Optional[float] = None) -> None:
        """Spawn a daemon thread so playback never blocks the caller."""
        if not self.enabled:
            return
        threading.Thread(
            target=self._play_event, args=(event, max_seconds), daemon=True
        ).start()

    def _play_event(self, event: str, max_seconds: Optional[float] = None) -> None:
        """Play the sound file for the given event via subprocess."""
        path = _sound_path(self._SOUND_MAP[event])
        if os.path.isfile(path):
            self._play_file(path, max_seconds)

    def _play_overlapping(
        self, event: str, repeats: int, offset: float, max_seconds: Optional[float]
    ) -> None:
        """
        Start `repeats` playbacks `offset` seconds apart and wait for them all.

        Each playback is its own process, so they genuinely overlap instead of
        queuing up behind each other.
        """
        path = _sound_path(self._SOUND_MAP[event])
        if not os.path.isfile(path):
            return
        command = self._command_for(path, max_seconds)
        if command is None:
            return

        processes = []
        for index in range(repeats):
            if index:
                threading.Event().wait(offset)
            process = self._spawn(command)
            if process is None:
                break
            processes.append(process)

        for process in processes:
            self._await(process, path)

    def _play_file(self, path: str, max_seconds: Optional[float] = None) -> None:
        """Invoke the system audio player as a subprocess and wait for it."""
        command = self._command_for(path, max_seconds)
        if command is None:
            return
        process = self._spawn(command)
        if process is not None:
            self._await(process, path)

    def _command_for(self, path: str, max_seconds: Optional[float]) -> Optional[List[str]]:
        """Return the argv that plays `path`, or None if no player can decode it."""
        player = self._player_for(path)
        if player is None:
            self._warn_once(f"no audio player available for {os.path.basename(path)}")
            return None
        return player.command(path, self.volume, max_seconds)

    def _spawn(self, command: List[str]):
        """Start a player process, or return None if it could not be started."""
        try:
            return subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._warn_once(f"{command[0]} could not be run: {exc}")
            return None

    def _await(self, process, path: str) -> None:
        """Wait for a player process, reporting a failure at most once."""
        try:
            _, stderr = process.communicate(timeout=_PLAYBACK_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            self._warn_once(f"playback of {os.path.basename(path)} timed out")
            return
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip().splitlines()
            self._warn_once(
                f"failed to play {os.path.basename(path)}"
                + (f": {detail[-1]}" if detail else "")
            )

    def _player_for(self, path: str) -> Optional[_Player]:
        """Return the player to use for `path`, honouring an explicit override."""
        if self._player_override is not None:
            return self._player_override
        return _find_player(os.path.splitext(path)[1].lower())

    def _warn_once(self, message: str) -> None:
        """
        Report the first playback failure and stay quiet afterwards.

        Alerts are non-essential, so a broken player must not spam the console
        on every keypress — but it must not fail silently either, which is how
        the MP3-incompatible paplay went unnoticed.
        """
        if self._playback_failed:
            return
        self._playback_failed = True
        print(f"Warning: audio alerts disabled - {message}")
