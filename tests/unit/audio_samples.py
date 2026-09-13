"""
Synthetic PCM audio (paInt16, mono) shared by the unit tests.
"""

from array import array

RATE = 16000


def tone(seconds: float, rate: int = RATE) -> bytes:
    """A loud square wave: every window has the same, high energy."""
    samples = int(seconds * rate)
    return (array("h", [8000, -8000]) * (samples // 2 + 1))[:samples].tobytes()


def silence(seconds: float, rate: int = RATE) -> bytes:
    return b"\x00\x00" * int(seconds * rate)


def as_frames(pcm: bytes, chunk_bytes: int = 2048) -> list:
    """Cut PCM into recorder-sized chunks, like AudioRecorder hands them over."""
    return [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
