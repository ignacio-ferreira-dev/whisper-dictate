"""
Chunked Transcription
=====================

Long recordings do not fit in a single transcription request: the Whisper
API rejects uploads larger than 25 MB (about 13 minutes of 16 kHz audio).

ChunkedTranscriptionBackend wraps any TranscriptionBackend, splits the audio
into segments that fit, transcribes them in parallel and joins the texts in
their original order. Audio that already fits is passed through untouched,
so short dictations still cost exactly one request.

Segments are cut at the quietest moment shortly before each boundary rather
than at a fixed offset, so a word is not sliced in half between two requests.
"""

import asyncio
import time
from array import array
from typing import List, Optional

from whisper_dictate.transcription.base import INT16_SAMPLE_WIDTH, TranscriptionBackend

#: How far back from a segment boundary to look for a quiet place to cut.
SPLIT_SEARCH_SECONDS: float = 3.0

#: Length of the windows whose energy is compared when looking for that place.
SPLIT_WINDOW_SECONDS: float = 0.05

#: Shortest last segment produced. A sliver left over after the final cut is
#: likely silence, and the API rejects clips under 0.1 s.
MIN_SEGMENT_SECONDS: float = 1.0

#: Typed in place of a segment whose request failed, so one lost request does
#: not throw away the rest of a long dictation.
FAILED_SEGMENT_PLACEHOLDER: str = "[error processing this extract of voice command]"


def split_frames(
    frames: List[bytes],
    sample_rate: int,
    max_seconds: float,
    max_bytes: Optional[int] = None,
) -> List[bytes]:
    """
    Split PCM audio into segments no longer than max_seconds and no larger
    than max_bytes.

    Args:
        frames:      List of raw PCM byte chunks (paInt16, mono).
        sample_rate: Sample rate in Hz.
        max_seconds: Longest segment allowed, in seconds.
        max_bytes:   Largest segment allowed, in bytes of PCM (None: no limit).

    Returns:
        The segments, in order. Joined back together they are exactly the
        input audio. Empty list if there is no audio.
    """
    if max_seconds <= 0:
        raise ValueError(f"max_seconds must be positive, got {max_seconds}")
    if max_bytes is not None and max_bytes < INT16_SAMPLE_WIDTH:
        raise ValueError(f"max_bytes must hold at least one sample, got {max_bytes}")

    pcm = b"".join(frames)
    budget = _samples_to_bytes(int(max_seconds * sample_rate))
    if max_bytes is not None:
        budget = min(budget, _samples_to_bytes(max_bytes // INT16_SAMPLE_WIDTH))
    budget = max(budget, INT16_SAMPLE_WIDTH)

    # Never search more than a quarter of a segment back, nor keep a tail longer
    # than a quarter: every segment then keeps at least half its allowed length
    # and the loop always makes progress.
    quarter = _samples_to_bytes(budget // INT16_SAMPLE_WIDTH // 4)
    search = min(_samples_to_bytes(int(SPLIT_SEARCH_SECONDS * sample_rate)), quarter)
    min_tail = min(_samples_to_bytes(int(MIN_SEGMENT_SECONDS * sample_rate)), quarter)
    window = max(_samples_to_bytes(int(SPLIT_WINDOW_SECONDS * sample_rate)), INT16_SAMPLE_WIDTH)

    segments: List[bytes] = []
    start = 0
    while len(pcm) - start > budget:
        # Stop short of the end so the last segment is not a sliver.
        end = min(start + budget, len(pcm) - min_tail)
        cut = _quietest_cut(pcm, end - search, end, window)
        segments.append(pcm[start:cut])
        start = cut
    if start < len(pcm):
        segments.append(pcm[start:])
    return segments


def _samples_to_bytes(samples: int) -> int:
    return samples * INT16_SAMPLE_WIDTH


def _quietest_cut(pcm: bytes, low: int, high: int, window: int) -> int:
    """
    Return the byte offset in [low, high] at the end of the quietest window.

    On a tie the latest window wins, so audio with no pause at all is cut as
    close to the boundary as possible instead of 3 s early. Falls back to
    `high` (a plain cut at the boundary) when the search range is shorter
    than one window.
    """
    best_offset, best_energy = high, None
    for offset in range(low, high - window + 1, window):
        samples = array("h", pcm[offset:offset + window])
        energy = sum(sample * sample for sample in samples)
        if best_energy is None or energy <= best_energy:
            best_offset, best_energy = offset + window, energy
    return best_offset


class ChunkedTranscriptionBackend(TranscriptionBackend):
    """
    Transcribe long audio by splitting it into segments sent in parallel.

    Wraps another TranscriptionBackend; the wrapped backend never sees a
    segment longer than max_segment_seconds or larger than its own
    max_upload_bytes.
    """

    def __init__(
        self,
        inner: TranscriptionBackend,
        max_segment_seconds: float,
        max_parallel: int,
        verbose: bool = True,
    ):
        """
        Args:
            inner:               Backend that transcribes each segment.
            max_segment_seconds: Longest segment sent in one request.
            max_parallel:        Most requests in flight at the same time.
            verbose:             When True, print status messages.
        """
        if max_segment_seconds <= 0:
            raise ValueError(f"max_segment_seconds must be positive, got {max_segment_seconds}")
        if max_parallel < 1:
            raise ValueError(f"max_parallel must be at least 1, got {max_parallel}")
        self._inner = inner
        self._max_segment_seconds = max_segment_seconds
        self._max_parallel = max_parallel
        self._verbose = verbose

    async def transcribe(
        self,
        frames: list,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe the audio, splitting it first when it is too long.

        A segment whose request fails is replaced by FAILED_SEGMENT_PLACEHOLDER
        and the other segments are kept. Every segment is awaited, so no
        request is left running.

        Raises:
            Exception: when the audio is a single request and it fails, or when
                       every segment failed — there is nothing worth typing.
        """
        segments = split_frames(
            frames, sample_rate, self._max_segment_seconds, self._inner.max_upload_bytes
        )
        if len(segments) <= 1:
            return await self._inner.transcribe(frames, sample_rate, language)

        self._log(
            f"Long recording - transcribing {len(segments)} segments "
            f"({min(len(segments), self._max_parallel)} at a time)"
        )
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def transcribe_segment(segment: bytes) -> str:
            async with semaphore:
                return await self._inner.transcribe([segment], sample_rate, language)

        results = await asyncio.gather(
            *(transcribe_segment(segment) for segment in segments),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if len(failures) == len(results):
            raise failures[0]

        texts = []
        for number, result in enumerate(results, start=1):
            if isinstance(result, BaseException):
                self._log(f"Segment {number}/{len(results)} failed ({result}) - typing a placeholder")
                texts.append(FAILED_SEGMENT_PLACEHOLDER)
            elif result and result.strip():
                texts.append(result.strip())
        return " ".join(texts)

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[{time.strftime('%H:%M:%S')}] [Transcriber] {msg}")
