"""
Unit tests for whisper_dictate.transcription.chunked.

Covers:
- split_frames(): no split when the audio fits, segment limits in seconds and
  bytes, no audio lost or duplicated, cuts placed in the quiet spot before a
  boundary, argument validation
- ChunkedTranscriptionBackend: pass-through for short audio, one request per
  segment, results joined in order, bounded parallelism, error propagation

The wrapped backend is a small fake — no network calls are made.
"""

import asyncio
import os
from unittest.mock import patch

import pytest

from tests.unit.audio_samples import RATE
from tests.unit.audio_samples import as_frames as _as_frames
from tests.unit.audio_samples import silence as _silence
from tests.unit.audio_samples import tone as _tone

from whisper_dictate.transcription.base import TranscriptionBackend
from whisper_dictate.transcription.chunked import (
    FAILED_SEGMENT_PLACEHOLDER,
    MIN_SEGMENT_SECONDS,
    SPLIT_SEARCH_SECONDS,
    ChunkedTranscriptionBackend,
    split_frames,
)
from whisper_dictate.transcription.openai_backend import OpenAIWhisperBackend

pytestmark = pytest.mark.unit

BYTES_PER_SECOND = RATE * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBackend(TranscriptionBackend):
    """Records every request, tracks how many run at once, and can fail on demand."""

    def __init__(self, texts=None, delays=None, fail_on=(), max_upload_bytes=None,
                 error=RuntimeError):
        self.max_upload_bytes = max_upload_bytes
        self._error = error
        self.calls = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self._texts = texts
        self._delays = delays or {}
        self._fail_on = set(fail_on)

    async def transcribe(self, frames, sample_rate, language=None):
        index = len(self.calls)
        self.calls.append((b"".join(frames), sample_rate, language))
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delays.get(index, 0.01))
            if index in self._fail_on:
                raise self._error(f"segment {index} failed")
            return self._texts[index] if self._texts else f"part{index}"
        finally:
            self.in_flight -= 1


def _chunked(inner, **overrides) -> ChunkedTranscriptionBackend:
    options = {"max_segment_seconds": 30, "max_parallel": 4, "verbose": False}
    options.update(overrides)
    return ChunkedTranscriptionBackend(inner, **options)


# ---------------------------------------------------------------------------
# split_frames
# ---------------------------------------------------------------------------


class TestSplitFrames:
    """split_frames() cuts audio into segments that respect both limits."""

    def test_audio_that_fits_is_a_single_segment(self):
        pcm = _tone(10)
        assert split_frames(_as_frames(pcm), RATE, max_seconds=30) == [pcm]

    def test_no_audio_gives_no_segments(self):
        assert split_frames([], RATE, max_seconds=30) == []

    def test_long_audio_is_split_within_the_time_limit(self):
        segments = split_frames(_as_frames(_tone(100)), RATE, max_seconds=30)
        assert len(segments) == 4
        assert all(len(s) <= 30 * BYTES_PER_SECOND for s in segments)

    def test_segments_rejoin_to_exactly_the_original_audio(self):
        pcm = _tone(47) + _silence(1) + _tone(52)
        assert b"".join(split_frames(_as_frames(pcm), RATE, max_seconds=30)) == pcm

    def test_cut_lands_in_the_silence_before_the_boundary(self):
        """A pause 2 s before the limit is where the cut goes — not mid-word at 30 s."""
        pcm = _tone(28) + _silence(0.5) + _tone(10)
        first = split_frames(_as_frames(pcm), RATE, max_seconds=30)[0]
        assert 28 * BYTES_PER_SECOND < len(first) <= 28.5 * BYTES_PER_SECOND

    def test_the_last_segment_is_never_a_sliver(self):
        """30 s limit, 60 s + one recorder chunk of audio: no 64 ms tail request."""
        pcm = _tone(60) + _tone(1024 / RATE)
        segments = split_frames(_as_frames(pcm), RATE, max_seconds=30)
        assert len(segments[-1]) >= MIN_SEGMENT_SECONDS * BYTES_PER_SECOND
        assert b"".join(segments) == pcm

    def test_without_a_pause_the_cut_stays_near_the_boundary(self):
        first = split_frames(_as_frames(_tone(40)), RATE, max_seconds=30)[0]
        assert (30 - SPLIT_SEARCH_SECONDS) * BYTES_PER_SECOND <= len(first)
        assert len(first) <= 30 * BYTES_PER_SECOND

    def test_byte_limit_wins_when_it_is_the_tighter_one(self):
        """At high sample rates the upload limit is hit before the time limit."""
        segments = split_frames(_as_frames(_tone(10)), RATE, max_seconds=30, max_bytes=100_000)
        assert len(segments) >= 4
        assert all(len(s) <= 100_000 for s in segments)

    def test_an_odd_byte_limit_still_cuts_on_whole_samples(self):
        segments = split_frames(_as_frames(_tone(10)), RATE, max_seconds=30, max_bytes=100_001)
        assert all(len(s) % 2 == 0 for s in segments)

    @pytest.mark.parametrize("max_seconds", [0, -5, float("nan"), float("inf")])
    def test_rejects_a_non_positive_time_limit(self, max_seconds):
        with pytest.raises(ValueError, match="max_seconds"):
            split_frames([_tone(1)], RATE, max_seconds=max_seconds)

    def test_rejects_a_byte_limit_smaller_than_one_sample(self):
        with pytest.raises(ValueError, match="max_bytes"):
            split_frames([_tone(1)], RATE, max_seconds=30, max_bytes=1)


# ---------------------------------------------------------------------------
# ChunkedTranscriptionBackend
# ---------------------------------------------------------------------------


class TestShortAudio:
    """Audio that fits in one segment behaves exactly as before: one request."""

    async def test_short_audio_is_one_request_with_the_original_frames(self):
        inner = FakeBackend(texts=["Hello"])
        frames = _as_frames(_tone(5))
        result = await _chunked(inner).transcribe(frames, RATE, language="es")
        assert result == "Hello"
        assert inner.calls == [(b"".join(frames), RATE, "es")]


class TestLongAudio:
    """Audio longer than a segment is sent as several requests, joined in order."""

    async def test_one_request_per_segment(self):
        inner = FakeBackend()
        await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert len(inner.calls) == 4

    async def test_every_request_fits_the_backends_upload_limit(self):
        inner = FakeBackend(max_upload_bytes=500_000)
        await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert len(inner.calls) > 4
        assert all(len(pcm) <= 500_000 for pcm, _, _ in inner.calls)

    async def test_requests_cover_the_whole_recording(self):
        inner = FakeBackend()
        pcm = _tone(100)
        await _chunked(inner).transcribe(_as_frames(pcm), RATE)
        assert b"".join(sent for sent, _, _ in inner.calls) == pcm

    async def test_sample_rate_and_language_reach_every_request(self):
        inner = FakeBackend()
        await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE, language="es")
        assert {(rate, lang) for _, rate, lang in inner.calls} == {(RATE, "es")}

    async def test_texts_are_joined_in_recording_order(self):
        inner = FakeBackend()
        result = await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert result == "part0 part1 part2 part3"

    async def test_order_holds_when_later_segments_finish_first(self):
        inner = FakeBackend(delays={0: 0.2, 1: 0.1, 2: 0.0, 3: 0.0})
        result = await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert result == "part0 part1 part2 part3"

    async def test_blank_segments_are_dropped_and_text_is_trimmed(self):
        inner = FakeBackend(texts=[" Hello ", "   ", "", "world "])
        result = await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert result == "Hello world"


class TestParallelism:
    """Segments run concurrently, never more than max_parallel at once."""

    async def test_segments_are_transcribed_concurrently(self):
        inner = FakeBackend(delays={i: 0.05 for i in range(4)})
        await _chunked(inner, max_parallel=4).transcribe(_as_frames(_tone(100)), RATE)
        assert inner.peak_in_flight == 4

    async def test_concurrency_never_exceeds_max_parallel(self):
        inner = FakeBackend(delays={i: 0.02 for i in range(8)})
        await _chunked(inner, max_segment_seconds=15, max_parallel=2).transcribe(
            _as_frames(_tone(115)), RATE
        )
        assert len(inner.calls) == 8
        assert inner.peak_in_flight == 2

    async def test_max_parallel_of_one_is_sequential(self):
        inner = FakeBackend()
        await _chunked(inner, max_parallel=1).transcribe(_as_frames(_tone(100)), RATE)
        assert inner.peak_in_flight == 1


class TestFailures:
    """A failed segment becomes a placeholder; the rest of the dictation survives."""

    async def test_a_failed_segment_is_replaced_by_the_placeholder_in_place(self):
        inner = FakeBackend(fail_on={1})
        result = await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert result == f"part0 {FAILED_SEGMENT_PLACEHOLDER} part2 part3"

    async def test_every_segment_failing_raises(self):
        """Nothing but placeholders is not worth typing — the client beeps an error."""
        inner = FakeBackend(fail_on={0, 1, 2, 3})
        with pytest.raises(RuntimeError, match="segment 0 failed"):
            await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)

    async def test_failures_with_nothing_heard_elsewhere_raise(self):
        """Placeholders around silence would type nothing but error markers."""
        inner = FakeBackend(texts=["", "", "", ""], fail_on={1})
        with pytest.raises(RuntimeError, match="segment 1 failed"):
            await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)

    async def test_a_cancelled_segment_is_not_turned_into_a_placeholder(self):
        inner = FakeBackend(fail_on={1}, error=asyncio.CancelledError)
        with pytest.raises(asyncio.CancelledError):
            await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)

    async def test_a_short_recording_that_fails_still_raises(self):
        """A single request has nothing to salvage: behaviour is unchanged."""
        inner = FakeBackend(fail_on={0})
        with pytest.raises(RuntimeError, match="segment 0 failed"):
            await _chunked(inner).transcribe(_as_frames(_tone(5)), RATE)

    async def test_no_request_is_left_running_after_a_failure(self):
        inner = FakeBackend(fail_on={0}, delays={0: 0.0, 1: 0.1, 2: 0.1, 3: 0.1})
        await _chunked(inner).transcribe(_as_frames(_tone(100)), RATE)
        assert inner.in_flight == 0
        assert len(inner.calls) == 4

    @pytest.mark.parametrize("options,message", [
        ({"max_segment_seconds": 0}, "max_segment_seconds"),
        ({"max_segment_seconds": float("inf")}, "max_segment_seconds"),
        ({"max_parallel": 0}, "max_parallel"),
    ])
    def test_rejects_invalid_limits(self, options, message):
        with pytest.raises(ValueError, match=message):
            _chunked(FakeBackend(), **options)


# ---------------------------------------------------------------------------
# The OpenAI upload budget
# ---------------------------------------------------------------------------


class TestOpenAIUploadBudget:
    """Segments cut for Whisper stay under its documented 25 MB limit as WAV files."""

    API_LIMIT_BYTES = 25_000_000

    @pytest.mark.parametrize("rate", [16000, 44100])
    def test_every_segment_wav_is_under_the_api_limit(self, rate):
        """At 44.1 kHz a 300 s segment is 26.5 MB — the byte budget must cut it."""
        with patch("whisper_dictate.transcription.openai_backend.AsyncOpenAI"):
            backend = OpenAIWhisperBackend(api_key="sk-test")
        pcm = b"\x00\x00" * (rate * 320)
        segments = split_frames([pcm], rate, max_seconds=300,
                                max_bytes=backend.max_upload_bytes)
        for segment in segments:
            path = backend._write_temp_wav([segment], rate)
            try:
                assert os.path.getsize(path) <= self.API_LIMIT_BYTES
            finally:
                os.unlink(path)
