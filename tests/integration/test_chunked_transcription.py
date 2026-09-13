"""
Integration test for ChunkedTranscriptionBackend against the real Whisper API.

A 10.5 s LibriSpeech clip is repeated three times and sent with 12 s segments,
so it must go out as three parallel requests whose texts come back in order.

Requirements: OPENAI_API_KEY. Spends a few cents' worth of audio (~32 s), so
it only runs when asked for explicitly — the key alone is not enough, since
config loads it from .env on every test run.

Run with:
    RUN_API_TESTS=1 pytest tests/integration/test_chunked_transcription.py -v
"""

import os
import wave
from pathlib import Path

import pytest

from whisper_dictate.config import get_settings
from whisper_dictate.transcription.chunked import ChunkedTranscriptionBackend
from whisper_dictate.transcription.openai_backend import OpenAIWhisperBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_API_TESTS") != "1",
                       reason="calls the paid Whisper API — set RUN_API_TESTS=1"),
    pytest.mark.skipif(not get_settings().openai_api_key, reason="OPENAI_API_KEY is not set"),
]

CLIP = Path(__file__).parents[1] / "resources" / "3081-166546-0000.wav"
REPEATS = 3


class CountingBackend(OpenAIWhisperBackend):
    """The real backend, counting the requests it sends."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = 0

    async def transcribe(self, frames, sample_rate, language=None):
        self.requests += 1
        return await super().transcribe(frames, sample_rate, language)


@pytest.mark.timeout(120)
async def test_long_audio_is_transcribed_as_parallel_segments_in_order():
    with wave.open(str(CLIP), "rb") as wf:
        rate, clip = wf.getframerate(), wf.readframes(wf.getnframes())

    inner = CountingBackend(api_key=get_settings().openai_api_key)
    backend = ChunkedTranscriptionBackend(
        inner, max_segment_seconds=12, max_parallel=REPEATS, verbose=False
    )
    text = await backend.transcribe([clip * REPEATS], rate, language="en")

    assert inner.requests == REPEATS
    assert text.lower().count("breakfast") == REPEATS
