"""
Unit tests for whisper_dictate.transcription.

Covers:
- TranscriptionBackend ABC cannot be instantiated
- OpenAIWhisperBackend: WAV file is valid, API is called with correct params
- OpenAIWhisperBackend: language forwarding (present vs absent)
- OpenAIWhisperBackend: response is always returned as a stripped string
- OpenAIWhisperBackend: temp file is deleted after transcription

The OpenAI client is always mocked — no real network calls are made.
"""

import os
import wave
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from whisper_dictate.transcription.base import TranscriptionBackend
from whisper_dictate.transcription.openai_backend import OpenAIWhisperBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SILENCE_CHUNK = b"\x00\x00" * 512   # 512 int16 samples = 1024 bytes
ONE_SECOND_FRAMES = [SILENCE_CHUNK] * 16  # ~1s at 16 kHz


def _make_backend(mock_client: MagicMock) -> OpenAIWhisperBackend:
    with patch("whisper_dictate.transcription.openai_backend.AsyncOpenAI", return_value=mock_client):
        return OpenAIWhisperBackend(api_key="sk-test")


def _mock_client(return_value: str = "Hello") -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(return_value=return_value)
    return client


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class TestTranscriptionBackendInterface:
    """TranscriptionBackend enforces the ABC contract."""

    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            TranscriptionBackend()

    def test_concrete_subclass_must_implement_transcribe(self):
        class Incomplete(TranscriptionBackend):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_with_transcribe_is_valid(self):
        class Minimal(TranscriptionBackend):
            async def transcribe(self, frames, sample_rate, language=None):
                return ""
        assert isinstance(Minimal(), TranscriptionBackend)


# ---------------------------------------------------------------------------
# OpenAIWhisperBackend — WAV generation
# ---------------------------------------------------------------------------


class TestOpenAIWhisperBackendWavGeneration:
    """_write_temp_wav() produces a well-formed WAV file."""

    def test_wav_file_is_created(self):
        client = _mock_client()
        backend = _make_backend(client)
        path = backend._write_temp_wav(ONE_SECOND_FRAMES, sample_rate=16000)
        try:
            assert os.path.isfile(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_wav_file_has_correct_channels(self):
        backend = _make_backend(_mock_client())
        path = backend._write_temp_wav(ONE_SECOND_FRAMES, sample_rate=16000)
        try:
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
        finally:
            os.unlink(path)

    def test_wav_file_has_correct_sample_rate(self):
        backend = _make_backend(_mock_client())
        path = backend._write_temp_wav(ONE_SECOND_FRAMES, sample_rate=16000)
        try:
            with wave.open(path, "rb") as wf:
                assert wf.getframerate() == 16000
        finally:
            os.unlink(path)

    def test_wav_file_contains_audio_frames(self):
        backend = _make_backend(_mock_client())
        path = backend._write_temp_wav(ONE_SECOND_FRAMES, sample_rate=16000)
        try:
            with wave.open(path, "rb") as wf:
                assert wf.getnframes() > 0
        finally:
            os.unlink(path)

    def test_wav_file_is_deleted_after_transcription(self):
        """No temp files are leaked after transcribe() completes."""
        captured_paths = []
        original = OpenAIWhisperBackend._write_temp_wav

        def capturing_write(self, frames, sample_rate):
            path = original(self, frames, sample_rate)
            captured_paths.append(path)
            return path

        client = _mock_client()
        with patch("whisper_dictate.transcription.openai_backend.AsyncOpenAI", return_value=client), \
             patch.object(OpenAIWhisperBackend, "_write_temp_wav", capturing_write):
            backend = OpenAIWhisperBackend(api_key="sk-test")
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
            )

        assert len(captured_paths) == 1
        assert not os.path.exists(captured_paths[0]), "Temp WAV file was not deleted"


# ---------------------------------------------------------------------------
# OpenAIWhisperBackend — API call parameters
# ---------------------------------------------------------------------------


class TestOpenAIWhisperBackendApiCall:
    """transcribe() calls the OpenAI API with the right parameters."""

    @pytest.mark.asyncio
    async def test_calls_api_exactly_once(self):
        client = _mock_client("Hello")
        backend = _make_backend(client)
        await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
        client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_correct_model(self):
        client = _mock_client()
        backend = _make_backend(client)
        await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["model"] == OpenAIWhisperBackend.MODEL

    @pytest.mark.asyncio
    async def test_uses_text_response_format(self):
        client = _mock_client()
        backend = _make_backend(client)
        await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["response_format"] == "text"

    @pytest.mark.asyncio
    async def test_language_omitted_when_none(self):
        """When language=None, the key is absent so Whisper auto-detects."""
        client = _mock_client()
        backend = _make_backend(client)
        await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000, language=None)
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lang", ["es", "en", "fr", "de", "pt"])
    async def test_language_forwarded_when_set(self, lang):
        client = _mock_client("text")
        backend = _make_backend(client)
        await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000, language=lang)
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["language"] == lang


# ---------------------------------------------------------------------------
# OpenAIWhisperBackend — return value handling
# ---------------------------------------------------------------------------


class TestOpenAIWhisperBackendReturnValue:
    """transcribe() always returns a clean, stripped string."""

    @pytest.mark.asyncio
    async def test_returns_stripped_text(self):
        client = _mock_client("  Hello world  ")
        backend = _make_backend(client)
        result = await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_speech(self):
        client = _mock_client("")
        backend = _make_backend(client)
        result = await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_string_type(self):
        client = _mock_client("Hola")
        backend = _make_backend(client)
        result = await backend.transcribe(ONE_SECOND_FRAMES, sample_rate=16000)
        assert isinstance(result, str)
