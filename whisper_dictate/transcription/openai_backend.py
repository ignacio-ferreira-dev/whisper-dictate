"""
OpenAI Whisper Backend
======================

Implements TranscriptionBackend using the OpenAI Whisper API.
Audio frames are encoded as a WAV file in memory and sent to
the whisper-1 model via the openai Python SDK.
"""

import os
import tempfile
import wave
from typing import Optional

from openai import AsyncOpenAI

from whisper_dictate.transcription.base import INT16_SAMPLE_WIDTH, TranscriptionBackend


class OpenAIWhisperBackend(TranscriptionBackend):
    """
    Transcription backend that calls the OpenAI Whisper API.

    A temporary WAV file is created for each transcription request and
    deleted immediately after the API call completes.
    """

    MODEL = "whisper-1"

    #: The API documents a 25 MB upload limit and answers larger request
    #: bodies with HTTP 413 ("Maximum content size limit (26214400)"). Staying
    #: at 24 000 000 bytes of PCM leaves room under either reading for the WAV
    #: header and the multipart envelope.
    max_upload_bytes = 24_000_000

    def __init__(self, api_key: str, model: str = MODEL):
        """
        Args:
            api_key: OpenAI API key.
            model:   Whisper model identifier (default: 'whisper-1').
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def transcribe(
        self,
        frames: list,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> str:
        """
        Send PCM frames to OpenAI Whisper and return the transcript.

        Args:
            frames:      List of raw PCM byte chunks (paInt16, mono).
            sample_rate: Sample rate in Hz.
            language:    ISO 639-1 code or None for auto-detection.

        Returns:
            Transcribed text string (stripped). Empty if no speech detected.
        """
        tmp_path = self._write_temp_wav(frames, sample_rate)
        try:
            return await self._call_api(tmp_path, language)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_temp_wav(self, frames: list, sample_rate: int) -> str:
        """
        Write PCM frames to a temporary WAV file and return the file path.

        Args:
            frames:      List of raw PCM byte chunks (paInt16, mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Path to the temporary WAV file.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        with wave.open(tmp_path, "wb") as wf:  # pylint: disable=no-member
            wf.setnchannels(1)
            wf.setsampwidth(INT16_SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

        return tmp_path

    async def _call_api(self, wav_path: str, language: Optional[str]) -> str:
        """
        Open the WAV file and call the OpenAI transcription endpoint.

        Args:
            wav_path: Path to the WAV file.
            language: ISO 639-1 code or None.

        Returns:
            Raw transcription string from the API.
        """
        with open(wav_path, "rb") as audio_file:
            kwargs = {
                "model": self._model,
                "file": audio_file,
                "response_format": "text",
            }
            if language:
                kwargs["language"] = language

            result = await self._client.audio.transcriptions.create(**kwargs)

        return result.strip() if isinstance(result, str) else str(result).strip()
