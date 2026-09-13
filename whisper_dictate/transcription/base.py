"""
Transcription Backend Interface
================================

All transcription providers must implement TranscriptionBackend.
This makes it straightforward to swap OpenAI Whisper for any other
speech-to-text service without touching the rest of the codebase.

To add a new backend:
  1. Create a new module in whisper_dictate/transcription/
  2. Subclass TranscriptionBackend and implement transcribe()
  3. If the service caps the upload size, set max_upload_bytes — long
     recordings are then split to fit (see transcription/chunked.py)
  4. Register it in whisper_dictate/__main__.py's --backend choices
"""

from abc import ABC, abstractmethod
from typing import Optional

# paInt16 = 16-bit signed integer = 2 bytes per sample. This is a constant;
# instantiating PyAudio just to query it caused a malloc crash when a second
# PyAudio context was created while the recorder's context was still alive.
INT16_SAMPLE_WIDTH = 2


class TranscriptionBackend(ABC):
    """
    Abstract base class for speech-to-text backends.

    All implementations must be safe to call concurrently from an
    asyncio event loop (i.e. transcribe() must be a coroutine).
    """

    #: Largest PCM payload, in bytes, the service accepts in one request, or
    #: None for no limit. ChunkedTranscriptionBackend splits audio to fit.
    max_upload_bytes: Optional[int] = None

    @abstractmethod
    async def transcribe(
        self,
        frames: list,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe raw PCM audio frames to text.

        Args:
            frames:      List of raw PCM byte chunks (paInt16, mono).
            sample_rate: Sample rate of the audio in Hz (e.g. 16000).
            language:    ISO 639-1 language code, or None for auto-detection.

        Returns:
            Transcribed text string. Empty string if no speech was detected.

        Raises:
            Exception: Any backend-specific error (network, auth, etc.).
                       Callers are expected to handle these gracefully.
        """
