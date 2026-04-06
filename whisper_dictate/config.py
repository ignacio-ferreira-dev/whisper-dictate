"""
Configuration module for whisper_dictate.

Loads settings from environment variables and/or a .env file.
The .env file is never committed to version control - see .env.example.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load .env from the project root (one level up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

SUPPORTED_LANGUAGES: dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
}


class Settings:
    """
    Application-wide configuration.

    Values are resolved in this priority order:
      1. Real environment variables (e.g. exported in shell)
      2. Variables defined in the .env file at the project root
      3. Hard-coded defaults below
    """

    def __init__(self) -> None:
        # --- Transcription ---
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "whisper-1")
        self.transcription_backend: str = os.getenv("TRANSCRIPTION_BACKEND", "openai")

        # --- Language ---
        self.default_language: str = os.getenv("DEFAULT_LANGUAGE", "auto")
        self.enable_translation: bool = (
            os.getenv("ENABLE_TRANSLATION", "false").lower() == "true"
        )

        # --- Audio ---
        self.sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))

        # --- UI / Alerts ---
        self.alert_volume: float = float(os.getenv("ALERT_VOLUME", "0.8"))
        self.alerts_enabled: bool = (
            os.getenv("ALERTS_ENABLED", "true").lower() == "true"
        )

        # --- Server (legacy FastAPI server) ---
        self.host: str = os.getenv("HOST", "localhost")
        self.port: int = int(os.getenv("PORT", "8000"))

    def validate(self) -> None:
        """
        Raise a clear RuntimeError if required settings are missing.

        Preferred over returning bool so callers don't need to check
        the return value explicitly.
        """
        if not self.openai_api_key:
            raise RuntimeError(
                "\n"
                "OPENAI_API_KEY is not set.\n"
                "Please create a .env file at the project root with:\n"
                "  OPENAI_API_KEY=sk-...\n"
                "See .env.example for the full list of available settings."
            )

    def get_openai_api_key(self) -> Optional[str]:
        """Return the OpenAI API key, or None if not configured."""
        return self.openai_api_key

    def normalize_language_code(self, language: Optional[str]) -> Optional[str]:
        """
        Convert a language name or code to the ISO 639-1 code expected by
        the Whisper API. Returns None to enable Whisper's auto-detection.
        """
        if not language or language.lower() in ("auto", "detect", "automatic"):
            return None

        name_to_code = {
            "spanish": "es",
            "english": "en",
            "french": "fr",
            "german": "de",
            "portuguese": "pt",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko",
        }
        normalized = name_to_code.get(language.lower(), language.lower())
        return normalized if normalized in SUPPORTED_LANGUAGES else None

    def is_language_supported(self, language_code: str) -> bool:
        """Return True if the language code is known to Whisper."""
        return language_code in SUPPORTED_LANGUAGES


# Module-level singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
