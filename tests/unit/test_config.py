"""
Unit tests for the Settings configuration class.

Validates default values, language normalization, and the error message
raised when OPENAI_API_KEY is missing.
"""

import pytest
from unittest.mock import patch

from whisper_dictate.config import Settings, SUPPORTED_LANGUAGES


def test_default_language_is_auto():
    with patch.dict("os.environ", {}, clear=True):
        s = Settings()
        assert s.default_language == "auto"


def test_openai_api_key_read_from_env():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}, clear=False):
        s = Settings()
        assert s.openai_api_key == "sk-test123"


def test_validate_raises_when_key_missing():
    with patch.dict("os.environ", {}, clear=True):
        s = Settings()
        s.openai_api_key = None
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            s.validate()


def test_normalize_auto_returns_none():
    s = Settings()
    assert s.normalize_language_code("auto") is None
    assert s.normalize_language_code("Auto") is None
    assert s.normalize_language_code(None) is None


def test_normalize_full_name_to_code():
    s = Settings()
    assert s.normalize_language_code("spanish") == "es"
    assert s.normalize_language_code("english") == "en"


def test_normalize_code_passthrough():
    s = Settings()
    assert s.normalize_language_code("es") == "es"
    assert s.normalize_language_code("fr") == "fr"


def test_is_language_supported():
    s = Settings()
    assert s.is_language_supported("en")
    assert s.is_language_supported("es")
    assert not s.is_language_supported("xx")
