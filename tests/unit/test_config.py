"""
Unit tests for whisper_dictate.config.Settings.

All tests are fully isolated: environment variables are patched so the
host machine's shell environment never influences the results.
"""

import pytest
from unittest.mock import patch

from whisper_dictate.config import Settings, SUPPORTED_LANGUAGES

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_with_env(**env_vars) -> Settings:
    """Return a Settings instance built from an explicit environment dict."""
    with patch.dict("os.environ", env_vars, clear=True):
        return Settings()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """Settings resolves sensible defaults when no environment is configured."""

    def test_default_language_is_auto(self):
        s = _settings_with_env()
        assert s.default_language == "auto"

    def test_default_model_is_whisper_1(self):
        s = _settings_with_env()
        assert s.whisper_model == "whisper-1"

    def test_default_backend_is_openai(self):
        s = _settings_with_env()
        assert s.transcription_backend == "openai"

    def test_default_translation_is_disabled(self):
        s = _settings_with_env()
        assert s.enable_translation is False

    def test_default_alerts_enabled(self):
        s = _settings_with_env()
        assert s.alerts_enabled is True

    def test_api_key_is_none_when_not_set(self):
        s = _settings_with_env()
        assert s.openai_api_key is None


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestSettingsEnvironmentOverrides:
    """Settings reads every recognised environment variable correctly."""

    def test_reads_openai_api_key(self):
        s = _settings_with_env(OPENAI_API_KEY="sk-test123")
        assert s.openai_api_key == "sk-test123"

    def test_reads_whisper_model(self):
        s = _settings_with_env(WHISPER_MODEL="whisper-2")
        assert s.whisper_model == "whisper-2"

    def test_reads_default_language(self):
        s = _settings_with_env(DEFAULT_LANGUAGE="es")
        assert s.default_language == "es"

    def test_enable_translation_true(self):
        s = _settings_with_env(ENABLE_TRANSLATION="true")
        assert s.enable_translation is True

    def test_enable_translation_false(self):
        s = _settings_with_env(ENABLE_TRANSLATION="false")
        assert s.enable_translation is False

    def test_alerts_can_be_disabled_via_env(self):
        s = _settings_with_env(ALERTS_ENABLED="false")
        assert s.alerts_enabled is False


# ---------------------------------------------------------------------------
# Long recordings
# ---------------------------------------------------------------------------


LONG_RECORDING_SETTINGS = [
    ("MAX_RECORDING_MINUTES", "max_recording_minutes", 30),
    ("TRANSCRIPTION_CHUNK_SECONDS", "transcription_chunk_seconds", 600),
    ("TRANSCRIPTION_MAX_PARALLEL", "transcription_max_parallel", 4),
]


class TestLongRecordingSettings:
    """The recording cap and the split/parallel knobs are read from the env and sanity-checked."""

    @pytest.mark.parametrize("env_name,attribute,default", LONG_RECORDING_SETTINGS)
    def test_default(self, env_name, attribute, default):
        assert getattr(_settings_with_env(), attribute) == default

    @pytest.mark.parametrize("env_name,attribute,default", LONG_RECORDING_SETTINGS)
    def test_env_overrides_the_default(self, env_name, attribute, default):
        assert getattr(_settings_with_env(**{env_name: "7"}), attribute) == 7

    def test_recording_cap_is_exposed_in_seconds(self):
        assert _settings_with_env(MAX_RECORDING_MINUTES="1.5").max_recording_seconds == 90

    @pytest.mark.parametrize("value", ["0", "-3", "abc", "nan", "inf"])
    @pytest.mark.parametrize("env_name,attribute,default", LONG_RECORDING_SETTINGS)
    def test_invalid_values_fall_back_to_the_default(
        self, capsys, env_name, attribute, default, value
    ):
        """Zero parallel requests would hang; zero minutes would stop at once."""
        assert getattr(_settings_with_env(**{env_name: value}), attribute) == default
        assert env_name in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Settings.validate() raises a descriptive error when required values are absent."""

    def test_validate_raises_when_api_key_missing(self):
        s = _settings_with_env()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            s.validate()

    def test_validate_raises_with_helpful_env_file_hint(self):
        s = _settings_with_env()
        with pytest.raises(RuntimeError, match=".env"):
            s.validate()

    def test_validate_passes_when_api_key_present(self):
        s = _settings_with_env(OPENAI_API_KEY="sk-valid")
        s.validate()  # must not raise


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------


class TestLanguageNormalisation:
    """Settings.normalize_language_code() maps user input to ISO 639-1 codes."""

    @pytest.mark.parametrize("value", ["auto", "Auto", "AUTO", "detect", "automatic", None])
    def test_auto_variants_return_none(self, value):
        s = _settings_with_env()
        assert s.normalize_language_code(value) is None

    @pytest.mark.parametrize("name,expected_code", [
        ("spanish", "es"),
        ("english", "en"),
        ("french", "fr"),
        ("german", "de"),
        ("portuguese", "pt"),
    ])
    def test_full_language_names_map_to_codes(self, name, expected_code):
        s = _settings_with_env()
        assert s.normalize_language_code(name) == expected_code

    @pytest.mark.parametrize("code", ["es", "en", "fr", "de", "pt", "ja", "zh"])
    def test_valid_codes_pass_through_unchanged(self, code):
        s = _settings_with_env()
        assert s.normalize_language_code(code) == code

    def test_unknown_code_returns_none(self):
        s = _settings_with_env()
        assert s.normalize_language_code("xx") is None


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------


class TestSupportedLanguages:
    """SUPPORTED_LANGUAGES dictionary is complete and consistent."""

    def test_auto_is_included(self):
        assert "auto" in SUPPORTED_LANGUAGES

    @pytest.mark.parametrize("code", ["en", "es", "fr", "de", "pt", "it", "ru", "ja", "ko", "zh"])
    def test_common_codes_are_supported(self, code):
        s = _settings_with_env()
        assert s.is_language_supported(code)

    def test_unknown_code_is_not_supported(self):
        s = _settings_with_env()
        assert not s.is_language_supported("xx")
