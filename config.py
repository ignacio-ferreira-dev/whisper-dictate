"""Configuration module for Whisper Flow"""

import os
from typing import Optional, List


# Supported languages by OpenAI Whisper (most common ones)
SUPPORTED_LANGUAGES = {
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
    "fi": "Finnish"
}


def load_env_file(file_path: str = ".env") -> dict:
    """
    Load environment variables from a file
    
    Args:
        file_path: Path to the environment file
        
    Returns:
        Dictionary with environment variables
    """
    env_vars = {}
    
    if not os.path.exists(file_path):
        return env_vars
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"Warning: Could not load config file {file_path}: {e}")
    
    return env_vars


class Config:
    """Configuration class for Whisper Flow"""
    
    def __init__(self, config_file: str = ".env"):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file
        """
        # Load from config file first
        file_vars = load_env_file(config_file)
        
        # OpenAI Configuration
        self.openai_api_key = (
            os.getenv("OPENAI_API_KEY") or 
            file_vars.get("OPENAI_API_KEY")
        )
        
        self.whisper_model = (
            os.getenv("WHISPER_MODEL") or 
            file_vars.get("WHISPER_MODEL") or 
            "whisper-1"
        )
        
        # Audio Configuration
        self.sample_rate = int(
            os.getenv("SAMPLE_RATE") or 
            file_vars.get("SAMPLE_RATE") or 
            "16000"
        )
        
        self.chunk_size = int(
            os.getenv("CHUNK_SIZE") or 
            file_vars.get("CHUNK_SIZE") or 
            "1024"
        )
        
        # Language Configuration
        self.default_language = (
            os.getenv("DEFAULT_LANGUAGE") or 
            file_vars.get("DEFAULT_LANGUAGE") or 
            "auto"  # auto-detect by default
        )
        
        self.enable_translation = (
            os.getenv("ENABLE_TRANSLATION", "false").lower() == "true" or
            file_vars.get("ENABLE_TRANSLATION", "false").lower() == "true"
        )
        
        # Server Configuration
        self.host = (
            os.getenv("HOST") or 
            file_vars.get("HOST") or 
            "localhost"
        )
        
        self.port = int(
            os.getenv("PORT") or 
            file_vars.get("PORT") or 
            "8000"
        )
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
        """
        if not self.openai_api_key:
            print("ERROR: OPENAI_API_KEY is required!")
            print("Please set it in .env file or as environment variable")
            return False
        
        return True
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return self.openai_api_key
    
    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages"""
        return SUPPORTED_LANGUAGES
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language code is supported
        
        Args:
            language_code: Language code to check (e.g., 'en', 'es', 'auto')
            
        Returns:
            True if language is supported
        """
        return language_code in SUPPORTED_LANGUAGES
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get human-readable language name
        
        Args:
            language_code: Language code (e.g., 'en', 'es')
            
        Returns:
            Human-readable language name
        """
        return SUPPORTED_LANGUAGES.get(language_code, f"Unknown ({language_code})")
    
    def normalize_language_code(self, language: Optional[str]) -> Optional[str]:
        """
        Normalize language code for OpenAI API
        
        Args:
            language: Language code or 'auto'
            
        Returns:
            Normalized language code or None for auto-detection
        """
        if not language or language.lower() in ["auto", "detect", "automatic"]:
            return None  # Let OpenAI auto-detect
        
        # Normalize common variations
        lang_map = {
            "spanish": "es",
            "english": "en", 
            "french": "fr",
            "german": "de",
            "portuguese": "pt",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko"
        }
        
        normalized = lang_map.get(language.lower(), language.lower())
        
        # Return only if supported, otherwise None for auto-detection
        return normalized if self.is_language_supported(normalized) else None


# Global configuration instance
_config = None


def get_config(config_file: str = ".env") -> Config:
    """
    Get global configuration instance
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config
