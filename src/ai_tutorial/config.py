"""
Configuration management for AI Engineering Tutorial.

Uses pydantic-settings for environment variable management and validation.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    APP_NAME: str = "AI Engineering Tutorial"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    BASE_URL: str = "http://localhost:8080"  # Override in production

    # API Keys (optional - users provide their own)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    XAI_API_KEY: Optional[str] = None  # Grok

    # Database
    DATABASE_URL: str = "sqlite:///./ai_tutorial.db"

    # Security
    # Generate a secure key with: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str = "change-this-in-production-use-a-real-secret-key"

    def validate_secret_key(self) -> None:
        """Validate SECRET_KEY is not the default value in production."""
        if not self.DEBUG and self.SECRET_KEY == "change-this-in-production-use-a-real-secret-key":
            raise ValueError(
                "SECRET_KEY must be changed in production! "
                "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Cache settings
    ENABLE_RESPONSE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
