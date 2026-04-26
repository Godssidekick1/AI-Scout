"""
AI Scout System — Configuration
Loads all settings from .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All configuration for the AI Scout system."""

    # === API Keys (all free tiers) ===
    GOOGLE_API_KEY: str = ""
    API_FOOTBALL_KEY: str = ""
    SERPER_API_KEY: str = ""

    # === LLM Settings ===
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # === Feature Flags ===
    ENABLE_API_FOOTBALL: bool = True
    ENABLE_UNDERSTAT: bool = True
    ENABLE_TRANSFERMARKT: bool = True
    ENABLE_EA_FC: bool = True
    ENABLE_NEWS_SEARCH: bool = True

    # === Cache ===
    CACHE_DIR: str = ".cache"
    CACHE_TTL_HOURS: int = 24

    # === Paths ===
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def cache_path(self) -> Path:
        path = self.BASE_DIR / self.CACHE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_api_football(self) -> bool:
        return self.ENABLE_API_FOOTBALL and bool(self.API_FOOTBALL_KEY)

    def has_serper(self) -> bool:
        return self.ENABLE_NEWS_SEARCH and bool(self.SERPER_API_KEY)

    def has_gemini(self) -> bool:
        return bool(self.GOOGLE_API_KEY)


# Global settings instance
settings = Settings()
