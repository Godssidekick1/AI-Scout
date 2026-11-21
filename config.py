"""
AI Scout System — Configuration
Loads all settings from the environment / .env file.
"""

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def current_understat_season(today: date | None = None) -> int:
    """Understat labels a season by its starting year (2025 = 2025-26).
    A new season starts in August."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


class Settings(BaseSettings):
    """All configuration for the AI Scout system."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # === API keys ===
    GOOGLE_API_KEY: str = ""
    SERPER_API_KEY: str = ""

    # === LLM ===
    GEMINI_MODEL: str = "gemini-2.5-flash"          # reasoning-heavy steps (parse, find, summarise)
    GEMINI_FAST_MODEL: str = "gemini-2.5-flash-lite"  # cheap bulk steps (news sentiment)
    LLM_TIMEOUT_S: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_CACHE: bool = True                          # cache structured LLM answers by prompt hash

    # Approximate list prices in USD per million tokens (input, output), used only
    # for the cost estimate printed after a run. Update if Google changes pricing.
    LLM_PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.5-pro": (1.25, 10.00),
    }

    # === Agent loop ===
    FINDER_MAX_TOOL_STEPS: int = 8      # tool-calling turns per finder attempt
    FINDER_MAX_ATTEMPTS: int = 3        # finder <-> verifier loops
    MIN_VERIFIED_CANDIDATES: int = 5    # verifier sends the finder back below this
    MAX_CANDIDATES: int = 10            # candidates carried into checks and reports

    # === Forecast model (v3) ===
    ENABLE_FORECAST: bool = True        # use the fitted model + role search when its artifact exists
    ENABLE_VECTOR_INDEX: bool = True    # ChromaDB style index for role search (numpy fallback)

    # === Concurrency ===
    MAX_CONCURRENCY: int = 5            # simultaneous outbound requests per run

    # === Data sources ===
    UNDERSTAT_SEASON: int = Field(default_factory=current_understat_season)
    ENABLE_TRANSFERMARKT: bool = True
    ENABLE_EA_FC: bool = True
    ENABLE_NEWS_SEARCH: bool = True
    HTTP_TIMEOUT_S: float = 20.0
    HTTP_MAX_RETRIES: int = 2

    # === Cache ===
    CACHE_DIR: str = ".cache"
    CACHE_TTL_HOURS: int = 24

    # === Paths ===
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent)

    @property
    def cache_path(self) -> Path:
        path = self.BASE_DIR / self.CACHE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_serper(self) -> bool:
        return self.ENABLE_NEWS_SEARCH and bool(self.SERPER_API_KEY)

    def has_gemini(self) -> bool:
        return bool(self.GOOGLE_API_KEY)


# Global settings instance
settings = Settings()
