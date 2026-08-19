"""Конфигурация LLM-слоя.

Единственное место, где читается окружение. Всё остальное получает готовый
объект `LLMSettings` — так конфиг тестируется подстановкой, а не патчем os.environ.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["openai", "ollama"]


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LLM_", extra="ignore"
    )

    profile: Profile = "openai"
    base_url: str
    model: str
    api_key: str = "EMPTY"

    temperature: float = 0.0
    seed: int = 42

    # Ollama обрезает промпт до num_ctx молча. Держим значение явным,
    # чтобы обрезка few-shot примеров не превратилась в невидимую деградацию.
    num_ctx: int = 16384

    @model_validator(mode="after")
    def _check_profile_consistency(self) -> "LLMSettings":
        if self.profile == "openai" and self.api_key in ("", "EMPTY"):
            raise ValueError("Профиль 'openai' требует непустой LLM_API_KEY")
        if self.profile == "ollama" and "/v1" in self.base_url:
            raise ValueError(
                "Для профиля 'ollama' укажи базовый URL без /v1 "
                "(например http://localhost:11434) — используется нативный API "
                "ради надёжного structured output, см. ADR-002"
            )
        return self


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Кэшируется: настройки читаются один раз за процесс."""
    return LLMSettings()  # type: ignore[call-arg]
