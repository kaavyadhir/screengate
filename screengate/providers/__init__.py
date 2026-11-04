from __future__ import annotations

from ..config import Settings
from .base import DEFAULT_JOB_DESCRIPTION, SYSTEM_PROMPT, Provider
from .mock import MockProvider

__all__ = [
    "Provider",
    "MockProvider",
    "SYSTEM_PROMPT",
    "DEFAULT_JOB_DESCRIPTION",
    "build_provider",
]


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key, model=settings.gemini_model
        )
    raise ValueError(f"unknown provider: {settings.provider}")
