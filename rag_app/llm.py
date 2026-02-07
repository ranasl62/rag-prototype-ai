from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.config import settings


def _mock_response(prompt: str) -> str:
    return (
        "Mock response (no LLM configured). "
        "Set OPENAI_API_KEY or USE_OLLAMA=1 for real answers. "
        f"Prompt excerpt: {prompt[:200]}"
    )


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    is_available: Callable[[], bool]
    build: Callable[[Optional[str]], object]


def _build_openai(model_name: Optional[str]):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name or settings.openai_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )


def _build_anthropic(model_name: Optional[str]):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model_name or settings.anthropic_model,
        temperature=0.2,
        api_key=settings.anthropic_api_key,
    )


def _build_groq(model_name: Optional[str]):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model_name or settings.groq_model,
        temperature=0.2,
        api_key=settings.groq_api_key,
    )


def _build_gemini(model_name: Optional[str]):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name or settings.gemini_model,
        temperature=0.2,
        google_api_key=settings.google_api_key,
    )


def _build_mistral(model_name: Optional[str]):
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=model_name or settings.mistral_model,
        temperature=0.2,
        api_key=settings.mistral_api_key,
    )


def _build_ollama(model_name: Optional[str]):
    from langchain_community.chat_models import ChatOllama

    return ChatOllama(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )


PROVIDERS = [
    ProviderSpec("openai", lambda: bool(settings.openai_api_key), _build_openai),
    ProviderSpec("anthropic", lambda: bool(settings.anthropic_api_key), _build_anthropic),
    ProviderSpec("groq", lambda: bool(settings.groq_api_key), _build_groq),
    ProviderSpec("gemini", lambda: bool(settings.google_api_key), _build_gemini),
    ProviderSpec("mistral", lambda: bool(settings.mistral_api_key), _build_mistral),
    ProviderSpec("ollama", lambda: settings.force_ollama or bool(settings.ollama_base_url), _build_ollama),
]


def _resolve_provider(name: str) -> Optional[ProviderSpec]:
    name = name.lower()
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None


def _resolve_auto() -> Optional[ProviderSpec]:
    for spec in PROVIDERS:
        if spec.is_available():
            return spec
    return None


def get_chat_model(provider: Optional[str] = None, model_name: Optional[str] = None):
    if settings.force_mock_llm or (provider and provider.lower() == "mock"):
        return None

    selected = (provider or settings.llm_provider).lower()
    spec = _resolve_auto() if selected == "auto" else _resolve_provider(selected)
    if not spec or not spec.is_available():
        return None
    try:
        return spec.build(model_name)
    except Exception:
        return None


def chat(
    system_prompt: str,
    user_prompt: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    model = get_chat_model(provider=provider, model_name=model_name)
    if model is None:
        return _mock_response(user_prompt)
    messages: List = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    result = model.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)
