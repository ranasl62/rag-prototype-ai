from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    chroma_dir: Path = Path(os.getenv("RAG_CHROMA_DIR", "data/chroma"))
    chroma_http: bool = _env_bool("RAG_CHROMA_HTTP", False)
    chroma_host: str = os.getenv("RAG_CHROMA_HOST", "chroma")
    chroma_port: int = int(os.getenv("RAG_CHROMA_PORT", "8000"))
    upload_dir: Path = Path(os.getenv("RAG_UPLOAD_DIR", "data/uploads"))
    registry_file: Path = Path(os.getenv("RAG_REGISTRY_FILE", "data/documents.jsonl"))
    embeddings_model: str = os.getenv(
        "RAG_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "auto")
    async_ingest: bool = _env_bool("RAG_ASYNC_INGEST", False)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rate_limit: str = os.getenv("RAG_RATE_LIMIT", "60/minute")
    json_logs: bool = _env_bool("RAG_JSON_LOGS", True)
    otel_endpoint: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "rag-app")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url: str | None = os.getenv("OLLAMA_BASE_URL")
    force_ollama: bool = _env_bool("USE_OLLAMA", False)
    force_mock_llm: bool = _env_bool("USE_MOCK_LLM", False)

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def groq_api_key(self) -> str | None:
        return os.getenv("GROQ_API_KEY")

    @property
    def google_api_key(self) -> str | None:
        return os.getenv("GOOGLE_API_KEY")

    @property
    def mistral_api_key(self) -> str | None:
        return os.getenv("MISTRAL_API_KEY")


settings = Settings()


def ensure_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.registry_file.parent.mkdir(parents=True, exist_ok=True)
