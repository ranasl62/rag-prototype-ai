# Agentic RAG Aggregator

Portfolio-ready RAG app with a full workflow: ingest documents, retrieve context,
draft an answer, critique it, and finalize the response.

## Features
- Document ingestion (PDF, TXT, MD)
- Vector search with Chroma
- Agentic aggregator (plan → retrieve/draft → critique → finalize)
- FastAPI service with simple endpoints
- Docker-ready
- Prometheus metrics and request logging

## Quick Start (Local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Start a worker (async ingestion)
```bash
python -m rag_app.worker
```

## Quick Start (Docker)
```bash
docker compose up --build
```

## Production Docker Notes
- All services run inside Docker: API, worker, Redis, and Chroma.
- Health checks: `/ready` for the API and `redis-cli ping` for Redis.
- Use `.env` to configure providers and runtime settings.
- Rate limiting, JSON logging, and optional OpenTelemetry are enabled via env.

## Configuration
- Copy `.env` and fill in the provider API keys you want to use.
- Set `RAG_LLM_PROVIDER` to force a provider, or keep `auto`.
- Data is stored under `data/` by default (uploads, registry).
- When using Docker, Chroma runs as its own service and persists in `chroma-data`.
- For async ingestion, set `REDIS_URL` and use `/ingest_async`.

## Environment Variables
- `OPENAI_API_KEY`: enables OpenAI chat model
- `OPENAI_MODEL`: default `gpt-4o-mini`
- `ANTHROPIC_API_KEY`: enables Anthropic models
- `ANTHROPIC_MODEL`: default `claude-3-5-sonnet-latest`
- `GROQ_API_KEY`: enables Groq models
- `GROQ_MODEL`: default `llama-3.1-70b-versatile`
- `GOOGLE_API_KEY`: enables Gemini models
- `GEMINI_MODEL`: default `gemini-1.5-pro`
- `MISTRAL_API_KEY`: enables Mistral models
- `MISTRAL_MODEL`: default `mistral-large-latest`
- `RAG_LLM_PROVIDER`: `auto`, `openai`, `anthropic`, `groq`, `gemini`, `mistral`, `ollama`, or `mock`
- `USE_OLLAMA=1`: use Ollama instead of OpenAI
- `OLLAMA_BASE_URL`: optional Ollama server URL
- `OLLAMA_MODEL`: default `llama3.1`
- `USE_MOCK_LLM=1`: disable LLM calls (returns mock responses)
- `REDIS_URL`: Redis connection for async ingestion jobs
- `RAG_CHROMA_HTTP`: set `1` to use the Chroma HTTP server
- `RAG_CHROMA_HOST`: default `chroma`
- `RAG_CHROMA_PORT`: default `8000`
- `RAG_RATE_LIMIT`: default `60/minute`
- `RAG_JSON_LOGS`: `1` for JSON logs, `0` for plain text
- `OTEL_EXPORTER_OTLP_ENDPOINT`: optional OpenTelemetry collector URL
- `OTEL_SERVICE_NAME`: service name for tracing

### Provider resolver
The app uses a resolver pattern to choose the LLM provider:
- `RAG_LLM_PROVIDER=auto` selects the first available provider in this order:
  `openai → anthropic → groq → gemini → mistral → ollama`
- You can override per request using `llm_provider` and `llm_model`.

## API Usage
### Ingest
```bash
curl -F "files=@/path/to/book1.pdf" \
  -F "files=@/path/to/book2.pdf" \
  http://localhost:8000/ingest
```

### Ingest Async (recommended for large books)
```bash
curl -F "files=@/path/to/book1.pdf" \
  -F "files=@/path/to/book2.pdf" \
  http://localhost:8000/ingest_async
```

### Job Status
```bash
curl http://localhost:8000/jobs/<job_id>
```

### Upload (single file)
```bash
curl -F "file=@/path/to/book.pdf" http://localhost:8000/upload
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Find the topic on safety requirements"}'
```

### Query with specific model
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find the topic on safety requirements",
    "llm_provider": "anthropic",
    "llm_model": "claude-3-5-sonnet-latest"
  }'
```

### Query with Filters (specific books)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find the topic on safety requirements",
    "title_filter": ["Book Title A", "Book Title B"]
  }'
```

### Search (vector-only, no LLM)
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "safety requirements", "top_k": 5}'
```

### Compare Two Books (no storage)
```bash
curl -F "file_a=@/path/to/bookA.pdf" \
  -F "file_b=@/path/to/bookB.pdf" \
  -F "question=Compare the safety chapter" \
  -F "scope=chapter 5" \
  http://localhost:8000/compare
```

### List Documents
```bash
curl http://localhost:8000/documents
```

### Health
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
curl http://localhost:8000/metrics
```
# rag-prototype-ai
