from __future__ import annotations

import logging
import time
import uuid
from typing import List

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_app.agentic_aggregator import AgenticAggregator
from rag_app.config import ensure_dirs
from rag_app.ingest import ingest_file, save_upload
from rag_app.models import (
    AgentResponse,
    DocumentListResponse,
    HealthResponse,
    IngestJobItem,
    IngestJobResponse,
    IngestItem,
    IngestResponse,
    JobStatusResponse,
    QueryRequest,
    SourceChunk,
    SearchRequest,
    SearchResponse,
)
from rag_app.registry import list_documents
from rag_app.rag import rag_search
from rag_app.tasks import enqueue_ingest, fetch_job, get_redis_conn
from rag_app.config import settings


ensure_dirs()
aggregator = AgenticAggregator()
logger = logging.getLogger("rag_app")
handler = logging.StreamHandler()
if settings.json_logs:
    handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s"))
else:
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

app = FastAPI(
    title="Agentic RAG Aggregator",
    version="1.0.0",
    description="Portfolio-ready RAG app with an agentic aggregator workflow.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"}))

Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

if settings.otel_endpoint:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.otel_service_name})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        logger.exception("OpenTelemetry init failed")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    try:
        get_redis_conn().ping()
    except Exception:
        return HealthResponse(status="degraded")
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: List[UploadFile] = File(...),
    title: str | None = Form(None),
) -> IngestResponse:
    items: List[IngestItem] = []
    for file in files:
        content = await file.read()
        path = save_upload(file.filename, content)
        result, _ = ingest_file(path, title=title)
        items.append(
            IngestItem(
                filename=result.filename,
                doc_id=result.doc_id,
                chunks_added=result.chunks_added,
                skipped=result.skipped,
                message=result.message,
            )
        )
    return IngestResponse(items=items)


@app.post("/ingest_async", response_model=IngestJobResponse)
async def ingest_async(
    files: List[UploadFile] = File(...),
    title: str | None = Form(None),
) -> IngestJobResponse:
    items: List[IngestJobItem] = []
    for file in files:
        content = await file.read()
        path = save_upload(file.filename, content)
        job_id = enqueue_ingest(str(path), title)
        items.append(IngestJobItem(filename=path.name, job_id=job_id))
    return IngestJobResponse(items=items)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = fetch_job(job_id)
    if not job:
        return JobStatusResponse(job_id=job_id, status="not_found")
    status = job.get_status()
    result = None
    error = None
    if status == "failed":
        error = job.exc_info
    if status == "finished" and isinstance(job.result, dict):
        result = IngestItem(**job.result)
    return JobStatusResponse(job_id=job_id, status=status, result=result, error=error)


@app.post("/upload", response_model=IngestResponse)
async def upload(
    file: UploadFile = File(...),
    title: str | None = Form(None),
) -> IngestResponse:
    content = await file.read()
    path = save_upload(file.filename, content)
    result, _ = ingest_file(path, title=title)
    item = IngestItem(
        filename=result.filename,
        doc_id=result.doc_id,
        chunks_added=result.chunks_added,
        skipped=result.skipped,
        message=result.message,
    )
    return IngestResponse(items=[item])


@app.post("/query", response_model=AgentResponse)
def query(payload: QueryRequest) -> AgentResponse:
    filters = []
    if payload.source_filter:
        filters.append(_build_filter("source", payload.source_filter))
    if payload.title_filter:
        filters.append(_build_filter("title", payload.title_filter))
    if payload.doc_id_filter:
        filters.append(_build_filter("doc_id", payload.doc_id_filter))
    metadata_filter = None
    if filters:
        metadata_filter = filters[0] if len(filters) == 1 else {"$and": filters}
    return aggregator.answer(
        payload.question,
        top_k=payload.top_k,
        metadata_filter=metadata_filter,
        include_steps=payload.include_steps,
        llm_provider=payload.llm_provider,
        llm_model=payload.llm_model,
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    filters = []
    if payload.source_filter:
        filters.append(_build_filter("source", payload.source_filter))
    if payload.title_filter:
        filters.append(_build_filter("title", payload.title_filter))
    if payload.doc_id_filter:
        filters.append(_build_filter("doc_id", payload.doc_id_filter))
    metadata_filter = None
    if filters:
        metadata_filter = filters[0] if len(filters) == 1 else {"$and": filters}
    sources = rag_search(payload.query, top_k=payload.top_k, metadata_filter=metadata_filter)
    chunks = [
        SourceChunk(
            source=src,
            title=title,
            doc_id=doc_id,
            page=page,
            content=content,
        )
        for src, title, doc_id, page, content in sources
    ]
    return SearchResponse(chunks=chunks)


@app.get("/documents", response_model=DocumentListResponse)
def documents() -> DocumentListResponse:
    return DocumentListResponse(documents=list_documents())


def _build_filter(key: str, values: List[str]) -> dict:
    if len(values) == 1:
        return {key: values[0]}
    return {key: {"$in": values}}
