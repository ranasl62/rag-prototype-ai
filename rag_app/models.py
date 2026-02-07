from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestItem(BaseModel):
    filename: str
    doc_id: str
    chunks_added: int
    skipped: bool
    message: str


class IngestResponse(BaseModel):
    items: List[IngestItem]


class IngestJobItem(BaseModel):
    filename: str
    job_id: str


class IngestJobResponse(BaseModel):
    items: List[IngestJobItem]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[IngestItem] = None
    error: Optional[str] = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    title: str
    sha256: str
    chunks: int
    ingested_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    include_steps: bool = True
    source_filter: Optional[List[str]] = None
    title_filter: Optional[List[str]] = None
    doc_id_filter: Optional[List[str]] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    source_filter: Optional[List[str]] = None
    title_filter: Optional[List[str]] = None
    doc_id_filter: Optional[List[str]] = None


class SourceChunk(BaseModel):
    source: str
    title: str
    doc_id: str
    page: Optional[int] = None
    content: str


class SearchResponse(BaseModel):
    chunks: List[SourceChunk]


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1)
    scope: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class CompareResponse(BaseModel):
    answer: str
    book_a: List[SourceChunk]
    book_b: List[SourceChunk]


class AgentStep(BaseModel):
    name: str
    detail: str


class AgentResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    steps: List[AgentStep]


class HealthResponse(BaseModel):
    status: str
