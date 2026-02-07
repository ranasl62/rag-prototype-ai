from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_app.config import ensure_dirs, settings
from rag_app.registry import find_by_hash, register_document
from rag_app.vectorstore import add_documents


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class IngestResult:
    filename: str
    doc_id: str
    chunks_added: int
    skipped: bool
    message: str


def _get_loader(path: Path):
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path))
    if path.suffix.lower() in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _pdf_title(path: Path) -> str | None:
    try:
        reader = PdfReader(str(path))
        title = reader.metadata.title if reader.metadata else None
        if title:
            return str(title).strip()
    except Exception:
        return None
    return None


def ingest_file(path: Path, title: str | None = None) -> Tuple[IngestResult, List[Document]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    file_bytes = path.read_bytes()
    file_hash = _sha256(file_bytes)
    existing = find_by_hash(file_hash)
    if existing:
        return (
            IngestResult(
                filename=path.name,
                doc_id=existing["doc_id"],
                chunks_added=0,
                skipped=True,
                message="Already ingested; skipped.",
            ),
            [],
        )

    loader = _get_loader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    doc_id = file_hash[:16]
    if title:
        doc_title = title
    elif path.suffix.lower() == ".pdf":
        doc_title = _pdf_title(path) or path.stem
    else:
        doc_title = path.stem
    ids = []
    for idx, chunk in enumerate(chunks):
        chunk.metadata["source"] = path.name
        chunk.metadata["title"] = doc_title
        chunk.metadata["doc_id"] = doc_id
        ids.append(f"{doc_id}:{idx}")

    added = add_documents(chunks, ids=ids)
    register_document(
        {
            "doc_id": doc_id,
            "filename": path.name,
            "title": doc_title,
            "sha256": file_hash,
            "chunks": added,
        }
    )
    return (
        IngestResult(
            filename=path.name,
            doc_id=doc_id,
            chunks_added=added,
            skipped=False,
            message="Ingestion complete.",
        ),
        chunks,
    )


def save_upload(filename: str, content: bytes) -> Path:
    ensure_dirs()
    safe_name = filename.replace("/", "_").replace("\\", "_")
    file_hash = _sha256(content)[:12]
    path = settings.upload_dir / f"{file_hash}_{safe_name}"
    path.write_bytes(content)
    return path
