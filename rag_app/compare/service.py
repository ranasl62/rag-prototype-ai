from __future__ import annotations

import uuid
from io import BytesIO
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_app.llm import chat
from rag_app.vectorstore import get_embeddings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _documents_from_upload(filename: str, content: bytes) -> List[Document]:
    suffix = filename.lower().split(".")[-1]
    if f".{suffix}" not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: .{suffix}")
    if suffix == "pdf":
        reader = PdfReader(BytesIO(content))
        docs: List[Document] = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": idx + 1},
                )
            )
        return docs
    text = content.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": filename})]


def _build_vectorstore(collection_name: str):
    client = chromadb.Client(
        ChromaSettings(is_persistent=False, anonymized_telemetry=False)
    )
    return client, collection_name


def _search_chunks(
    docs: List[Document], query: str, top_k: int, label: str
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["title"] = label
        chunk.metadata["doc_id"] = label

    client, collection = _build_vectorstore(f"compare-{uuid.uuid4().hex}")
    # Use LangChain's Chroma wrapper to apply configured embeddings.
    from langchain_community.vectorstores import Chroma

    store = Chroma(
        collection_name=collection,
        client=client,
        embedding_function=get_embeddings(),
    )
    store.add_documents(chunks)
    return store.similarity_search(query, k=top_k)


def _build_context(chunks: List[Document], label: str) -> Tuple[str, List[Tuple[str, str, str, int | None, str]]]:
    context_lines = []
    sources: List[Tuple[str, str, str, int | None, str]] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        title = chunk.metadata.get("title", label)
        doc_id = chunk.metadata.get("doc_id", label)
        page = chunk.metadata.get("page")
        content = chunk.page_content.strip()
        tag = f"[{label} | Source: {source}"
        if page is not None:
            tag += f" | Page: {page}"
        tag += "]"
        context_lines.append(f"{tag}\n{content}")
        sources.append((source, title, doc_id, page, content))
    return "\n\n".join(context_lines), sources


def compare_books(
    filename_a: str,
    content_a: bytes,
    filename_b: str,
    content_b: bytes,
    question: str,
    scope: str | None,
    top_k: int,
) -> Tuple[str, List[Tuple[str, str, str, int | None, str]], List[Tuple[str, str, str, int | None, str]]]:
    docs_a = _documents_from_upload(filename_a, content_a)
    docs_b = _documents_from_upload(filename_b, content_b)
    query = question if not scope else f"{question}\nScope: {scope}"

    chunks_a = _search_chunks(docs_a, query, top_k, "Book A")
    chunks_b = _search_chunks(docs_b, query, top_k, "Book B")

    context_a, sources_a = _build_context(chunks_a, "Book A")
    context_b, sources_b = _build_context(chunks_b, "Book B")

    system_prompt = (
        "You are a careful reviewer comparing two books. "
        "Use only the provided context and call out gaps."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Scope: {scope or 'general'}\n\n"
        f"Book A Context:\n{context_a}\n\n"
        f"Book B Context:\n{context_b}\n\n"
        "Compare the two and respond with short paragraphs, "
        "highlighting similarities, differences, and any missing information."
    )
    answer = chat(system_prompt, user_prompt)
    return answer, sources_a, sources_b
