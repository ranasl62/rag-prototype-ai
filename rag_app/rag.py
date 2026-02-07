from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

from rag_app.config import settings
from rag_app.llm import chat
from rag_app.vectorstore import similarity_search


def build_context(chunks: List[Document]) -> Tuple[str, List[Tuple[str, str, str, int | None]]]:
    context_lines = []
    sources = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        title = chunk.metadata.get("title", "untitled")
        doc_id = chunk.metadata.get("doc_id", "unknown")
        page = chunk.metadata.get("page")
        content = chunk.page_content.strip()
        label = f"[Source: {source} | Title: {title} | Doc: {doc_id}"
        if page is not None:
            label += f" | Page: {page}"
        label += "]"
        context_lines.append(f"{label}\n{content}")
        sources.append((source, title, doc_id, page, content))
    return "\n\n".join(context_lines), sources


def rag_search(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> List[Tuple[str, str, str, int | None, str]]:
    k = top_k or settings.top_k
    chunks = similarity_search(query, k, metadata_filter=metadata_filter)
    _, sources = build_context(chunks)
    return sources


def rag_answer(
    question: str,
    top_k: int | None = None,
    metadata_filter: dict | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> Tuple[str, List[Tuple[str, str, str, int | None, str]]]:
    k = top_k or settings.top_k
    chunks = similarity_search(question, k, metadata_filter=metadata_filter)
    context, sources = build_context(chunks)
    system_prompt = (
        "You are a helpful research assistant. Use only the provided context. "
        "If the context is insufficient, say you do not know."
    )
    user_prompt = (
        f"Question: {question}\n\nContext:\n{context}\n\n"
        "Answer in a natural, human tone with short paragraphs. "
        "Include a brief conclusion."
    )
    answer = chat(system_prompt, user_prompt, provider=llm_provider, model_name=llm_model)
    return answer, sources
