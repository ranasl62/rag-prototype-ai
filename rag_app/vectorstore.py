from __future__ import annotations

from typing import Iterable, List, Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_app.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embeddings_model)


def get_vectorstore() -> Chroma:
    if settings.chroma_http:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        return Chroma(
            collection_name="rag",
            client=client,
            embedding_function=get_embeddings(),
        )
    return Chroma(
        collection_name="rag",
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )


def add_documents(docs: Iterable[Document], ids: Optional[List[str]] = None) -> int:
    vectorstore = get_vectorstore()
    documents = list(docs)
    if not documents:
        return 0
    if ids:
        vectorstore.add_documents(documents, ids=ids)
    else:
        vectorstore.add_documents(documents)
    vectorstore.persist()
    return len(documents)


def similarity_search(
    query: str, top_k: int, metadata_filter: Optional[dict] = None
) -> List[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=top_k, filter=metadata_filter)
