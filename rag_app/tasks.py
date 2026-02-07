from __future__ import annotations

from typing import Optional

import redis
from rq import Queue
from rq.job import Job

from rag_app.config import settings
from pathlib import Path

from rag_app.ingest import ingest_file


QUEUE_NAME = "rag-ingest"


def get_redis_conn() -> redis.Redis:
    return redis.from_url(settings.redis_url)


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis_conn())


def process_ingest(path_str: str, title: Optional[str]) -> dict:
    result, _ = ingest_file(path=Path(path_str), title=title)
    return {
        "filename": result.filename,
        "doc_id": result.doc_id,
        "chunks_added": result.chunks_added,
        "skipped": result.skipped,
        "message": result.message,
    }


def enqueue_ingest(path_str: str, title: Optional[str]) -> str:
    queue = get_queue()
    job = queue.enqueue(process_ingest, path_str, title)
    return job.id


def fetch_job(job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=get_redis_conn())
    except Exception:
        return None
