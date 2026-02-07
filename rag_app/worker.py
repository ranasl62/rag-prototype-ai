from __future__ import annotations

from rq import Connection, Queue, Worker

from rag_app.tasks import QUEUE_NAME, get_redis_conn


def main() -> None:
    with Connection(get_redis_conn()):
        worker = Worker([Queue(QUEUE_NAME)])
        worker.work()


if __name__ == "__main__":
    main()
