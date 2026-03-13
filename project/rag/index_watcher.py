"""Watcher para indexação automática de documentos."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from project.config.settings import Settings, load_settings
from project.rag.retriever import rebuild_faiss_store
from project.services.logging_service import setup_logging


@dataclass
class _Debouncer:
    delay_s: float
    _lock: Lock = Lock()
    _timer: Timer | None = None

    def trigger(self, fn) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = Timer(self.delay_s, fn)
            self._timer.daemon = True
            self._timer.start()


class _Handler(FileSystemEventHandler):
    def __init__(self, *, settings: Settings, logger) -> None:
        self._settings = settings
        self._logger = logger
        self._debouncer = _Debouncer(delay_s=settings.rag_watch_debounce_s)

    def on_any_event(self, event) -> None:
        path = getattr(event, "src_path", "")
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() not in {".pdf", ".txt", ".md"}:
            return

        def _reindex() -> None:
            self._logger.info("Reindexação automática acionada (%s)", p.name)
            try:
                rebuild_faiss_store(self._settings, self._logger)
            except Exception as exc:
                self._logger.exception("Falha na reindexação automática: %s", exc)

        self._debouncer.trigger(_reindex)


def start_background_watcher(*, settings: Settings, logger):
    docs_dir = settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    handler = _Handler(settings=settings, logger=logger)
    observer.schedule(handler, str(docs_dir), recursive=True)
    observer.daemon = True
    observer.start()
    logger.info("Watcher de documentos ativo em: %s", docs_dir)
    return observer


def run(*, project_root: Path | None = None) -> None:
    settings = load_settings(project_root=project_root)
    logger = setup_logging(settings.logs_dir)

    observer = start_background_watcher(settings=settings, logger=logger)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    run()
