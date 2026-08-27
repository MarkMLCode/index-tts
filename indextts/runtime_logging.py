"""Per-run API logs and lightweight, monotonic startup timings."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


LOGGER = logging.getLogger("indextts.startup")
_CONFIG_LOCK = threading.Lock()
_RUN_LOG_DIR: Optional[Path] = None
_FILE_ONLY_OUTPUT = ContextVar("indextts_file_only_output", default=False)


@contextmanager
def file_only_output():
    """Keep warmup prints in the file without hiding other threads' output."""
    token = _FILE_ONLY_OUTPUT.set(True)
    try:
        yield
    finally:
        _FILE_ONLY_OUTPUT.reset(token)


class _PrintLoggingStream:
    """Keep console prints unchanged and timestamp complete lines in the file.

    Log handlers use the original streams, never this wrapper, to avoid recursion
    and duplicate console messages. Native subprocess output is not intercepted.
    """

    def __init__(self, logger: logging.Logger, original: TextIO, level: int):
        self._logger = logger
        self._original = original
        self._level = level
        self._buffer = ""
        self._lock = threading.RLock()

    def write(self, message: str) -> int:
        with self._lock:
            if not _FILE_ONLY_OUTPUT.get():
                self._original.write(message)
            self._buffer += message.replace("\r", "\n")
            lines = self._buffer.split("\n")
            self._buffer = lines.pop()
            for line in lines:
                if line.strip():
                    self._logger.log(self._level, line.rstrip())
        return len(message)

    def flush(self) -> None:
        with self._lock:
            self._original.flush()
            if self._buffer.strip():
                self._logger.log(self._level, self._buffer.rstrip())
            self._buffer = ""

    def __getattr__(self, name: str):
        return getattr(self._original, name)


def _configure_logger(name, handlers, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers = list(handlers)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _console_handler(stream, formatter):
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    return handler


def configure_run_logging(log_root: Optional[Path] = None) -> Path:
    """Configure once per process, including when launched via uvicorn api:APP."""
    global _RUN_LOG_DIR
    with _CONFIG_LOCK:
        if _RUN_LOG_DIR is not None:
            return _RUN_LOG_DIR

        project_root = Path(__file__).resolve().parents[1]
        root = Path(log_root or os.environ.get("INDEXTTS_API_LOG_DIR", project_root / "logs"))
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
        run_dir.mkdir()
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(run_dir / "api.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # Respect existing embedding/application handlers; add a console only if
        # there isn't one (FileHandler is also a StreamHandler).
        if not any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            for handler in root_logger.handlers
        ):
            root_logger.addHandler(_console_handler(sys.stderr, logging.Formatter("%(levelname)s [%(name)s] %(message)s")))
        root_logger.addHandler(file_handler)

        # Added diagnostics belong only in the run file. Do not propagate them
        # to root handlers, including consoles configured by an embedding app.
        _configure_logger(LOGGER.name, [file_handler])

        # Match Uvicorn's original console formatting, colors and streams.
        # Keep its logger names/timestamps only in the separate file handler.
        from uvicorn.logging import AccessFormatter, DefaultFormatter

        server_handler = _console_handler(sys.stderr, DefaultFormatter("%(levelprefix)s %(message)s"))
        access_handler = _console_handler(sys.stdout, AccessFormatter(
            '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        ))
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            _configure_logger(name, [access_handler if name == "uvicorn.access" else server_handler, file_handler])

        for stream_name, level in (("stdout", logging.INFO), ("stderr", logging.WARNING)):
            logger = _configure_logger(f"indextts.{stream_name}", [file_handler], level)
            setattr(sys, stream_name, _PrintLoggingStream(logger, getattr(sys, stream_name), level))

        _RUN_LOG_DIR = run_dir
        LOGGER.info("Run log: %s", run_dir / "api.log")
        LOGGER.info("Stage timings are host elapsed time; nested stage totals overlap. No GPU synchronization is added.")
        return run_dir


@contextmanager
def timed_stage(label: str):
    """Log start, success/failure and elapsed seconds; also usable as a decorator.

    Deliberately does not synchronize GPU work or touch RNG/model state. These
    timings diagnose loading latency, not individual asynchronous GPU kernels.
    """
    started = time.perf_counter()
    LOGGER.info("START %s", label)
    try:
        yield
    except BaseException:
        LOGGER.error("FAILED %s | elapsed=%.3fs", label, time.perf_counter() - started)
        raise
    else:
        LOGGER.info("DONE %s | elapsed=%.3fs", label, time.perf_counter() - started)
