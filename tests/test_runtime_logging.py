from __future__ import annotations

import io
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from indextts import runtime_logging


def test_stage_reports_monotonic_elapsed_time(caplog, monkeypatch):
    clock = iter((100.0, 102.125))
    monkeypatch.setattr(runtime_logging.time, "perf_counter", lambda: next(clock))
    with caplog.at_level(logging.INFO, logger="indextts.startup"):
        with runtime_logging.timed_stage("CPU checkpoint load"):
            pass
    assert "START CPU checkpoint load" in caplog.text
    assert "DONE CPU checkpoint load | elapsed=2.125s" in caplog.text


def test_stage_logs_failure_and_preserves_exception(caplog, monkeypatch):
    clock = iter((100.0, 101.5))
    monkeypatch.setattr(runtime_logging.time, "perf_counter", lambda: next(clock))
    error = ValueError("bad checkpoint")
    with caplog.at_level(logging.INFO, logger="indextts.startup"):
        with pytest.raises(ValueError) as raised:
            with runtime_logging.timed_stage("GPU setup"):
                raise error
    assert raised.value is error
    assert "FAILED GPU setup | elapsed=1.500s" in caplog.text
    assert "DONE GPU setup" not in caplog.text


def test_stage_decorator_can_be_reused(caplog, monkeypatch):
    clock = iter((10.0, 11.0, 20.0, 23.0))
    monkeypatch.setattr(runtime_logging.time, "perf_counter", lambda: next(clock))

    @runtime_logging.timed_stage("load")
    def load(value):
        """Original documentation."""
        return value

    with caplog.at_level(logging.INFO, logger="indextts.startup"):
        assert load(1) == 1
        assert load(2) == 2
    assert load.__name__ == "load"
    assert load.__doc__ == "Original documentation."
    assert "elapsed=1.000s" in caplog.text
    assert "elapsed=3.000s" in caplog.text


def test_print_stream_handles_fragmented_lines_and_flush():
    original = io.StringIO()
    records = io.StringIO()
    logger = logging.Logger("test.print")
    logger.addHandler(logging.StreamHandler(records))
    wrapped = runtime_logging._PrintLoggingStream(logger, original, logging.INFO)

    assert wrapped.write("GPT ") == 4
    assert records.getvalue() == ""
    wrapped.write("ready\nprogress\rpartial")
    wrapped.flush()

    assert original.getvalue() == "GPT ready\nprogress\rpartial"
    assert records.getvalue() == "GPT ready\nprogress\npartial\n"
    assert wrapped.isatty() is False


@pytest.mark.parametrize("existing_console", [False, True])
def test_per_run_logs_capture_prints_and_uvicorn_once_with_timestamps(tmp_path, existing_console):
    # Isolate global logger/stream setup from pytest and the other tests.
    script = """
import logging
import sys
from pathlib import Path
import api
from indextts.runtime_logging import configure_run_logging, timed_stage
root = Path(sys.argv[1])
if sys.argv[2] == 'True':
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s [%(name)s] %(message)s'))
    logging.getLogger().addHandler(handler)
run = configure_run_logging(root)
assert configure_run_logging(root) == run
print('stdout marker')
print('stderr marker', file=sys.stderr)
logging.getLogger('uvicorn.error').info('server marker')
logging.getLogger('uvicorn.access').info('access marker')
api.STARTUP_LOGGER.info('startup diagnostic marker')
api.LOGGER.error('api error marker')
with timed_stage('test load'):
    logging.getLogger('api').info('model marker')
try:
    with timed_stage('test failure'):
        raise ValueError('expected test failure')
except ValueError:
    pass
sys.stdout.write('partial marker')
sys.stdout.flush()
logging.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "run logs"), str(existing_console)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    logs = list((tmp_path / "run logs").glob("*/api.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    console = result.stdout + result.stderr
    for marker in ("stdout marker", "stderr marker", "server marker", "access marker", "model marker", "partial marker", "api error marker"):
        assert content.count(marker) == 1
        assert console.count(marker) == 1
    for line in content.splitlines():
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} ", line)
    assert not re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", console, re.MULTILINE)
    assert "INFO [uvicorn.error] server marker" in console
    assert "INFO [uvicorn.access] access marker" in console
    assert "INFO [api] model marker" in console
    for marker in (
        "Run log:", "Stage timings are host elapsed time", "startup diagnostic marker",
        "START test load", "DONE test load | elapsed=",
        "START test failure", "FAILED test failure | elapsed=",
    ):
        assert content.count(marker) == 1
        assert marker not in console


def test_importing_api_does_not_configure_logging_or_create_files(tmp_path):
    script = """
import os
import sys
os.environ['INDEXTTS_API_LOG_DIR'] = sys.argv[1]
import api
from indextts import runtime_logging
assert runtime_logging._RUN_LOG_DIR is None
"""
    log_root = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, "-c", script, str(log_root)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not log_root.exists()
