"""Isolated parsing: every file is parsed in a worker process with a timeout and (on Linux/macOS)
an address-space cap, so a malformed or hostile file can hang or crash only its own worker.

Results never raise across the process boundary: a worker returns ("ok", Loaded) or
("err", rule, message); the parent turns errors into ingest-report entries.

Edge cases NOT handled: the memory cap is not enforced on Windows (no RLIMIT_AS); a timeout
terminates the whole pool and recreates it, so files in flight at that moment are retried once.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.pool import Pool
from pathlib import Path

from cgprr.loaders import Limits, Loaded, LoaderError, load_file

Result = tuple[str, object, str]  # ("ok", Loaded, "") | ("err", rule, message)
_MEM_CAP_BYTES = 2 * 1024**3


def _init_worker() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (_MEM_CAP_BYTES, _MEM_CAP_BYTES))
    except (ImportError, ValueError, OSError):  # Windows, or cap not permitted
        pass


def _work(args: tuple[str, Limits]) -> Result:
    path, limits = args
    try:
        return ("ok", load_file(Path(path), limits), "")
    except LoaderError as exc:
        return ("err", exc.rule, str(exc))
    except MemoryError:
        return ("err", "memory_limit", f"{Path(path).name}: exceeded the parser memory cap")
    except Exception as exc:  # noqa: BLE001 - any parser bug becomes a rejected file, not a crash
        return ("err", "parser_error", f"{Path(path).name}: {type(exc).__name__}")


def load_in_process(paths: list[Path], limits: Limits) -> dict[Path, Result]:
    return {p: _work((str(p), limits)) for p in paths}


def load_isolated(paths: list[Path], limits: Limits | None = None, workers: int | None = None) -> dict[Path, Result]:
    lim = limits or Limits()
    if not paths:
        return {}
    if os.environ.get("CGPRR_ISOLATE_PARSING", "1") in ("0", "false", "no"):
        return load_in_process(paths, lim)
    n = workers or max(1, min(4, (os.cpu_count() or 2) // 2, len(paths)))
    ctx = mp.get_context("spawn")  # same behaviour on Windows and Linux; no inherited state
    out: dict[Path, Result] = {}
    pending = list(paths)
    retried: set[Path] = set()
    while pending:
        pool: Pool = ctx.Pool(n, initializer=_init_worker)
        jobs = [(p, pool.apply_async(_work, ((str(p), lim),))) for p in pending]
        pending = []
        timed_out = False
        for p, job in jobs:
            if timed_out:
                if p not in retried:
                    retried.add(p)
                    pending.append(p)
                else:
                    out[p] = ("err", "parse_timeout", f"{p.name}: parsing interrupted twice")
                continue
            try:
                out[p] = job.get(timeout=lim.timeout_s)
            except mp.TimeoutError:
                out[p] = ("err", "parse_timeout", f"{p.name}: parsing exceeded {lim.timeout_s:.0f} s")
                timed_out = True  # a hung worker: tear the pool down, retry the rest once
            except Exception as exc:  # noqa: BLE001 - worker died (e.g. segfault in a C extension)
                out[p] = ("err", "parser_crash", f"{p.name}: {type(exc).__name__}")
        if timed_out:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    return out


def ok(result: Result) -> Loaded | None:
    return result[1] if result[0] == "ok" and isinstance(result[1], Loaded) else None
