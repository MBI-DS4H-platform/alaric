"""Minimal fresh-process entry point for W6 A9 cold-spawn checks."""
from __future__ import annotations

import os


def run_check(send_conn, payload: tuple) -> None:
    try:
        for name in ("SEAMLESS_CACHE", "SEAMLESS_DATABASE", "SEAMLESS_HASHSERVER"):
            os.environ.pop(name, None)
        from alaric.gaussian_check import run_gaussian_check

        send_conn.send(("ok", run_gaussian_check(*payload)))
    except BaseException as exc:  # propagate worker failures to parent pytest
        send_conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        send_conn.close()
