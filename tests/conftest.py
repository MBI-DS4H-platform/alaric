"""Test-process cleanup for optional global libraries with registered atexit hooks."""

import sys


def pytest_sessionfinish(session, exitstatus):
    # The middle checksum helpers use Seamless' global buffer cache. Close it while the
    # imported modules and their remote database API are still intact; its atexit hook is
    # deliberately best-effort and can otherwise run after Python starts tearing modules
    # down.
    seamless = sys.modules.get("seamless")
    if seamless is not None:
        seamless.close()
