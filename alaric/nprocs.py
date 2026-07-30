"""Default worker counts that honor a batch-scheduler CPU allocation.

``os.cpu_count()`` reports the cores of the whole node, which oversubscribes a SLURM
reservation (``sbatch -c N``). Python >= 3.13 lets ``PYTHON_CPU_COUNT`` override
``os.cpu_count()``, and the generated run scripts export it from ``SLURM_CPUS_PER_TASK``
(see ``middle/deploy.py``) -- but Python 3.12 ignores that variable entirely, so relying on
the interpreter silently gives back the full node core count. The allocation is therefore
read here instead, which works on every supported Python version.

Worker counts are non-load-bearing: they never change the canonical result, so resolving
them from the environment does not affect checksums.
"""

from __future__ import annotations

import os

# Checked in order; the first one that parses as a positive integer wins.
# ``ALARIC_NPROCS`` is the deployment-wide manual override; the rest mirror the ordering
# used by the SLURM prologue in ``middle/deploy.py``.
_ENV_VARS = (
    "ALARIC_NPROCS",
    "PYTHON_CPU_COUNT",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
)


def default_nprocs() -> int:
    """Number of worker processes/threads to use when the caller gave no explicit count."""
    for var in _ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        try:
            count = int(value)
        except ValueError:
            # e.g. PYTHON_CPU_COUNT=default, or a malformed SLURM value; fall through.
            continue
        if count > 0:
            return count
    try:
        # Honors cgroup/affinity binding, which is how SLURM confines a task when the env
        # vars above are absent.
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available outside Linux
        return max(1, os.cpu_count() or 1)
