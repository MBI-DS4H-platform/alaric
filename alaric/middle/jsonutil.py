from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(value, sort_keys=True, indent=2, separators=(",", ": "))
    return (text + "\n").encode()


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)
