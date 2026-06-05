from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import MiddleError


ACTION_FILE = "alaric.yaml"


@dataclass(frozen=True)
class ActionDir:
    name: str
    path: Path
    yaml_path: Path


@dataclass(frozen=True)
class Project:
    root: Path

    @classmethod
    def discover(cls, start: str | Path = ".") -> "Project":
        path = Path(start).resolve()
        if path.is_file():
            path = path.parent
        candidates = [path, *path.parents]
        for candidate in candidates:
            if (candidate / "DATA").is_dir():
                return cls(candidate)
            if (candidate / ACTION_FILE).is_file() and (candidate.parent / "DATA").is_dir():
                return cls(candidate.parent)
        raise MiddleError(f"could not discover project root from {start!s}: no DATA/ found")

    @property
    def data_dir(self) -> Path:
        return self.root / "DATA"

    @property
    def cache_dir(self) -> Path:
        return self.root / "CACHE"

    @property
    def parameters_dir(self) -> Path:
        return self.cache_dir / "parameters"

    @property
    def checksum_dir(self) -> Path:
        return self.cache_dir / "checksum"

    @property
    def results_dir(self) -> Path:
        return self.cache_dir / "results"

    def ensure_cache(self) -> None:
        self.parameters_dir.mkdir(parents=True, exist_ok=True)
        self.checksum_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def action_dirs(self) -> list[ActionDir]:
        result: list[ActionDir] = []
        for child in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name in {"DATA", "CACHE", "templates"}:
                continue
            yaml_path = child / ACTION_FILE
            if yaml_path.is_file():
                result.append(ActionDir(child.name, child, yaml_path))
        return result

    def get_action_dir(self, name_or_path: str | Path = ".") -> ActionDir:
        path = Path(name_or_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file():
            path = path.parent
        if (path / ACTION_FILE).is_file():
            root = Project.discover(path).root
            return ActionDir(path.name, path, path / ACTION_FILE)
        candidate = self.root / str(name_or_path)
        if (candidate / ACTION_FILE).is_file():
            return ActionDir(candidate.name, candidate, candidate / ACTION_FILE)
        raise MiddleError(f"not an action directory: {name_or_path}")

    def iter_data_files(self) -> Iterable[Path]:
        if not self.data_dir.is_dir():
            return []
        return sorted(p for p in self.data_dir.rglob("*") if p.is_file())
