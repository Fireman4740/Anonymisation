from __future__ import annotations

import sys
from dataclasses import dataclass


def log(message: str) -> None:
    print(f"[atlas] {message}", flush=True)


@dataclass
class ProgressBar:
    total: int
    label: str
    width: int = 28
    current: int = 0

    def __post_init__(self) -> None:
        self.update(self.current)

    def update(self, current: int, extra: str = "") -> None:
        self.current = max(0, min(current, self.total))
        ratio = self.current / self.total if self.total else 1.0
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(ratio * 100)
        suffix = f" {extra}" if extra else ""
        sys.stdout.write(f"\r[atlas] {self.label:<18} [{bar}] {self.current}/{self.total} {percent:>3}%{suffix}")
        sys.stdout.flush()

    def advance(self, step: int = 1, extra: str = "") -> None:
        self.update(self.current + step, extra=extra)

    def close(self, extra: str = "") -> None:
        self.update(self.total, extra=extra)
        sys.stdout.write("\n")
        sys.stdout.flush()
