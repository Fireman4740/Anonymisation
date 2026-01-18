from __future__ import annotations

from typing import Optional


class AppError(Exception):
    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
