from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class ProgressLogger:
    """Small stdout progress logger for long-running reproduction commands."""

    enabled: bool = True
    stream: TextIO = sys.stdout
    start_time: float = field(default_factory=time.monotonic)

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.start_time
        print(f"[{elapsed:8.1f}s] {message}", file=self.stream, flush=True)

    def step(self, message: str) -> None:
        self.log(f"Starting: {message}")

    def done(self, message: str) -> None:
        self.log(f"Finished: {message}")

