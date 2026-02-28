from __future__ import annotations

import time


def _format_duration(seconds: float) -> str:
    if seconds is None:
        return "--:--"
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    total = max(0, int(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"


class ProgressTracker:
    def __init__(
        self,
        name: str,
        total: int,
        *,
        enabled: bool = True,
        percent_step: int = 5,
    ) -> None:
        self.name = str(name)
        self.total = max(1, int(total))
        self.enabled = bool(enabled)
        self.percent_step = max(1, int(percent_step))
        self.current = 0
        self._start = time.time()
        self._last_bucket = -1
        if self.enabled:
            print(f"[{self.name}] {self.current}/{self.total} (0.0%)")

    def update(self, advance: int = 1, *, extra: str | None = None) -> None:
        self.current = min(self.total, self.current + max(0, int(advance)))
        if not self.enabled:
            return
        pct = 100.0 * float(self.current) / float(self.total)
        bucket = int(pct // self.percent_step)
        should_print = self.current >= self.total or bucket > self._last_bucket
        if not should_print:
            return
        elapsed = time.time() - self._start
        eta = float("inf")
        if self.current > 0:
            eta = (elapsed / float(self.current)) * float(self.total - self.current)
        msg = (
            f"[{self.name}] {self.current}/{self.total} ({pct:.1f}%) "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
        )
        if extra:
            msg = f"{msg} | {extra}"
        print(msg)
        self._last_bucket = bucket

    def close(self, *, extra: str | None = None) -> None:
        if self.current < self.total:
            self.update(self.total - self.current, extra=extra)
            return
        if not self.enabled:
            return
        elapsed = time.time() - self._start
        msg = f"[{self.name}] done in {_format_duration(elapsed)}"
        if extra:
            msg = f"{msg} | {extra}"
        print(msg)
