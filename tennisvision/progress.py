"""Small terminal progress helpers with no external dependencies."""

import sys
import time


class Progress:
    """Displays completed work, elapsed time and ETA on one terminal line."""

    def __init__(self, label: str, total: int | None = None,
                 min_interval: float = 0.5):
        self.label = label
        self.total = total
        self.min_interval = min_interval
        self.started = time.monotonic()
        self.last_print = 0.0
        self.completed = 0
        self.final_printed = False
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._write(0, final=False)

    def update(self, completed: int) -> None:
        """Updates the display if enough time passed or work is complete."""
        self.completed = completed
        now = time.monotonic()
        final = self.total is not None and completed >= self.total
        if final or now - self.last_print >= self.min_interval:
            self._write(completed, final=final)

    def close(self, completed: int | None = None) -> None:
        """Prints the final progress state and terminates the line."""
        if completed is not None:
            self.completed = completed
        self._write(self.completed, final=True)

    def _write(self, completed: int, final: bool) -> None:
        if final and self.final_printed:
            return
        now = time.monotonic()
        elapsed = now - self.started
        parts = [self.label]
        if self.total:
            completed = min(completed, self.total)
            percent = 100.0 * completed / self.total
            parts.append(f"{completed}/{self.total} ({percent:5.1f}%)")
            if completed > 0 and completed < self.total:
                eta = elapsed * (self.total - completed) / completed
                parts.append(f"ETA {_duration(eta)}")
        else:
            parts.append(str(completed))
        parts.append(f"elapsed {_duration(elapsed)}")
        text = "  " + " | ".join(parts)

        if self.is_tty:
            print(f"\r{text:<100}", end="\n" if final else "", flush=True)
        elif final or now - self.last_print >= max(2.0, self.min_interval):
            print(text, flush=True)
        self.last_print = now
        self.final_printed = final


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
