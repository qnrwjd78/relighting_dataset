from __future__ import annotations

import sys
import time
from typing import Iterable, TypeVar


T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal containers
    tqdm = None


class SimpleProgress:
    def __init__(self, iterable: Iterable[T] | None = None, total: int | None = None, desc: str = "", initial: int = 0):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.n = initial
        self._last_printed = initial
        self.start_time = time.monotonic()
        self.postfix = ""
        self._print()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, value: int = 1) -> None:
        self.n += value
        if self.n != self._last_printed:
            self._print()
            self._last_printed = self.n

    def set_postfix(self, *args, **kwargs) -> None:
        parts = []
        for arg in args:
            if arg:
                parts.append(str(arg))
        for key, value in kwargs.items():
            parts.append(f"{key}={value}")
        self.postfix = ", ".join(parts)
        self._print()

    def close(self) -> None:
        self._print(final=True)

    def _print(self, final: bool = False) -> None:
        total = "?" if self.total is None else str(self.total)
        elapsed = max(time.monotonic() - self.start_time, 0.0)
        rate = self.n / elapsed if elapsed > 0.0 and self.n > 0 else 0.0
        eta = None
        if self.total is not None and rate > 0.0:
            eta = max((self.total - self.n) / rate, 0.0)
        percent = ""
        if self.total:
            percent = f" {min(max(self.n / self.total, 0.0), 1.0) * 100.0:5.1f}%"
        rate_text = f"{rate:.2f}/s" if rate > 0.0 else "?/s"
        eta_text = format_duration(eta) if eta is not None else "?"
        elapsed_text = format_duration(elapsed)
        postfix = f" | {self.postfix}" if self.postfix else ""
        end = "\n" if final else "\r"
        print(
            f"{self.desc}: {self.n}/{total}{percent} | elapsed {elapsed_text} | eta {eta_text} | {rate_text}{postfix}",
            end=end,
            file=sys.stderr,
            flush=True,
        )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    initial: int = 0,
    unit: str = "scene",
    leave: bool = True,
    **kwargs,
):
    if tqdm is None:
        return SimpleProgress(iterable=iterable, total=total, desc=desc, initial=initial)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        initial=initial,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        **kwargs,
    )


def progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)
