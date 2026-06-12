from __future__ import annotations

import sys
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
        return None

    def close(self) -> None:
        self._print(final=True)

    def _print(self, final: bool = False) -> None:
        total = "?" if self.total is None else str(self.total)
        end = "\n" if final else "\r"
        print(f"{self.desc}: {self.n}/{total}", end=end, file=sys.stderr, flush=True)


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
