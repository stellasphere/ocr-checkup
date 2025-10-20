from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
import json


@contextmanager
def jsonl_writer(path: Path) -> Iterator[Callable[[dict], None]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        def write(obj: dict) -> None:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
            f.flush()

        yield write


def iterate_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


__all__ = [
    "jsonl_writer",
    "iterate_jsonl",
]
