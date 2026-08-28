"""Этап 1 загрузчика: дорогой обход файлов с данными.

Открывает каждый файл, чтобы узнать то, чего нет в метаданных: есть ли данные
вообще, за какие годы они реально есть, сколько географических единиц покрыто.
Результат кэшируется — файлы на диске не меняются, а значит и факты о них тоже.

См. ADR-003, решение 5.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


class FileFacts(BaseModel):
    key: str
    status: Literal[
        "Данные есть", "Нет файла", "Файл есть, но пустой", "Файл есть, но не читается"
    ]
    real_year_from: int | None = None
    real_year_to: int | None = None
    n_geo_units: int | None = None


class Source(Protocol):
    """Контракт модуля-источника. Реализуют worldbank.py и fedstat.py."""

    def iter_metadata(self) -> Iterator[tuple[str, dict]]: ...
    def data_path(self, code: str) -> Path: ...
    def scan_file(self, path: Path) -> FileFacts: ...


def scan_source(source: Source, log_every: int = 2000) -> Iterator[FileFacts]:
    """Обходит все показатели источника и отдаёт факты по каждому."""
    for i, (code, _meta) in enumerate(source.iter_metadata(), start=1):
        yield source.scan_file(source.data_path(code))
        if log_every and i % log_every == 0:
            print(f"  просканировано {i}")


def save_facts(facts: Iterator[FileFacts], path: Path) -> int:
    """Пишет факты построчно в JSONL. Возвращает число записей.

    JSONL, а не JSON: строки пишутся по одной, поэтому 35 000 записей
    не нужно держать в памяти целиком ни при записи, ни при чтении.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in facts:
            f.write(item.model_dump_json() + "\n")
            n += 1
    return n


def load_facts(path: Path) -> dict[str, FileFacts]:
    """Читает кэш фактов и раскладывает по ключу для быстрого доступа."""
    result: dict[str, FileFacts] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = FileFacts.model_validate_json(line)
            result[item.key] = item
    return result
