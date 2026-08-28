import json
from collections.abc import Iterator
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "fedstatru"

# Поля метаданных, которые не должны попасть ни в каталог, ни в поисковый индекс,
# ни в контекст LLM. «Ответственный» содержит ФИО, рабочий телефон и почту
# конкретных сотрудников Росстата.
DROPPED_PROPS = {"Ответственный"}


def iter_metadata(root: Path = DATA_ROOT) -> Iterator[tuple[str, dict]]:
    """Отдаёт метаданные показателей ЕМИСС по одному: (код, очищенная запись).

    Метаданные лежат в 7330 отдельных файлах, поэтому читаем их по одному и сразу
    отдаём — держать всё в памяти незачем.

    Персональные данные вырезаются здесь, на входной границе: это единственное
    место, где сырые метаданные попадают в систему.
    """
    for path in sorted((root / "metadata").glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # один битый файл не должен рушить обход остальных 7329

        code = rec.get("code") or path.stem
        rec["props"] = {
            k: v for k, v in (rec.get("props") or {}).items() if k not in DROPPED_PROPS
        }
        yield str(code), rec


def data_path(code: str, root: Path = DATA_ROOT) -> Path:
    """Где лежит файл с данными этого показателя."""
    return root / "parquet" / f"{code}.parquet"
