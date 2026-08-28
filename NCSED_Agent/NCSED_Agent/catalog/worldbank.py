import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ..domain.dataset_card import DatasetCard
from .scan import FileFacts

# Корень данных считаем от расположения пакета, а не от текущей папки:
# относительный путь сломался бы при запуске из другой директории.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "wb"

# У World Bank один файл = один индикатор по всем странам и годам.
# Разрез и уровень географии одинаковы всегда, поэтому это константы, а не разбор.
LEVEL = "Страны мира"
MEASUREMENTS = ["страна", "год"]

# Бюджет описания в коротком тексте. Лимит эмбеддера — 512 токенов, для английского
# это примерно 2000 символов. Оставляем запас под название и темы.
# Число подберём замером recall@k в Ф2b, пока — обоснованная прикидка.
SHORT_DESCRIPTION_CHARS = 1200


def iter_metadata(root: Path = DATA_ROOT) -> Iterator[tuple[str, dict]]:
    """Отдаёт метаданные индикаторов World Bank по одному: (код, сырая запись).

    Весь indicators.json (16 МБ, 29 470 записей) читается в память — для такого
    размера это дешевле и проще, чем потоковый разбор.
    """
    with open(root / "indicators.json", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        code = rec.get("id")
        if not code:
            continue  # запись без идентификатора бесполезна: не найти ни файл, ни ссылку
        yield code, rec


def data_path(code: str, root: Path = DATA_ROOT) -> Path:
    """Где лежит файл с данными этого индикатора."""
    return root / "parquet" / f"{code}.parquet"


def scan_file(path: Path) -> FileFacts:
    if not path.exists():
        return FileFacts(
            key=f"wb:{path.stem}",
            status="Нет файла",
        )

    try:
        df = pd.read_parquet(path, columns=["date", "value", "country_id"])
    except Exception:
        return FileFacts(
            key=f"wb:{path.stem}",
            status="Файл есть, но не читается",
        )
    df = df.dropna(subset=["value"])
    if df.empty:
        return FileFacts(
            key=f"wb:{path.stem}",
            status="Файл есть, но пустой",
            n_geo_units=0,
        )
    years = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
    n_units = int(df["country_id"].dropna().nunique())
    if years.notna().any():
        min_date = int(years.min())
        max_date = int(years.max())

        return FileFacts(
            key=f"wb:{path.stem}",
            status="Данные есть",
            real_year_from=min_date,
            real_year_to=max_date,
            n_geo_units=n_units,
        )
    else:
        return FileFacts(
            key=f"wb:{path.stem}",
            status="Данные есть",
            n_geo_units=n_units,
        )


def _shorten(text: str, limit: int) -> str:
    """Обрезает текст по границе слова, чтобы не рвать слово пополам."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(" ,;:—-") + "…"


def to_card(code: str, meta: dict, facts: FileFacts) -> DatasetCard:
    """Складывает метаданные индикатора и факты из файла в карточку каталога.

    Ничего не открывает и не считает — только раскладывает уже полученное.
    Дорогую часть сделал scan_file.
    """
    name = (meta.get("name") or code).strip()
    description = (meta.get("sourceNote") or "").strip() or None
    publisher = (meta.get("sourceOrganization") or "").strip() or None

    # У тем World Bank в значениях висят хвостовые пробелы: "Poverty ", "Education ".
    topics = [
        (t.get("value") or "").strip()
        for t in (meta.get("topics") or [])
        if (t.get("value") or "").strip()
    ]
    theme = ", ".join(topics) or None

    # Полный текст уходит в BM25 — там нет ни лимита длины, ни усреднения смысла.
    long_text = ". ".join(p for p in (name, theme, publisher, description) if p)

    # Короткий уходит в эмбеддер: длинное описание размывает вектор, см. Решение 4.
    short_parts = [name, theme]
    if description:
        short_parts.append(_shorten(description, SHORT_DESCRIPTION_CHARS))
    short_text = ". ".join(p for p in short_parts if p)

    return DatasetCard(
        key=facts.key,
        source="WorldBank",
        name=name,
        description=description,
        # unit пуст во всех 29 470 записях, угадывать из названия ненадёжно.
        # periodicity выводится из типа колонки date, но её пока не считает scan_file.
        # Оба — долги Д1 и Д2 в docs/data-notes.md.
        unit=None,
        periodicity=None,
        theme=theme,
        real_year_from=facts.real_year_from,
        real_year_to=facts.real_year_to,
        n_geo_units=facts.n_geo_units,
        level=LEVEL,
        long_search_text=long_text,
        short_search_text=short_text,
        status=facts.status,
        url=f"https://data.worldbank.org/indicator/{code}",
        # Путь относительный: каталог переносится между машинами вместе с репозиторием.
        filepath=(
            f"data/wb/parquet/{code}.parquet" if facts.status != "Нет файла" else None
        ),
        publisher=publisher,
        measurements=list(MEASUREMENTS),
    )
