import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ..domain.dataset_card import DatasetCard
from .scan import FileFacts

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "fedstatru"

# Бюджет описания в коротком тексте. Лимит эмбеддера 512 токенов; для русского
# это примерно 3 символа на токен, то есть ~1500 символов. Оставляем запас
# под название, тему и единицу. Уточним замером recall@k в Ф2b.
SHORT_DESCRIPTION_CHARS = 900

# Названия полей метаданных ЕМИСС — длинные, выносим в константы,
# чтобы опечатка падала при импорте, а не превращалась в тихий None.
P_UNIT = "Единицы измерения"
P_PERIODICITY = "Периодичность и характеристика временного ряда"
P_DIMENSIONS = "Признаки (перечень на базе классификаторов и справочников)"
P_METHODOLOGY = "Методологические пояснения"
P_SOURCES = "Источники и способ формирования показателя"
P_AGENCY = "Ведомство (субъект статистического учета)"
P_PLACEMENT = "Размещение"

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


def _as_year(value) -> int | None:
    """Подписи колонок-годов приходят как float64: 2009.0, а не '2009'."""
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def scan_file(path: Path) -> FileFacts:
    """Достаёт из широкой выгрузки ЕМИСС то, чего нет в метаданных.

    Формат: строка 0 — заголовок, годы разложены ПО КОЛОНКАМ, схема своя
    в каждом файле. Поэтому читаем файл целиком: какие колонки нужны,
    known только после разбора заголовка.
    """
    key = f"fedstat:{path.stem}"

    if not path.exists():
        return FileFacts(key=key, status="Нет файла")

    try:
        df = pd.read_parquet(path)
    except Exception:
        return FileFacts(key=key, status="Файл есть, но не читается")

    if df.empty:
        return FileFacts(key=key, status="Файл есть, но пустой", n_geo_units=0)

    header = df.iloc[0]
    year_columns = {
        col: year
        for col in df.columns
        if (year := _as_year(header[col])) is not None
    }
    if not year_columns:
        # Примерно у 3% файлов строки-заголовка нет вовсе: строка 0 — уже данные.
        # Без подписей нельзя понять, где годы, а угадывать значит выдумывать.
        return FileFacts(key=key, status="Файл есть, но не читается")

    body = df.iloc[1:]
    filled = body[list(year_columns)].notna()
    if not filled.to_numpy().any():
        return FileFacts(key=key, status="Файл есть, но пустой", n_geo_units=0)

    # Только те годы, где реально есть значения: колонка 2025 может быть пустой.
    years = [year for col, year in year_columns.items() if filled[col].any()]
    rows_with_data = filled.any(axis=1)

    geo_column = next((c for c in df.columns if "ОКАТО" in str(header[c])), None)
    if geo_column is None:
        # Разреза по территориям нет — показатель идёт по РФ целиком.
        n_geo_units = 1
    else:
        n_geo_units = int(body.loc[rows_with_data, geo_column].nunique())

    return FileFacts(
        key=key,
        status="Данные есть",
        real_year_from=min(years),
        real_year_to=max(years),
        n_geo_units=n_geo_units,
    )


def _top_level_items(value: str | None) -> list[str]:
    """Разбирает многоуровневый список ЕМИСС, оставляя только верхний уровень.

    Верхний уровень помечен «- », вложенные подробности — «  * ». Например
    у периодичности верхний уровень это «Годовая», а вложенное —
    «Характеристика: не охарактеризована», которое нам не нужно.
    """
    if not value:
        return []
    items = []
    for line in value.splitlines():
        if line.startswith("- "):
            item = line[2:].strip()
            if item:
                items.append(item)
    return items


def _clean_units(value: str | None) -> str | None:
    """Единицы приходят как «* процент», иногда несколькими строками.

    Значение «-» означает, что единица у показателя не определена, — возвращаем
    None, а не прочерк: «неизвестно» и «прочерк» это разные утверждения.
    """
    if not value:
        return None
    items = [line.strip().lstrip("*").strip() for line in value.splitlines()]
    items = [i for i in items if i and i != "-"]
    return ", ".join(dict.fromkeys(items)) or None


def _shorten(text: str, limit: int) -> str:
    """Обрезает текст по границе слова, чтобы не рвать слово пополам."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(" ,;:—-") + "…"


def to_card(code: str, meta: dict, facts: FileFacts) -> DatasetCard:
    """Складывает метаданные показателя и факты из файла в карточку каталога.

    Ничего не открывает: дорогую часть сделал scan_file.
    """
    props = meta.get("props") or {}
    name = (meta.get("name") or code).strip()

    unit = _clean_units(props.get(P_UNIT))
    periodicity = ", ".join(_top_level_items(props.get(P_PERIODICITY))) or None
    dimensions = _top_level_items(props.get(P_DIMENSIONS))
    description = (props.get(P_METHODOLOGY) or "").strip() or None
    publisher = (props.get(P_AGENCY) or "").strip() or None
    sources = (props.get(P_SOURCES) or "").strip() or None

    # «Размещение» — название раздела на сайте, единственный кандидат в тему
    # (своей таксономии у ЕМИСС нет). Бывает многострочным перечнем вопросников,
    # поэтому берём только первую строку, остальное — шум.
    placement = (props.get(P_PLACEMENT) or "").strip()
    theme = placement.splitlines()[0].strip() if placement else None

    # Уровень географии берём ИЗ ДАННЫХ, а не из метаданных: у 15% показателей
    # ОКАТО заявлен среди измерений, но заполнен единственным значением
    # «643 Российская Федерация». Метаданные обещают разбивку по субъектам,
    # которой в файле нет. К метаданным откатываемся, только если файл не читался.
    if facts.n_geo_units is not None:
        level = "Регионы РФ" if facts.n_geo_units > 1 else "РФ"
    else:
        level = "Регионы РФ" if any("ОКАТО" in d for d in dimensions) else "РФ"

    # Полный текст уходит в BM25: лимита длины нет, лишние слова не мешают.
    long_text = ". ".join(
        p
        for p in (name, theme, publisher, unit, ", ".join(dimensions), sources, description)
        if p
    )

    # Короткий уходит в эмбеддер: длинная методология размывает вектор (ADR-003).
    short_parts = [name, theme, unit]
    if description:
        short_parts.append(_shorten(description, SHORT_DESCRIPTION_CHARS))
    short_text = ". ".join(p for p in short_parts if p)

    return DatasetCard(
        key=facts.key,
        source="Росстат",
        name=name,
        description=description,
        unit=unit,
        periodicity=periodicity,
        theme=theme,
        real_year_from=facts.real_year_from,
        real_year_to=facts.real_year_to,
        n_geo_units=facts.n_geo_units,
        level=level,
        long_search_text=long_text,
        short_search_text=short_text,
        status=facts.status,
        url=f"https://www.fedstat.ru/indicator/{code}",
        filepath=(
            f"data/fedstatru/parquet/{code}.parquet"
            if facts.status != "Нет файла"
            else None
        ),
        publisher=publisher,
        measurements=dimensions or None,
    )
