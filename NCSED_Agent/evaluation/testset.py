"""Тест-набор для оценки поиска.

Каждый кейс — пара «запрос пользователя → карточки, которые обязаны найтись».
Эталон намеренно неполный: под запрос «динамика ВВП» подходят десятки карточек,
но размечать все не нужно. Нам важно сравнивать варианты поиска между собой
на ОДНОМ И ТОМ ЖЕ эталоне, а для этого достаточно, чтобы он был последовательным.

Ключи карточек стабильны между пересборками каталога, поэтому ссылаемся на них,
а не на позиции в файле.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CASES_PATH = Path(__file__).resolve().parent / "cases" / "search.json"

Kind = Literal[
    "lookup",
    "cross_lingual",
    "multi_source",
    "ambiguous",
    # Показателя нет ни в одном источнике. Ловится на этапе поиска.
    "not_found",
    # Показатель ЕСТЬ, но данных по нужной стране или периоду нет.
    # Поиск обязан найти карточку — это правильное поведение. Отсутствие покрытия
    # обнаруживается только открытием файла, поэтому проверяется на более позднем шаге.
    "no_coverage",
]


class SearchCase(BaseModel):
    """Один тест-кейс поиска."""

    query: str = Field(description="Запрос так, как его написал бы пользователь")
    kind: Kind = Field(description="Тип кейса — метрики считаются и в разрезе типов")
    must_find: list[str] = Field(
        default_factory=list,
        description="Ключи карточек, которые обязаны попасть в топ-k. "
        "Для kind='not_found' список пуст: правильный ответ — ничего не найти.",
    )
    note: str = Field(default="", description="Почему выбраны именно эти карточки")


def load_cases(path: Path = CASES_PATH) -> list[SearchCase]:
    with open(path, encoding="utf-8") as f:
        return [SearchCase.model_validate(item) for item in json.load(f)]


def validate_cases(cases: list[SearchCase], known_keys: set[str]) -> None:
    """Проверяет, что все упомянутые ключи есть в каталоге.

    Без этой проверки исчезнувший из каталога ключ перестал бы проверяться молча,
    метрика выросла бы, и мы решили бы, что поиск стал лучше. Ошибка должна быть
    громкой.
    """
    missing = {
        key
        for case in cases
        for key in case.must_find
        if key not in known_keys
    }
    if missing:
        raise ValueError(
            f"В тест-наборе {len(missing)} ключей, которых нет в каталоге: "
            f"{sorted(missing)[:5]}. Пересобери каталог или поправь кейсы."
        )
