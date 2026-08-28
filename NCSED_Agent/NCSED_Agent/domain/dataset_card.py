from pydantic import BaseModel
from typing import Literal


class DatasetCard(BaseModel):
    """Класс описания карточки датасета для ответа на запрос юзера"""

    key: str
    source: Literal["WorldBank", "Росстат"]
    name: str
    description: str | None
    unit: str | None
    periodicity: str | None
    theme: str | None
    real_year_from: int | None
    real_year_to: int | None
    level: Literal["Страны мира", "РФ", "Регионы РФ"]
    long_search_text: str
    short_search_text: str
    status: Literal[
        "Данные есть", "Нет файла", "Файл есть, но пустой", "Файл есть, но не читается"
    ]
    url: str
    filepath: str | None
    publisher: str | None
    measurements: list[str] | None
    # Имя нейтральное: у World Bank это страны, у Росстата — регионы РФ.
    # Совпадает с полем в FileFacts, откуда значение и приходит.
    n_geo_units: int | None
