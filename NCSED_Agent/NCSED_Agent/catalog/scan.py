from pydantic import BaseModel
from typing import Literal


class FileFacts(BaseModel):
    key: str
    status: Literal[
        "Данные есть", "Нет файла", "Файл есть, но пустой", "Файл есть, но не читается"
    ]
    real_year_from: int | None = None
    real_year_to: int | None = None
    n_geo_units: int | None = None
