from pydantic import BaseModel, Field, model_validator
from typing import Literal


class ResearchDefinitionDraft(BaseModel):
    """Часть определения, которую заполняет LLM"""

    year_from: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Год начала периода исследования. Если не указан оставь null",
    )
    year_to: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Год окончания периода исследования. Если не указан оставь null",
    )
    period_need: Literal[
        "Указан пользователем",
        "Любой, пользователю неважно",
        "Не указан, надо уточнить",
    ] = Field(
        description=(
            "Как период задан в запросе. "
            "«Указан пользователем» — названы конкретные годы или интервал. "
            "«Любой, пользователю неважно» — пользователь явно сказал, что период "
            "не важен или что нужны все доступные данные. "
            "«Не указан, надо уточнить» — про период в запросе ничего нет."
        )
    )

    discipline: list[str] | None = Field(
        default=None,
        description=(
            "Дисциплинарный ракурс: с точки зрения какой области смотрим на вопрос. "
            "Например: экономика, социология, демография, экология, здравоохранение, "
            "маркетинг. Если из запроса не следует — верни null."
        ),
    )
    geography: list[str] | None = Field(
        default=None,
        description="Страны или регионы для исследования. Если не указаны оставь null",
    )

    research_questions: list[str] | None = Field(
        default=None,
        max_length=5,
        description=(
            "Конкретные исследовательские вопросы, на которые должно ответить "
            "исследование. Формулируй в вопросительной форме, например: "
            "«Как изменился объём товарооборота между Россией и Казахстаном "
            "в 2015-2024 годах?». Не более 5 вопросов. "
            "Если запрос слишком общий, чтобы поставить вопрос — верни null."
        ),
    )

    @model_validator(mode="after")
    def _check_year_order(self):
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from не может быть больше year_to")
        return self


class ResearchDefinition(ResearchDefinitionDraft):
    """Полное определение: то что известно коду и то что заполнила LLM"""

    original_query: str = Field(description="Исходный текст запроса пользователя")
