"""Лексический поиск по каталогу.

BM25 ранжирует карточки по совпадению слов. Он не понимает синонимов —
«ВРП» не найдёт «валовой региональный продукт», — зато точно находит коды
классификаторов и редкие термины, которые векторный поиск теряет.
Вторую половину пары делает векторный индекс, см. ADR-003, решение 1.

Ручная проверка:
    python -m NCSED_Agent.catalog.bm25 "динамика ВРП Архангельской области"
"""

import re
import sys
from functools import lru_cache

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

from ..domain.dataset_card import DatasetCard

# Слово — буквы кириллицы или латиницы и цифры. Всё остальное (дефисы, скобки,
# знаки процента) считаем разделителями.
WORD_RE = re.compile(r"[а-яёa-z0-9]+")
CYRILLIC_RE = re.compile(r"[а-яё]")

_ru = SnowballStemmer("russian")
_en = SnowballStemmer("english")

# Каталог двуязычный: русские названия Росстата и английские World Bank.
# Стеммер выбирается по наличию кириллицы в самом слове, а не по языку документа.
#
# Кэш обязателен: в каталоге 2.2 млн словоупотреблений, но лишь 31 тыс. уникальных
# слов — без него стемминг занимает 71 секунду вместо двух. Размер ограничен,
# чтобы поток запросов пользователей не растил кэш бесконечно.
@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    return _ru.stem(word) if CYRILLIC_RE.search(word) else _en.stem(word)


def tokenize(text: str) -> list[str]:
    """Приводит текст к списку основ слов.

    Стемминг критичен для русского: без него запрос «продукт» не найдёт
    карточку со словом «продукта». Словарь при этом сокращается вдвое.
    """
    return [stem(word) for word in WORD_RE.findall(text.lower())]


class BM25Index:
    """Лексический индекс над карточками каталога.

    Строится за несколько секунд и занимает около 40 МБ, поэтому не кэшируется
    на диск — дешевле пересобрать, чем поддерживать согласованность файла.
    """

    def __init__(self, cards: list[DatasetCard]):
        self.cards = cards
        # Индексируем длинный текст: у BM25 нет ни лимита длины, ни усреднения
        # смысла, поэтому методология целиком идёт сюда (ADR-003, решение 4).
        self._bm25 = BM25Okapi([tokenize(c.long_search_text) for c in cards])

    def search(self, query: str, k: int = 10) -> list[tuple[DatasetCard, float]]:
        """Топ-k карточек по запросу. Карточки с нулевым счётом не возвращаются.

        Отсекать нули важно: BM25 ранжирует всё подряд, и без отсечки на запрос
        без совпадений вернулись бы десять случайных карточек с околонулевым
        весом. Система обязана уметь ответить «ничего не нашлось».
        """
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # argpartition находит k наибольших без полной сортировки 36 800 значений.
        top = np.argpartition(scores, -k)[-k:] if len(scores) > k else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(self.cards[i], float(scores[i])) for i in top if scores[i] > 0]


def _main() -> None:
    from .build import load_cards

    query = " ".join(sys.argv[1:])
    if not query:
        print(__doc__)
        return

    cards = load_cards()
    index = BM25Index(cards)
    for card, score in index.search(query):
        print(f"{score:6.2f}  {card.key:22} {card.name[:60]}")
        print(f"        {card.status} | {card.real_year_from}-{card.real_year_to} | {card.level}")


if __name__ == "__main__":
    _main()
