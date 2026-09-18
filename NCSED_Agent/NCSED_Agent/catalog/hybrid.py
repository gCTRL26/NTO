"""Гибридный поиск: объединение лексической и векторной выдач.

BM25 и векторный поиск ошибаются по-разному — первый не знает синонимов,
второй теряет точные коды классификаторов. Объединение берёт сильные стороны
обеих (ADR-003, решение 1).

Счета двух поисков несравнимы: у BM25 диапазон 4…37 и зависит от запроса,
у косинуса — узкая полоса 0.75…0.90. Поэтому счета не используются вовсе,
объединение идёт по МЕСТАМ в выдачах.
"""

from ..domain.dataset_card import DatasetCard
from .bm25 import BM25Index
from .vectors import VectorIndex

# Сглаживающая константа RRF. При C=60 разрыв между первым и вторым местом мал,
# поэтому важнее не высокое место в одном списке, а присутствие в обоих.
RRF_C = 60

# Сколько кандидатов берём у каждого поиска до объединения.
# Замеры показали цели на 17-м месте у BM25 и на 68-м у вектора (eval-log, замер 4),
# так что глубина должна заметно превышать k на выходе, иначе объединять нечего.
FETCH_K = 100


def rrf_weight(rank: int | None, c: int = RRF_C) -> float:
    """Вклад одного поиска в общий счёт карточки.

    rank — место в выдаче, начиная с 1. None означает, что этот поиск карточку
    не нашёл: тогда вклад нулевой, но карточка не отбрасывается — её может
    вытянуть второй поиск.
    """
    return 0.0 if rank is None else 1.0 / (c + rank)


class HybridIndex:
    """Объединяет выдачи двух индексов методом Reciprocal Rank Fusion.

    Индексами НЕ владеет: они создаются снаружи и закрываются снаружи.
    У VectorIndex есть файловая блокировка на хранилище, и владелец ресурса
    должен быть один — тот, кто его создал.
    """

    def __init__(
        self,
        bm25: BM25Index,
        vectors: VectorIndex,
        fetch_k: int = FETCH_K,
        c: int = RRF_C,
    ):
        self.bm25 = bm25
        self.vectors = vectors
        self.fetch_k = fetch_k
        self.c = c

    def search(self, query: str, k: int = 10) -> list[tuple[DatasetCard, float]]:
        """Топ-k карточек по объединённому счёту.

        Счёт карточки — сумма вкладов от тех поисков, которые её нашли.
        Найденная обоими обгоняет найденную одним, даже если у того она первая.
        """
        results = [
            self.bm25.search(query, k=self.fetch_k),
            self.vectors.search(query, k=self.fetch_k),
        ]

        # Места по каждому поиску плюс сами карточки — чтобы потом не искать их
        # заново. Ключ — card.key: он стабилен и сравнивается за константное
        # время, в отличие от сравнения объектов pydantic по всем полям.
        cards: dict[str, DatasetCard] = {}
        ranks: list[dict[str, int]] = []
        for found in results:
            positions: dict[str, int] = {}
            for rank, (card, _score) in enumerate(found, start=1):
                positions[card.key] = rank
                cards[card.key] = card
            ranks.append(positions)

        scored = [
            (cards[key], sum(rrf_weight(r.get(key), self.c) for r in ranks))
            for key in cards
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]
