"""Векторный поиск по каталогу.

Дополняет BM25 там, где тот бессилен: связывает «ВВП» с «GDP» и «сколько живут»
с «продолжительностью жизни». Замеры показали, что лексический поиск даёт ноль
на всех десяти межъязыковых кейсах — см. docs/eval-log.md, замер 3.

В отличие от BM25 индекс СОХРАНЯЕТСЯ на диск: прогон 36 800 текстов через
нейросеть занимает минуты, при каждом старте это делать нельзя.

Построение:
    python -m NCSED_Agent.catalog.vectors build
Проверка:
    python -m NCSED_Agent.catalog.vectors "динамика ВВП по странам"
"""

import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..domain.dataset_card import DatasetCard

COLLECTION = "cards"


class VectorSettings(BaseSettings):
    """Имена полей совпадают с именами переменных в .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Модель фиксируем в настройках: индекс намертво привязан к ней, и запрос
    # обязан кодироваться той же самой — см. ADR-003, решение 2.
    embedding_model: str = "intfloat/multilingual-e5-base"

    # Семейство e5 обучено с префиксами, причём РАЗНЫМИ для запроса и документа.
    # Перепутать или забыть — тихая просадка качества без единой ошибки.
    embedding_query_prefix: str = "query:"
    embedding_passage_prefix: str = "passage:"

    qdrant_path: str = "./qdrant_storage"


class VectorIndex:
    """Плотный индекс над карточками каталога.

    Модель загружается лениво: она весит около гигабайта, и при поиске по
    готовому индексу нужна только для кодирования одного запроса.
    """

    def __init__(
        self, cards: list[DatasetCard], settings: VectorSettings | None = None
    ):
        self.cards = cards
        self.by_key = {c.key: c for c in cards}
        self.settings = settings or VectorSettings()
        self.client = QdrantClient(path=self.settings.qdrant_path)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def build(self, batch_size: int = 32, chunk: int = 2000) -> None:
        """Кодирует все карточки и складывает векторы в Qdrant.

        В вектор идёт КОРОТКИЙ текст: эмбеддинг усредняет смысл по всему входу,
        и длинная методология размывает название (ADR-003, решение 4).
        """
        dim = self.model.get_embedding_dimension()
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        prefix = self.settings.embedding_passage_prefix
        texts = [f"{prefix} {c.short_search_text}" for c in self.cards]

        # Пишем частями: держать 36 800 векторов и все объекты Qdrant в памяти
        # одновременно незачем, а прогресс виден по ходу.
        for start in range(0, len(texts), chunk):
            part = texts[start : start + chunk]
            vectors = self.model.encode(
                part,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.client.upsert(
                COLLECTION,
                points=[
                    # id — позиция карточки: ключи вида "wb:NY.GDP.MKTP.CD"
                    # Qdrant в качестве идентификатора не принимает.
                    # Сам ключ кладём в payload, карточку берём из словаря в памяти.
                    PointStruct(
                        id=start + i,
                        vector=vec.tolist(),
                        payload={"key": self.cards[start + i].key},
                    )
                    for i, vec in enumerate(vectors)
                ],
            )
            print(
                f"  проиндексировано {min(start + chunk, len(texts))} из {len(texts)}"
            )

    def close(self) -> None:
        """Локальный Qdrant держит файловую блокировку на каталоге хранилища.

        Без явного закрытия она снимается в __del__ при завершении процесса,
        когда часть модулей уже выгружена, — отсюда шумный ModuleNotFoundError
        в конце каждого прогона.
        """
        self.client.close()

    def __enter__(self) -> "VectorIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def search(self, query: str, k: int = 10) -> list[tuple[DatasetCard, float]]:
        """Топ-k карточек по смысловой близости.

        Порог не ставим: у косинусной близости ближайший сосед существует всегда,
        поэтому «ничего не нашлось» здесь выразить нельзя. Честный отказ —
        отдельный шаг проверки релевантности, см. docs/eval-log.md.
        """
        vector = self.model.encode(
            f"{self.settings.embedding_query_prefix} {query}",
            normalize_embeddings=True,
        )
        found = self.client.query_points(COLLECTION, query=vector.tolist(), limit=k)
        return [
            (self.by_key[p.payload["key"]], float(p.score))
            for p in found.points
            if p.payload["key"] in self.by_key
        ]


def main() -> None:
    from .build import load_cards

    args = sys.argv[1:]
    cards = load_cards()

    with VectorIndex(cards) as index:
        if args and args[0] == "build":
            import time

            t0 = time.perf_counter()
            index.build()
            print(f"готово за {time.perf_counter() - t0:.0f} c")
            return

        query = " ".join(args)
        if not query:
            print(__doc__)
            return
        for card, score in index.search(query):
            print(f"{score:.3f}  {card.key:22} {card.name[:60]}")


if __name__ == "__main__":
    main()
