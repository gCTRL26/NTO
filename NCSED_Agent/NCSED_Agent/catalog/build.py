"""Этап 2 загрузчика: сборка карточек каталога.

Читает метаданные источника и кэш фактов, полученный на этапе scan, и складывает
их в DatasetCard. Ничего не открывает и не считает, поэтому перезапускается
за секунды — можно сколько угодно менять состав поисковых текстов.

Запуск:  python -m NCSED_Agent.catalog.build
"""

import time
from collections.abc import Iterator
from pathlib import Path

from ..domain.dataset_card import DatasetCard
from . import fedstat, worldbank
from .scan import CACHE_ROOT, FileFacts, load_facts, save_facts, scan_source

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "registry" / "cards.jsonl"

# Ключ здесь — префикс ключа карточки: он же используется в FileFacts.key,
# поэтому менять его нельзя, не пересобрав кэш скана.
SOURCES: dict[str, object] = {"wb": worldbank, "fedstat": fedstat}


def load_cards(path: Path = REGISTRY_PATH) -> list[DatasetCard]:
    """Читает готовый каталог. Обратная операция к main(), поэтому живёт здесь же."""
    with open(path, encoding="utf-8") as f:
        return [DatasetCard.model_validate_json(line) for line in f]


def get_facts(name: str, source, refresh: bool = False) -> dict[str, FileFacts]:
    """Возвращает факты по файлам источника, сканируя их только при необходимости.

    Скан дорогой (World Bank — около 4.5 минут), а его результат не меняется,
    пока не менялись сами файлы. Поэтому по умолчанию берём из кэша.
    """
    cache_path = CACHE_ROOT / f"scan_{name}.jsonl"
    if refresh or not cache_path.exists():
        print(f"[{name}] кэша нет, сканирую файлы...")
        t0 = time.perf_counter()
        n = save_facts(scan_source(source), cache_path)
        print(f"[{name}] отсканировано {n} за {time.perf_counter() - t0:.1f} c")
    return load_facts(cache_path)


def build_source(name: str, source, facts: dict[str, FileFacts]) -> Iterator[DatasetCard]:
    """Собирает карточки одного источника.

    Показатели, для которых нет фактов, пропускаются: значит скан их не видел,
    и мы не знаем ни статуса, ни покрытия. Молча выдумывать нельзя.
    """
    missing = 0
    for code, meta in source.iter_metadata():
        f = facts.get(f"{name}:{code}")
        if f is None:
            missing += 1
            continue
        yield source.to_card(code, meta, f)
    if missing:
        print(f"[{name}] пропущено без фактов: {missing}")


def main(refresh: bool = False) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.perf_counter()

    with open(REGISTRY_PATH, "w", encoding="utf-8") as out:
        for name, source in SOURCES.items():
            facts = get_facts(name, source, refresh=refresh)
            n = 0
            for card in build_source(name, source, facts):
                out.write(card.model_dump_json() + "\n")
                n += 1
            print(f"[{name}] карточек: {n}")
            total += n

    print(f"\nВсего {total} карточек за {time.perf_counter() - t0:.1f} c")
    print(f"Каталог: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
