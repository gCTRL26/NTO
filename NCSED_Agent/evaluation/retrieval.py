"""Прогон тест-набора по поиску и подсчёт метрик.

Запуск:
    python -m evaluation.retrieval
"""

from collections import defaultdict

from NCSED_Agent.catalog.bm25 import BM25Index
from NCSED_Agent.catalog.build import load_cards

from .testset import SearchCase, load_cases, validate_cases

K = 10


def evaluate_case(index, case: SearchCase, k: int = K) -> dict:
    """Считает метрики одного кейса.

    recall  — какая доля обязательных карточек попала в топ-k;
    success — попала ли хоть одна (пользователь получил что-то полезное);
    rr      — 1/позиция первой найденной (насколько высоко в выдаче).
    """
    found_keys = [card.key for card, _ in index.search(case.query, k=k)]

    if case.kind == "not_found":
        # Правильное поведение — не найти ничего. Метрика обратная.
        return {"n_found": len(found_keys), "correct_silence": not found_keys}

    if not case.must_find:
        # Кейсы вроде ambiguous: эталон по карточкам не задан, метрику не считаем.
        return {"n_found": len(found_keys), "skipped": True}

    hits = [key for key in case.must_find if key in found_keys]
    rr = 0.0
    for position, key in enumerate(found_keys, start=1):
        if key in case.must_find:
            rr = 1 / position
            break

    return {
        "recall": len(hits) / len(case.must_find),
        "success": bool(hits),
        "rr": rr,
        "n_found": len(found_keys),
    }


def run(index, cases: list[SearchCase], k: int = K, label: str = "") -> None:
    per_kind: dict[str, list[dict]] = defaultdict(list)

    print(f"\n{'=' * 78}\n{label}  (k={k})\n{'=' * 78}")
    for case in cases:
        m = evaluate_case(index, case, k)
        per_kind[case.kind].append(m)

        if case.kind == "not_found":
            mark = "OK " if m["correct_silence"] else "ПЛОХО"
            print(f"  {mark} [{case.kind}] вернулось {m['n_found']} карточек — {case.query[:48]}")
        elif m.get("skipped"):
            print(f"  --  [{case.kind}] эталон не задан — {case.query[:48]}")
        else:
            mark = "OK " if m["success"] else "МИМО"
            print(f"  {mark} [{case.kind}] recall={m['recall']:.2f} rr={m['rr']:.2f} — {case.query[:48]}")

    print(f"\n  {'тип кейса':16} {'кейсов':>7} {'recall@k':>10} {'success@k':>10} {'MRR':>7}")
    scored = []
    for kind, rows in sorted(per_kind.items()):
        graded = [r for r in rows if "recall" in r]
        if not graded:
            if kind == "not_found":
                ok = sum(r["correct_silence"] for r in rows)
                print(f"  {kind:16} {len(rows):7} {'молчит ' + str(ok) + '/' + str(len(rows)):>10}")
            else:
                print(f"  {kind:16} {len(rows):7} {'—':>10}")
            continue
        scored += graded
        rec = sum(r["recall"] for r in graded) / len(graded)
        suc = sum(r["success"] for r in graded) / len(graded)
        mrr = sum(r["rr"] for r in graded) / len(graded)
        print(f"  {kind:16} {len(graded):7} {rec:10.2f} {suc:10.2f} {mrr:7.2f}")

    if scored:
        rec = sum(r["recall"] for r in scored) / len(scored)
        suc = sum(r["success"] for r in scored) / len(scored)
        mrr = sum(r["rr"] for r in scored) / len(scored)
        print(f"  {'ИТОГО':16} {len(scored):7} {rec:10.2f} {suc:10.2f} {mrr:7.2f}")


def main() -> None:
    cards = load_cards()
    cases = load_cases()
    validate_cases(cases, {c.key for c in cards})
    print(f"каталог: {len(cards)} карточек, тест-набор: {len(cases)} кейсов")

    run(BM25Index(cards), cases, label="BM25 (лексический поиск)")


if __name__ == "__main__":
    main()
