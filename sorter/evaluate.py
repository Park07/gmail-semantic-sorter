from __future__ import annotations

import csv
from collections import Counter

from .categories import resolve_ids
from .classify import classify_messages, training_examples
from .config import SorterError
from .models import REVIEW
from .reports import report


def training_threads(store, categories):
    threads = {
        m.thread_id for examples in training_examples(store, categories).values() for m in examples
    }
    threads.update(m.thread_id for i in store.overrides() if (m := store.message(i)))
    return threads


def export_sample(store, paths, categories, count=100):
    excluded = training_threads(store, categories)
    seen, sample = set(), []
    # Prefer recent unseen threads as a temporal holdout, not training examples.
    for message in store.messages():
        if message.thread_id in excluded or message.thread_id in seen:
            continue
        seen.add(message.thread_id)
        sample.append(
            {
                "message_id": message.id,
                "thread_id": message.thread_id,
                "sender": message.sender,
                "subject": message.subject,
                "snippet": message.snippet,
                "expected_category_ids": "",
            }
        )
        if len(sample) == count:
            break
    if not sample:
        raise SorterError("No unseen threads are available for evaluation. Import more mail.")
    location = report(
        paths,
        "evaluation-sample",
        "Label these emails before evaluating",
        sample,
        {
            "instructions": "Fill expected_category_ids in the CSV with semicolon-separated category IDs; use REVIEW for unknown. Predictions are intentionally omitted.",
            "category_ids": {c.id: c.name for c in categories},
            "sample_size": len(sample),
        },
    )
    return {
        "report": str(location),
        "csv": str(paths.reports / "evaluation-sample.csv"),
        "sample_size": len(sample),
    }


def evaluate(store, paths, categories, client, csv_path):
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise SorterError("Cannot read the evaluation CSV.") from exc
    excluded, seen, messages, expected = training_threads(store, categories), set(), [], []
    for row in rows:
        truth = row.get("expected_category_ids", "").strip()
        if not truth:
            continue
        message = store.message(row.get("message_id"))
        if not message or not message.eligible or message.id in seen:
            raise SorterError("Evaluation contains a missing, duplicate, or ineligible message.")
        if message.thread_id in excluded:
            raise SorterError("Evaluation overlaps training threads. Export a new holdout sample.")
        seen.add(message.id)
        ids = (
            [REVIEW]
            if truth == "REVIEW"
            else resolve_ids([s.strip() for s in truth.split(";")], categories)
        )
        messages.append(message)
        expected.append(set(ids))
    if not messages:
        raise SorterError("Fill at least one expected_category_ids cell before evaluation.")
    predictions = classify_messages(store, categories, client, messages, refresh=True)
    exact = sum(set(p.category_ids) == e for p, e in zip(predictions, expected, strict=True))
    automatic = [
        (p, e) for p, e in zip(predictions, expected, strict=True) if REVIEW not in p.category_ids
    ]
    confusion = Counter(
        (";".join(sorted(e)), ";".join(sorted(p.category_ids)))
        for p, e in zip(predictions, expected, strict=True)
    )
    metrics = []
    for category_id in [c.id for c in categories] + [REVIEW]:
        tp = sum(
            category_id in p.category_ids and category_id in e
            for p, e in zip(predictions, expected, strict=True)
        )
        fp = sum(
            category_id in p.category_ids and category_id not in e
            for p, e in zip(predictions, expected, strict=True)
        )
        fn = sum(
            category_id not in p.category_ids and category_id in e
            for p, e in zip(predictions, expected, strict=True)
        )
        precision, recall = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
        metrics.append(
            {
                "category_id": category_id,
                "support": tp + fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            }
        )
    summary = {
        "evaluated": len(messages),
        "exact_match_accuracy": exact / len(messages),
        "automatic_coverage": len(automatic) / len(messages),
        "automatic_exact_match_precision": sum(set(p.category_ids) == e for p, e in automatic)
        / len(automatic)
        if automatic
        else None,
        "note": "This pilot does not establish calibration or reliable accuracy for rare categories. Review missed important mail separately.",
    }
    report(paths, "evaluation-categories", "Per-category evaluation", metrics, summary)
    report(
        paths,
        "evaluation-confusion",
        "Confusion table (label sets)",
        [{"expected_ids": a, "predicted_ids": b, "count": n} for (a, b), n in confusion.items()],
        summary,
    )
    location = report(
        paths,
        "evaluation",
        "Evaluation results",
        [
            {
                "message_id": m.id,
                "subject": m.subject,
                "expected": sorted(e),
                "predicted": p.category_ids,
                "correct": set(p.category_ids) == e,
                "method": p.method,
                "reason": p.reason,
            }
            for m, p, e in zip(messages, predictions, expected, strict=True)
        ],
        summary,
    )
    store.set_meta("last_evaluation", summary)
    return {**summary, "report": str(location)}
