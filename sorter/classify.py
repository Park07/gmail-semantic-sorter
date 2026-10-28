from __future__ import annotations

import re
import uuid
import numpy as np

from .categories import taxonomy_hash
from .config import SorterError, digest
from .models import REVIEW, Decision
from .ollama import embed_cached, normalise, valid_probability
from .reports import report
from .storage import now

ATTENTION_PATTERNS = [
    (
        r"\b(overdue|payment failed|payment declined|past due|final notice)\b",
        "Possible payment issue",
    ),
    (r"\b(deadline|due date|due (?:on|by)|submit by|submission closes)\b", "Possible deadline"),
    (
        r"\b(interview|please (?:reply|respond|confirm)|response required|action required)\b",
        "Possible response needed",
    ),
    (
        r"\b(unusual (?:login|sign.in)|security alert|password (?:changed|reset)|account suspended)\b",
        "Possible account/security issue",
    ),
]


def attention(message):
    text = message.subject + " " + message.snippet
    return [label for pattern, label in ATTENTION_PATTERNS if re.search(pattern, text, re.I)]


def rule_matches(message, categories):
    return [
        c.id
        for c in categories
        if message.sender in c.senders
        or any(message.domain == d or message.domain.endswith("." + d) for d in c.domains)
    ]


def training_examples(store, categories):
    overrides = store.overrides()
    allowed = {c.id for c in categories}
    result = {c.id: set(c.example_ids) for c in categories}
    # An explicit correction also removes that message from contradictory seed examples.
    for message_id, ids in overrides.items():
        for category_id in allowed:
            result[category_id].discard(message_id)
        for category_id in set(ids) & allowed:
            result[category_id].add(message_id)
    return {
        key: [m for i in sorted(ids) if (m := store.message(i)) and m.eligible]
        for key, ids in result.items()
    }


def classification_fingerprint(store, categories, settings):
    training = training_examples(store, categories)
    return digest(
        [
            taxonomy_hash(categories),
            store.learning_hash(),
            settings.fingerprint(),
            {k: [(m.id, m.content_hash) for m in v] for k, v in training.items()},
        ]
    )


def classify_messages(store, categories, client, messages, refresh=False):
    overrides, result, semantic = store.overrides(), {}, []
    allowed = {c.id for c in categories}
    base = classification_fingerprint(store, categories, client.settings)
    for message in messages:
        if message.id in overrides:
            ids = overrides[message.id]
            if set(ids) - allowed:
                result[message.id] = Decision(
                    message.id,
                    [REVIEW],
                    "manual-stale",
                    0,
                    "A manual correction refers to a retired category. Review it locally.",
                    attention(message),
                )
            else:
                result[message.id] = Decision(
                    message.id,
                    ids,
                    "manual",
                    1,
                    "Explicit correction or observed Gmail label change.",
                    attention(message),
                )
        elif matches := rule_matches(message, categories):
            result[message.id] = Decision(
                message.id,
                matches,
                "rule",
                1,
                "Matched an approved sender/domain rule.",
                attention(message),
            )
        else:
            semantic.append(message)
    if semantic:
        try:
            embedding_id = client.identity(client.settings.embedding_model)
            try:
                chat_id = client.identity(client.settings.chat_model)
            except SorterError:
                chat_id = "unavailable"
            fingerprint = digest([base, embedding_id, chat_id])
            pending = []
            for message in semantic:
                key = digest([fingerprint, message.content_hash])
                cached = store.cached_decision(message.id, key) if not refresh else None
                if cached and cached["method"] != "error":
                    result[message.id] = Decision(**cached)
                else:
                    pending.append(message)
            if pending:
                training = training_examples(store, categories)
                centres, supported = [], []
                for category in categories:
                    examples = training[category.id]
                    texts = [m.text for m in examples] or [
                        category.name + ": " + category.description
                    ]
                    vectors = embed_cached(client, store, texts, "classification: ")
                    centres.append(normalise(vectors.mean(axis=0)))
                    supported.append(bool(examples))
                mail_vectors = embed_cached(
                    client, store, [m.text for m in pending], "classification: "
                )
                similarities = mail_vectors @ np.asarray(centres).T
                for message, scores in zip(pending, similarities, strict=True):
                    decision = decide(message, categories, training, scores, supported, client)
                    result[message.id] = decision
                    store.save_decision(decision, digest([fingerprint, message.content_hash]))
        except SorterError:
            for message in semantic:
                if message.id not in result:
                    result[message.id] = Decision(
                        message.id,
                        [REVIEW],
                        "error",
                        0,
                        "Local embedding/model processing failed. Retry when Ollama is ready.",
                        attention(message),
                    )
    return [result[m.id] for m in messages]


def decide(message, categories, training, scores, supported, client):
    order = np.argsort(-scores)
    best = int(order[0])
    similarity = float(scores[best])
    runner_up = float(scores[order[1]]) if len(order) > 1 else 0.0
    settings = client.settings
    if (
        supported[best]
        and similarity >= settings.similarity_threshold
        and similarity - runner_up >= settings.similarity_margin
    ):
        return Decision(
            message.id,
            [categories[best].id],
            "centroid",
            similarity,
            "Above the configured similarity and separation thresholds.",
            attention(message),
        )
    candidates = [categories[int(i)] for i in order[:4]]
    lookup = {c.id: float(scores[categories.index(c)]) for c in candidates}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(lookup)},
                "maxItems": 3,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["category_ids", "confidence", "reason"],
    }
    try:
        value = client.chat(
            "Classify this email into zero to three of the candidate categories. Return an empty list when uncertain or none fits. Confidence is an advisory self-assessment, not a calibrated probability.",
            {
                "email": message.text[:600],
                "categories": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": c.description[:240],
                        "examples": [m.text[:120] for m in training[c.id][:2]],
                    }
                    for c in candidates
                ],
            },
            schema,
        )
        ids, confidence, reason = (
            value.get("category_ids"),
            value.get("confidence"),
            value.get("reason"),
        )
        if (
            not isinstance(ids, list)
            or len(ids) > 3
            or not all(isinstance(i, str) and i in lookup for i in ids)
            or not valid_probability(confidence)
            or not isinstance(reason, str)
        ):
            raise SorterError("Invalid classification response.")
        if (
            ids
            and confidence >= settings.llm_confidence_threshold
            and all(lookup[i] >= settings.fallback_similarity_floor for i in ids)
        ):
            return Decision(
                message.id,
                sorted(set(ids)),
                "llm",
                float(confidence),
                reason[:300],
                attention(message),
            )
        return Decision(
            message.id,
            [REVIEW],
            "review",
            float(confidence),
            "Ambiguous or below configured thresholds. " + reason[:220],
            attention(message),
        )
    except SorterError:
        return Decision(
            message.id,
            [REVIEW],
            "error",
            0,
            "Local classification failed or returned invalid IDs. No automatic Gmail write is permitted for this result.",
            attention(message),
        )


def preview(store, paths, categories, client, limit=1000, include_processed=False, refresh=False):
    messages = store.messages(limit, include_processed=include_processed)
    decisions = classify_messages(store, categories, client, messages, refresh)
    names = {c.id: c.label for c in categories} | {REVIEW: "AI/Review"}
    plan = {
        "id": uuid.uuid4().hex,
        "created": now(),
        "account": store.meta("account"),
        "taxonomy": taxonomy_hash(categories),
        "settings": client.settings.fingerprint(),
        "classification": classification_fingerprint(store, categories, client.settings),
        "items": [],
    }
    rows = []
    for message, decision in zip(messages, decisions, strict=True):
        labels = [names[i] for i in decision.category_ids]
        plan["items"].append(
            {**decision.as_dict(), "labels": labels, "content_hash": message.content_hash}
        )
        rows.append(
            {
                "message_id": message.id,
                "subject": message.subject,
                "sender": message.sender,
                "snippet": message.snippet,
                "proposed_labels": labels,
                "method": decision.method,
                "score_not_probability": round(decision.score, 4),
                "reason": decision.reason,
                "attention_suggestions": decision.attention,
            }
        )
    store.save_plan(plan)
    summary = {
        "plan_id": plan["id"],
        "messages": len(rows),
        "review": sum(REVIEW in d.category_ids for d in decisions),
        "errors_not_writable": sum(d.method in {"error", "manual-stale"} for d in decisions),
        "note": "Similarity and model self-confidence are different scores, neither is a calibrated probability. Attention suggestions are heuristic and may miss important mail.",
    }
    location = report(paths, "preview-" + plan["id"], "Gmail label preview", rows, summary)
    return {**summary, "report": str(location)}
