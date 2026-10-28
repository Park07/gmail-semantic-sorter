from __future__ import annotations

from collections import Counter
import re
import uuid

import numpy as np
import yaml
from sklearn.cluster import HDBSCAN, KMeans

from .categories import document
from .config import SorterError, atomic_write
from .models import Category
from .ollama import embed_cached
from .reports import report


NAME_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
    "required": ["name", "description"],
}


def sample_messages(messages, limit):
    # One latest message per thread, then spread the sample over available dates.
    threads = {}
    for message in sorted(messages, key=lambda m: m.internal_date, reverse=True):
        threads.setdefault(message.thread_id, message)
    ordered = sorted(threads.values(), key=lambda m: (m.internal_date, m.id))
    if len(ordered) <= limit:
        return ordered
    return [ordered[i] for i in np.linspace(0, len(ordered) - 1, limit, dtype=int)]


def discover(store, paths, client, limit=1000, method="hdbscan", clusters=10):
    messages = sample_messages(store.messages(), limit)
    if len(messages) < 2:
        raise SorterError(
            "Import at least two distinct email threads before discovering categories."
        )
    vectors = embed_cached(client, store, [m.text for m in messages], "clustering: ")
    settings = client.settings
    if method == "kmeans":
        labels = KMeans(
            n_clusters=min(clusters, len(messages)), random_state=42, n_init=10
        ).fit_predict(vectors)
    elif len(messages) < settings.min_cluster_size:
        labels = np.full(len(messages), -1)
    else:
        labels = HDBSCAN(
            min_cluster_size=settings.min_cluster_size,
            min_samples=min(settings.min_samples, len(messages)),
            metric="euclidean",
        ).fit_predict(vectors)
    categories, rows, used_names, warnings = [], [], set(), []
    for cluster in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == cluster)
        centroid = vectors[indices].mean(axis=0)
        representative = indices[np.argsort(-(vectors[indices] @ centroid))[:10]]
        examples = [messages[i] for i in representative]
        name, description = (
            f"Group {len(categories) + 1}",
            "Edit this description after reviewing the sample messages.",
        )
        try:
            proposed = client.chat(
                "Name this email group with a short topic label and one sentence explaining what belongs. Prefer useful topics over sender company names. Avoid urgency or action-status names.",
                {"samples": [m.text[:330] for m in examples[:8]]},
                NAME_SCHEMA,
            )
            if not isinstance(proposed.get("name"), str) or not isinstance(
                proposed.get("description"), str
            ):
                raise SorterError("Invalid category name response.")
            candidate = re.sub(r"[/\x00-\x1f\x7f]", " ", proposed["name"]).strip()[:65]
            if (
                not candidate
                or candidate.casefold() == "review"
                or not proposed["description"].strip()
            ):
                raise SorterError("Invalid category name response.")
            name, description = candidate, proposed["description"].strip()[:500]
        except SorterError:
            warnings.append(
                f"Group {len(categories) + 1} needs a name; the local naming model was unavailable or returned invalid data."
            )
        base, number = name, 2
        while name.casefold() in used_names:
            name = f"{base} ({number})"
            number += 1
        used_names.add(name.casefold())
        category = Category(
            "c_" + uuid.uuid4().hex[:12], name, description, [m.id for m in examples], [], []
        )
        categories.append(category)
        domains = Counter(messages[i].domain for i in indices).most_common(5)
        for message in examples:
            rows.append(
                {
                    "category_id": category.id,
                    "proposed_name": name,
                    "description": description,
                    "cluster_size": len(indices),
                    "top_domains": domains,
                    "message_id": message.id,
                    "sender": message.sender,
                    "subject": message.subject,
                    "snippet": message.snippet,
                }
            )
    # Every sampled outlier is reported, including rare messages that may matter.
    for i in np.flatnonzero(labels == -1):
        message = messages[i]
        rows.append(
            {
                "category_id": "unclustered",
                "proposed_name": "Review unclustered messages",
                "description": "Rare does not mean unimportant.",
                "cluster_size": 1,
                "message_id": message.id,
                "sender": message.sender,
                "subject": message.subject,
                "snippet": message.snippet,
            }
        )
    pairs = []
    if len(categories) > 1:
        descriptions = embed_cached(
            client, store, [c.description for c in categories], "clustering: "
        )
        similarities = descriptions @ descriptions.T
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                if similarities[i, j] >= 0.90:
                    pairs.append(
                        [
                            categories[i].name,
                            categories[j].name,
                            round(float(similarities[i, j]), 3),
                        ]
                    )
    atomic_write(
        paths.proposals, yaml.safe_dump(document(categories), sort_keys=False, allow_unicode=True)
    )
    summary = {
        "sampled_threads": len(messages),
        "proposed_categories": len(categories),
        "unclustered": int(sum(labels == -1)),
        "method": method,
        "possible_merges_for_review": pairs,
        "warnings": warnings,
        "next": "Edit data/categories.proposed.yaml, then run approve. Existing approved categories are unchanged.",
    }
    location = report(paths, "discovery", "Discovered category proposals", rows, summary)
    return {**summary, "report": str(location), "proposals": str(paths.proposals)}
