from __future__ import annotations

import re
import yaml

from .config import SorterError, atomic_write, digest
from .models import Category


def parse_categories(document, store=None):
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise SorterError("Category files require schema_version: 1.")
    raw = document.get("categories")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 50:
        raise SorterError(
            "Approve between 1 and 50 categories. Edit the proposals or try another clustering method."
        )
    result, seen_ids, seen_labels = [], set(), set()
    allowed = {"id", "name", "description", "example_ids", "senders", "domains"}
    for item in raw:
        if not isinstance(item, dict) or set(item) - allowed:
            raise SorterError("Invalid category fields.")
        try:
            category = Category(**{**{"example_ids": [], "senders": [], "domains": []}, **item})
        except TypeError as exc:
            raise SorterError("Each category needs id, name, and description.") from exc
        if not isinstance(category.id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{0,63}", category.id
        ):
            raise SorterError(
                "Category IDs must use lowercase letters, numbers, underscores, or hyphens."
            )
        if not isinstance(category.name, str) or category.name.casefold() == "review":
            raise SorterError("Review is a reserved category name.")
        label = category.label
        if (
            not isinstance(category.description, str)
            or not 1 <= len(category.description.strip()) <= 500
        ):
            raise SorterError("Each category needs a description of 1 to 500 characters.")
        if category.id in seen_ids or label.casefold() in seen_labels:
            raise SorterError("Category IDs and label names must be unique.")
        for field in (category.example_ids, category.senders, category.domains):
            if not isinstance(field, list) or not all(isinstance(v, str) for v in field):
                raise SorterError("Examples, senders, and domains must be lists of strings.")
        category.example_ids = list(dict.fromkeys(category.example_ids))
        if len(category.example_ids) > 500:
            raise SorterError("A category can have at most 500 example IDs.")
        if store:
            for message_id in category.example_ids:
                message = store.message(message_id)
                if not message or not message.eligible:
                    raise SorterError(
                        "A category example is missing or ineligible. Remove that example ID."
                    )
        category.senders = [s.strip().lower() for s in category.senders]
        category.domains = [s.strip().lower() for s in category.domains]
        if any(not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", s) for s in category.senders):
            raise SorterError("Sender rules must be complete email addresses.")
        if any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}", d)
            for d in category.domains
        ):
            raise SorterError("Domain rules must be domain names without wildcards.")
        seen_ids.add(category.id)
        seen_labels.add(label.casefold())
        result.append(category)
    return result


def read_categories(path, store=None):
    try:
        return parse_categories(yaml.safe_load(path.read_text()), store)
    except (OSError, yaml.YAMLError) as exc:
        raise SorterError("Cannot read the category YAML file.") from exc


def document(categories):
    return {"schema_version": 1, "categories": [c.as_dict() for c in categories]}


def taxonomy_hash(categories):
    return digest(document(categories))


def approve(source, paths, store):
    categories = read_categories(source, store)
    if paths.categories.exists():
        previous = paths.categories.read_text()
        atomic_write(paths.data / "taxonomies" / (digest(previous) + ".yaml"), previous)
    atomic_write(
        paths.categories, yaml.safe_dump(document(categories), sort_keys=False, allow_unicode=True)
    )
    store.set_meta("approved_taxonomy", taxonomy_hash(categories))
    return categories


def approved(paths, store):
    # Previously approved example messages may since have been removed from Gmail.
    # The classifier ignores unavailable examples; this must not invalidate names.
    categories = read_categories(paths.categories)
    if taxonomy_hash(categories) != store.meta("approved_taxonomy"):
        raise SorterError(
            "Categories changed or have not been approved. Review them, then run approve."
        )
    return categories


def resolve_ids(values, categories):
    lookup = {c.id: c.id for c in categories}
    lookup.update({c.name: c.id for c in categories})
    lookup.update({c.label: c.id for c in categories})
    try:
        return sorted({lookup[value] for value in values})
    except KeyError as exc:
        raise SorterError("Unknown category. Use an approved category ID, name, or label.") from exc
