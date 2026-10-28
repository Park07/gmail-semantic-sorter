from __future__ import annotations

from collections import defaultdict

from .categories import taxonomy_hash
from .classify import classification_fingerprint
from .config import SorterError
from .gmail import GmailError
from .ingest import connect_account
from .models import REVIEW, validate_label, validate_message_id


def validate_plan(plan, store, categories, settings):
    if store.meta("approved_taxonomy") != taxonomy_hash(categories):
        raise SorterError("Approve the categories before applying labels.")
    if not store.meta("account") or plan.get("account") != store.meta("account"):
        raise SorterError("The preview is not bound to the connected personal Gmail account.")
    if (
        plan.get("taxonomy") != taxonomy_hash(categories)
        or plan.get("settings") != settings.fingerprint()
    ):
        raise SorterError("Categories or settings changed after this preview. Run classify again.")
    if plan.get("classification") != classification_fingerprint(store, categories, settings):
        raise SorterError("Examples or corrections changed after this preview. Run classify again.")
    names = {c.id: c.label for c in categories} | {REVIEW: "AI/Review"}
    seen = set()
    for item in plan.get("items", []):
        message_id = validate_message_id(item.get("message_id"))
        if message_id in seen:
            raise SorterError("A preview contains a duplicate message ID.")
        seen.add(message_id)
        ids = item.get("category_ids")
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in names for i in ids):
            raise SorterError("A preview contains an unapproved category.")
        if len(ids) != len(set(ids)) or (REVIEW in ids and len(ids) != 1):
            raise SorterError("A preview has conflicting categories.")
        if item.get("labels") != [names[i] for i in ids]:
            raise SorterError("A preview label does not match its approved category.")
        for label in item["labels"]:
            validate_label(label)
        message = store.message(message_id)
        if not message or item.get("content_hash") != message.content_hash:
            raise SorterError(
                "Imported email content changed after this preview. Run classify again."
            )


def apply_plan(
    store, categories, settings, plan, gmail=None, apply=False, limit=100, retry_held=False
):
    if type(limit) is not int or limit <= 0:
        raise SorterError("Apply limit must be a positive integer.")
    validate_plan(plan, store, categories, settings)
    counts = {
        "applied": 0,
        "already_processed": 0,
        "protected": 0,
        "stale_or_missing": 0,
        "model_errors": 0,
        "dry_run": not apply,
    }
    candidates = []
    for item in plan["items"]:
        if item["method"] in {"error", "manual-stale"}:
            counts["model_errors"] += 1
            continue
        record = store.write_record(item["message_id"])
        if record:
            if not (
                retry_held
                and record["status"] == "held"
                and set(record["labels"]) == set(item["labels"])
            ):
                counts["already_processed"] += 1
                continue
        if not item["labels"]:
            counts["protected"] += 1
            continue
        candidates.append(item)
        if len(candidates) == limit:
            break
    counts["selected"] = len(candidates)
    if not apply:
        return counts
    if gmail is None or not gmail.write_allowed:
        raise SorterError("Applying labels requires an explicitly enabled Gmail writer.")
    connect_account(gmail, store)
    if not candidates:
        return counts
    labels = gmail.labels()
    id_to_name = {label["id"]: label["name"] for label in labels}
    current = gmail.messages([item["message_id"] for item in candidates])
    safe = []
    # Validate all selected messages and targets before any label is created.
    for item in candidates:
        message = current[item["message_id"]]
        if not message or not message.eligible or message.content_hash != item["content_hash"]:
            counts["stale_or_missing"] += 1
            continue
        existing = {
            id_to_name[i]
            for i in message.labels
            if i in id_to_name and id_to_name[i].startswith("AI/")
        }
        recorded = store.write_record(message.id)
        permitted_partial = (
            retry_held
            and recorded
            and recorded["status"] == "held"
            and existing <= set(item["labels"])
        )
        if existing and not permitted_partial:
            counts["protected"] += 1
            store.record_write(message.id, plan["id"], sorted(existing), "manual")
            store.save_message(message)
            continue
        safe.append(item)
    name_to_id = {label["name"]: label["id"] for label in labels if label.get("type") == "user"}
    for name in sorted({name for item in safe for name in item["labels"]}):
        if name not in name_to_id:
            try:
                created = gmail.create_ai_label(name)
            except GmailError as exc:
                if exc.status != 409:
                    raise
                created = next(
                    (
                        x
                        for x in gmail.labels()
                        if x.get("type") == "user" and x.get("name") == name
                    ),
                    None,
                )
                if not created:
                    raise SorterError("A label creation conflict needs a fresh preview.") from exc
            name_to_id[name] = created["id"]
    groups = defaultdict(list)
    for item in safe:
        groups[tuple(sorted(item["labels"]))].append(item["message_id"])
    for names, ids in groups.items():
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            for message_id in batch:
                store.record_write(message_id, plan["id"], list(names), "pending")
            # A timeout may mean Gmail accepted the request. Pending records block
            # automatic retries until read-only recovery determines what happened.
            gmail.add_labels(batch, [name_to_id[name] for name in names])
            for message_id in batch:
                store.record_write(message_id, plan["id"], list(names), "done")
                message = current[message_id]
                message.labels = sorted(set(message.labels) | {name_to_id[name] for name in names})
                store.save_message(message)
                counts["applied"] += 1
    return counts


def recover(store, gmail):
    connect_account(gmail, store)
    records = store.incomplete_writes()
    names = {label["id"]: label["name"] for label in gmail.labels()}
    found = gmail.messages([r["id"] for r in records]) if records else {}
    counts = {"confirmed": 0, "held_for_review": 0, "gmail_changed": False}
    for record in records:
        message = found[record["id"]]
        present = {names[i] for i in message.labels if i in names} if message else set()
        status = "done" if set(record["labels"]) <= present else "held"
        store.record_write(record["id"], record["plan_id"], record["labels"], status)
        if message:
            store.save_message(message)
        counts["confirmed" if status == "done" else "held_for_review"] += 1
    return counts
