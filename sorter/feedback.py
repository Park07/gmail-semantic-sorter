from .categories import resolve_ids
from .config import SorterError


def correction(store, message_id, values, categories):
    if not store.message(message_id):
        raise SorterError("Import this message before recording a correction.")
    ids = resolve_ids(values, categories)
    store.override(message_id, ids)
    return {"message_id": message_id, "category_ids": ids, "gmail_changed": False}


def observer(store, categories, labels):
    label_names = {label["id"]: label["name"] for label in labels}
    approved = {category.label: category.id for category in categories}

    def observe(message):
        recorded = store.write_record(message.id)
        if not recorded or recorded["status"] != "done":
            return
        row = store.db.execute("SELECT source FROM overrides WHERE id=?", (message.id,)).fetchone()
        if row and row[0] == "explicit":
            return
        current = {
            label_names[i]
            for i in message.labels
            if i in label_names and label_names[i].startswith("AI/")
        }
        if current != set(recorded["labels"]):
            # Empty is a meaningful correction: the user removed all sorter labels.
            store.override(
                message.id, [approved[name] for name in current if name in approved], "gmail"
            )

    return observe
