from __future__ import annotations

from .config import digest
from .gmail import GmailError
from .storage import now


def connect_account(gmail, store):
    profile = gmail.profile()
    store.bind_account(profile["emailAddress"])
    return profile


def drain(gmail, store, limit, on_message=None):
    count = 0
    while count < limit:
        ids = store.pending(min(50, limit - count))
        if not ids:
            break
        results = gmail.messages(ids)
        for message_id in ids:
            message = results[message_id]
            if message is None:
                store.missing(message_id)
            else:
                if on_message:
                    on_message(message)
                store.save_message(message)
            count += 1
    return count


def ingest(gmail, store, query, limit=1000, restart=False, on_message=None):
    profile = connect_account(gmail, store)
    if not store.meta("history_cursor"):
        store.set_meta("history_cursor", profile["historyId"])
    key = "import:" + digest(query)
    cursor = None if restart else store.meta(key)
    count = drain(gmail, store, limit, on_message)
    while count < limit and cursor != "COMPLETE":
        page = gmail.list_page(query, cursor, min(100, limit - count))
        cursor = page.get("nextPageToken", "COMPLETE")
        store.enqueue([m["id"] for m in page.get("messages", [])], (key, cursor))
        count += drain(gmail, store, limit - count, on_message)
    return {"imported": count, "complete": cursor == "COMPLETE", **store.counts()}


def sync(gmail, store, query, limit=1000, on_message=None):
    profile = connect_account(gmail, store)
    cursor = store.meta("history_cursor")
    recovery = False
    if not cursor:
        return ingest(gmail, store, query, limit, on_message=on_message)
    page_token = None
    try:
        while True:
            page = gmail.history_page(cursor, page_token)
            ids = set()
            for record in page.get("history", []):
                for field in ("messagesAdded", "messagesDeleted", "labelsAdded", "labelsRemoved"):
                    ids.update(item["message"]["id"] for item in record.get(field, []))
            store.enqueue(sorted(ids))
            page_token = page.get("nextPageToken")
            if not page_token:
                store.set_meta("history_cursor", page["historyId"])
                break
    except GmailError as exc:
        if exc.status != 404:
            raise
        # Keep the old history checkpoint until the entire scope has been queued.
        # A crash repeats listing safely; unprocessed IDs remain durable in pending.
        recovery = True
        store.enqueue([m.id for m in store.messages()])
        page_token = None
        while True:
            page = gmail.list_page(query, page_token, 500)
            store.enqueue([m["id"] for m in page.get("messages", [])])
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        store.set_meta("history_cursor", profile["historyId"])
    count = drain(gmail, store, limit, on_message)
    store.set_meta("last_sync", now())
    return {"refreshed": count, "history_recovered": recovery, **store.counts()}
