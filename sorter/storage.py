from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from .config import SorterError, digest
from .models import Message


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, internal_date INTEGER NOT NULL,
            payload TEXT NOT NULL, eligible INTEGER NOT NULL, content_hash TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS messages_date ON messages(internal_date DESC);
        CREATE TABLE IF NOT EXISTS pending (id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS overrides (
            id TEXT PRIMARY KEY, categories TEXT NOT NULL, source TEXT NOT NULL, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS writes (
            id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, labels TEXT NOT NULL,
            status TEXT NOT NULL, updated TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def bind_account(self, email):
        email = email.lower()
        bound = self.meta("account")
        if bound and bound != email:
            raise SorterError(
                "This project is bound to a different Gmail account. Use a separate project directory."
            )
        self.set_meta("account", email)

    def enqueue(self, ids, checkpoint=None):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO pending VALUES (?)", ((i,) for i in ids))
            if checkpoint:
                self.db.execute(
                    "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                    (checkpoint[0], json.dumps(checkpoint[1])),
                )

    def pending(self, limit=50):
        return [
            r[0] for r in self.db.execute("SELECT id FROM pending ORDER BY rowid LIMIT ?", (limit,))
        ]

    def save_message(self, message: Message):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?)",
                (
                    message.id,
                    message.thread_id,
                    message.internal_date,
                    json.dumps(message.as_dict()),
                    int(message.eligible),
                    message.content_hash,
                ),
            )
            self.db.execute("DELETE FROM pending WHERE id=?", (message.id,))

    def missing(self, message_id):
        msg = self.message(message_id)
        if msg:
            msg.unavailable = True
            self.save_message(msg)
        with self.db:
            self.db.execute("DELETE FROM pending WHERE id=?", (message_id,))

    def message(self, message_id):
        row = self.db.execute("SELECT payload FROM messages WHERE id=?", (message_id,)).fetchone()
        return Message(**json.loads(row[0])) if row else None

    def messages(self, limit=None, include_processed=True):
        sql = "SELECT payload FROM messages WHERE eligible=1"
        if not include_processed:
            sql += " AND id NOT IN (SELECT id FROM writes)"
        sql += " ORDER BY internal_date DESC, id"
        params = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [Message(**json.loads(r[0])) for r in self.db.execute(sql, params)]

    def vector(self, key):
        row = self.db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_vector(self, key, vector):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?)", (key, json.dumps(vector))
            )

    def overrides(self):
        return {
            r["id"]: json.loads(r["categories"]) for r in self.db.execute("SELECT * FROM overrides")
        }

    def override(self, message_id, category_ids, source="explicit"):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO overrides VALUES (?,?,?,?)",
                (message_id, json.dumps(sorted(set(category_ids))), source, now()),
            )

    def learning_hash(self):
        return digest(self.overrides())

    def cached_decision(self, message_id, fingerprint):
        row = self.db.execute(
            "SELECT payload FROM decisions WHERE id=? AND fingerprint=?", (message_id, fingerprint)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_decision(self, decision, fingerprint):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?,?,?)",
                (decision.message_id, fingerprint, json.dumps(decision.as_dict())),
            )

    def save_plan(self, plan):
        with self.db:
            self.db.execute("INSERT INTO plans VALUES (?,?)", (plan["id"], json.dumps(plan)))
        self.set_meta("latest_plan", plan["id"])

    def plan(self, plan_id=None):
        plan_id = plan_id or self.meta("latest_plan")
        row = self.db.execute("SELECT payload FROM plans WHERE id=?", (plan_id,)).fetchone()
        if not row:
            raise SorterError("No saved preview plan. Run classify first.")
        return json.loads(row[0])

    def write_record(self, message_id):
        row = self.db.execute("SELECT * FROM writes WHERE id=?", (message_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["labels"] = json.loads(result["labels"])
        return result

    def record_write(self, message_id, plan_id, labels, status):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO writes VALUES (?,?,?,?,?)",
                (message_id, plan_id, json.dumps(labels), status, now()),
            )

    def incomplete_writes(self):
        return [
            self.write_record(r[0])
            for r in self.db.execute("SELECT id FROM writes WHERE status IN ('pending','held')")
        ]

    def counts(self):
        return {
            "messages": self.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "eligible": self.db.execute(
                "SELECT COUNT(*) FROM messages WHERE eligible=1"
            ).fetchone()[0],
            "pending": self.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0],
            "writes": {
                r[0]: r[1]
                for r in self.db.execute("SELECT status,COUNT(*) FROM writes GROUP BY status")
            },
        }
