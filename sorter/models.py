from __future__ import annotations

from dataclasses import asdict, dataclass
from email.utils import parseaddr
import re

from .config import SorterError, digest

REVIEW = "__review__"
EXCLUDED = {"SENT", "DRAFT", "SPAM", "TRASH"}
PREPROCESS_VERSION = "domain-subject-snippet-v1"


def validate_label(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"AI/[^/\x00-\x1f\x7f]{1,80}", name):
        raise SorterError(
            "Labels must be a single nonempty name under AI/ (maximum 80 characters)."
        )
    if name != name.strip() or name[3:] != name[3:].strip():
        raise SorterError("Label names cannot have surrounding whitespace.")
    return name


def validate_message_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise SorterError("Invalid Gmail message ID.")
    return value


@dataclass
class Message:
    id: str
    thread_id: str
    internal_date: int
    sender: str
    domain: str
    subject: str
    snippet: str
    headers: dict
    labels: list[str]
    unavailable: bool = False

    @classmethod
    def from_gmail(cls, raw):
        headers = {
            h["name"].lower(): h.get("value", "") for h in raw.get("payload", {}).get("headers", [])
        }
        sender = parseaddr(headers.get("from", ""))[1].lower()
        return cls(
            validate_message_id(raw["id"]),
            raw.get("threadId", raw["id"]),
            int(raw.get("internalDate", 0)),
            sender,
            sender.rsplit("@", 1)[-1],
            headers.get("subject", "")[:2000],
            raw.get("snippet", "")[:2000],
            headers,
            list(raw.get("labelIds", [])),
        )

    @property
    def eligible(self):
        return not self.unavailable and not EXCLUDED.intersection(self.labels)

    @property
    def text(self):
        return f"From: {self.domain[:253]} | Subject: {self.subject[:200]} | {self.snippet[:300]}"

    @property
    def content_hash(self):
        return digest([self.sender, self.subject, self.snippet, self.internal_date])

    def as_dict(self):
        return asdict(self)


@dataclass
class Category:
    id: str
    name: str
    description: str
    example_ids: list[str]
    senders: list[str]
    domains: list[str]

    @property
    def label(self):
        return validate_label("AI/" + self.name)

    def as_dict(self):
        return asdict(self)


@dataclass
class Decision:
    message_id: str
    category_ids: list[str]
    method: str
    score: float
    reason: str
    attention: list[str]

    def as_dict(self):
        return asdict(self)
