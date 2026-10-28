import copy
import hashlib

import numpy as np
import pytest

from sorter.config import Paths, Settings
from sorter.gmail import GmailError
from sorter.models import Message
from sorter.storage import Store


def mail(number=1, topic="university", labels=None):
    return Message(
        f"{number:016x}",
        f"thread-{number}",
        1_700_000_000_000 + number * 1000,
        f"sender@{topic}.example",
        f"{topic}.example",
        f"{topic} message {number}",
        f"Details about {topic} number {number}",
        {"from": f"sender@{topic}.example"},
        list(labels if labels is not None else ["INBOX"]),
    )


class FakeGmail:
    def __init__(self, messages=()):
        self.mail = {m.id: copy.deepcopy(m) for m in messages}
        self.write_allowed = True
        self.address = "personal@example.test"
        self.label_data = [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "TRASH", "name": "TRASH", "type": "system"},
        ]
        self.added, self.created = [], []
        self.fail_fetch = False
        self.fail_write = None
        self.expired = False
        self.history_pages = [{"history": [], "historyId": "200"}]
        self.list_calls = []

    def profile(self):
        return {"emailAddress": self.address, "historyId": "100"}

    def list_page(self, query, page_token=None, size=100):
        self.list_calls.append((query, page_token, size))
        ids = list(self.mail)
        start = int(page_token or 0)
        result = {"messages": [{"id": i} for i in ids[start : start + size]]}
        if start + size < len(ids):
            result["nextPageToken"] = str(start + size)
        return result

    def messages(self, ids):
        if self.fail_fetch:
            raise GmailError(503)
        return {i: copy.deepcopy(self.mail.get(i)) for i in ids}

    def history_page(self, cursor, page_token=None):
        if self.expired:
            raise GmailError(404)
        return copy.deepcopy(self.history_pages[int(page_token or 0)])

    def labels(self):
        return copy.deepcopy(self.label_data)

    def create_ai_label(self, name):
        result = {"id": "Label_" + str(len(self.label_data)), "name": name, "type": "user"}
        self.label_data.append(result)
        self.created.append(name)
        return result

    def add_labels(self, ids, label_ids):
        if self.fail_write == "before":
            raise TimeoutError()
        self.added.append((list(ids), list(label_ids)))
        for i in ids:
            self.mail[i].labels = sorted(set(self.mail[i].labels) | set(label_ids))
        if self.fail_write == "after":
            raise TimeoutError()


class FakeOllama:
    def __init__(self, settings=None):
        self.settings = settings or Settings(min_cluster_size=3, min_samples=2)
        self.embed_calls, self.chat_calls = [], []
        self.response = None
        self.generation = "v1"

    def identity(self, model):
        return model + self.generation

    def embed(self, texts):
        self.embed_calls.extend(texts)
        vectors = []
        for text in texts:
            kind = 0 if "university" in text.lower() else 1 if "billing" in text.lower() else 2
            vector = np.array([0.01, 0.01, 0.01])
            vector[kind] = 1
            vector += (hashlib.sha256(text.encode()).digest()[0] / 255) * 0.002
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)

    def chat(self, instruction, data, schema):
        self.chat_calls.append(data)
        if self.response is not None:
            return copy.deepcopy(self.response)
        if "samples" in data:
            topic = "University" if "university" in data["samples"][0] else "Billing"
            return {"name": topic, "description": f"Messages about {topic.lower()}."}
        return {"category_ids": [], "confidence": 0.2, "reason": "Uncertain in test fixture."}


@pytest.fixture
def environment(tmp_path):
    paths = Paths(tmp_path)
    paths.prepare()
    store = Store(paths.database)
    yield paths, store
    store.close()
