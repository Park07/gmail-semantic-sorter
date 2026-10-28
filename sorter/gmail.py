from __future__ import annotations

import json
import random
import time

from .config import SorterError
from .models import Message, validate_label, validate_message_id

HEADERS = ["From", "To", "Subject", "Date", "List-Unsubscribe", "List-Id"]


class GmailError(SorterError):
    def __init__(self, status):
        self.status = status
        super().__init__(
            f"Gmail request failed (HTTP {status}). Work remains queued; check authorisation/quota and retry."
        )


def status_of(exc):
    return getattr(getattr(exc, "resp", None), "status", 0)


def retryable(exc):
    status = status_of(exc)
    if status in {429, 500, 502, 503, 504} or isinstance(
        exc, (TimeoutError, ConnectionError, OSError)
    ):
        return True
    if status == 403:
        try:
            errors = json.loads(exc.content).get("error", {}).get("errors", [])
            return any(
                e.get("reason") in {"rateLimitExceeded", "userRateLimitExceeded", "backendError"}
                for e in errors
            )
        except (ValueError, AttributeError):
            return False
    return False


class QuotaLimiter:
    def __init__(self, units_per_second=90):
        self.rate = units_per_second
        self.next_at = 0.0

    def wait(self, cost):
        delay = self.next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_at = time.monotonic() + cost / self.rate


class Gmail:
    """Only Gmail reads, creation of AI user labels, and validated label additions."""

    def __init__(self, service, write_allowed=False, limiter=None):
        self.service = service
        self.write_allowed = write_allowed
        self.limiter = limiter or QuotaLimiter()

    def _execute(self, request, cost=5):
        for attempt in range(5):
            self.limiter.wait(cost)
            try:
                return request.execute()
            except Exception as exc:
                if not retryable(exc) or attempt == 4:
                    raise GmailError(status_of(exc)) from exc
                time.sleep(min(16, 2**attempt) + random.random() / 4)

    def profile(self):
        return self._execute(self.service.users().getProfile(userId="me"))

    def list_page(self, query, page_token=None, size=100):
        args = dict(userId="me", q=query, maxResults=min(500, size), includeSpamTrash=False)
        if page_token:
            args["pageToken"] = page_token
        return self._execute(self.service.users().messages().list(**args))

    def message(self, message_id):
        return Message.from_gmail(self._execute(self._message_request(message_id), 20))

    def _message_request(self, message_id):
        return (
            self.service.users()
            .messages()
            .get(
                userId="me",
                id=validate_message_id(message_id),
                format="metadata",
                metadataHeaders=HEADERS,
            )
        )

    def messages(self, ids):
        results, errors = {}, {}

        def callback(request_id, response, exception):
            if exception:
                errors[request_id] = exception
            else:
                errors.pop(request_id, None)
                results[request_id] = Message.from_gmail(response)

        for offset in range(0, len(ids), 50):
            chunk = ids[offset : offset + 50]
            batch = self.service.new_batch_http_request(callback=callback)
            for message_id in chunk:
                batch.add(self._message_request(message_id), request_id=message_id)
            self._execute(batch, 20 * len(chunk))
        for message_id, error in errors.items():
            if status_of(error) == 404:
                results[message_id] = None
            elif retryable(error):
                try:
                    results[message_id] = self.message(message_id)
                except GmailError as exc:
                    if exc.status != 404:
                        raise
                    results[message_id] = None
            else:
                raise GmailError(status_of(error)) from error
        if set(results) != set(ids):
            raise SorterError(
                "Incomplete Gmail batch response. Pending messages have been retained."
            )
        return results

    def history_page(self, cursor, page_token=None):
        args = dict(userId="me", startHistoryId=cursor, maxResults=500)
        if page_token:
            args["pageToken"] = page_token
        return self._execute(self.service.users().history().list(**args), 2)

    def labels(self):
        return self._execute(self.service.users().labels().list(userId="me"), 1).get("labels", [])

    def create_ai_label(self, name):
        self._require_write()
        validate_label(name)
        result = self._execute(
            self.service.users()
            .labels()
            .create(
                userId="me",
                body={
                    "name": name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            ),
            5,
        )
        if result.get("type") != "user" or result.get("name") != name or not result.get("id"):
            raise SorterError("Gmail did not return the expected AI user label.")
        return result

    def add_labels(self, ids, label_ids):
        self._require_write()
        if not ids or len(ids) > 1000 or not label_ids:
            raise SorterError("A label batch requires 1 to 1,000 messages and at least one label.")
        for message_id in ids:
            validate_message_id(message_id)
        by_id = {label["id"]: label for label in self.labels()}
        for label_id in label_ids:
            label = by_id.get(label_id, {})
            if label.get("type") != "user":
                raise SorterError("Refusing to apply a system label or an unknown label.")
            validate_label(label.get("name"))
        return self._execute(
            self.service.users()
            .messages()
            .batchModify(
                userId="me", body={"ids": list(ids), "addLabelIds": sorted(set(label_ids))}
            ),
            50,
        )

    def _require_write(self):
        if not self.write_allowed:
            raise SorterError("Gmail writes are disabled for this client.")
