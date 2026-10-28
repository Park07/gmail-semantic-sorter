import pytest

from sorter.config import SorterError
from sorter.gmail import GmailError
from sorter.ingest import ingest, sync
from conftest import FakeGmail, mail


def test_bounded_import_resumes_without_losing_pages(environment):
    _, store = environment
    gmail = FakeGmail([mail(i) for i in range(1, 13)])
    assert ingest(gmail, store, "query", 5)["imported"] == 5
    assert store.counts()["messages"] == 5
    assert ingest(gmail, store, "query", 5)["imported"] == 5
    assert ingest(gmail, store, "query", 5)["complete"]
    assert store.counts()["messages"] == 12
    assert ingest(gmail, store, "query", 5)["imported"] == 0
    ingest(gmail, store, "query", 20, restart=True)
    assert store.counts()["messages"] == 12


def test_fetch_failure_keeps_durable_queue(environment):
    _, store = environment
    gmail = FakeGmail([mail(1), mail(2)])
    gmail.fail_fetch = True
    with pytest.raises(GmailError):
        ingest(gmail, store, "query", 2)
    assert store.counts()["pending"] == 2
    assert store.counts()["messages"] == 0
    gmail.fail_fetch = False
    ingest(gmail, store, "query", 2)
    assert store.counts()["messages"] == 2
    assert store.counts()["pending"] == 0


def test_history_pages_queue_before_checkpoint_and_fetch(environment):
    _, store = environment
    gmail = FakeGmail([mail(1), mail(2)])
    store.bind_account(gmail.address)
    store.set_meta("history_cursor", "10")
    gmail.history_pages = [
        {
            "history": [{"messagesAdded": [{"message": {"id": mail(1).id}}]}],
            "nextPageToken": "1",
            "historyId": "200",
        },
        {"history": [{"labelsAdded": [{"message": {"id": mail(2).id}}]}], "historyId": "200"},
    ]
    gmail.fail_fetch = True
    with pytest.raises(GmailError):
        sync(gmail, store, "query")
    assert store.meta("history_cursor") == "200"
    assert store.counts()["pending"] == 2
    gmail.fail_fetch = False
    sync(gmail, store, "query", limit=1)
    assert store.counts()["messages"] == 1
    assert store.counts()["pending"] == 1


def test_expired_history_rescans_scope_and_existing_messages(environment):
    _, store = environment
    gmail = FakeGmail([mail(i) for i in range(1, 6)])
    store.save_message(mail(90))
    store.set_meta("history_cursor", "expired")
    gmail.expired = True
    result = sync(gmail, store, "scope", limit=20)
    assert result["history_recovered"]
    assert store.message(mail(90).id).unavailable
    assert len(store.messages()) == 5
    assert store.meta("history_cursor") == "100"


def test_wrong_account_and_excluded_mail(environment):
    _, store = environment
    store.bind_account("different@example.test")
    gmail = FakeGmail()
    with pytest.raises(SorterError, match="different Gmail"):
        ingest(gmail, store, "query")
    assert not gmail.list_calls
    for i, label in enumerate(["SENT", "DRAFT", "SPAM", "TRASH"], start=1):
        store.save_message(mail(i, labels=[label]))
    assert not store.messages()
