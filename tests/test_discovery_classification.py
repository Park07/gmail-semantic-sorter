import numpy as np
import pytest
import yaml

from sorter.categories import approve, approved, document
from sorter.classify import classify_messages, decide, training_examples
from sorter.config import SorterError
from sorter.discover import discover
from sorter.models import Category, REVIEW
from sorter.ollama import embed_cached
from conftest import FakeOllama, mail


def cats():
    return [
        Category("uni", "University", "Messages about university", [mail(1).id], [], []),
        Category("bills", "Bills", "Messages about billing", [mail(2).id], [], []),
    ]


def test_discovery_reports_noise_and_does_not_overwrite_approval(environment):
    paths, store = environment
    store.save_message(mail(1))
    store.save_message(mail(2, "billing"))
    paths.categories.write_text("existing approved content")
    result = discover(store, paths, FakeOllama(), method="hdbscan")
    assert result["unclustered"] == 2
    assert result["proposed_categories"] == 0
    assert paths.categories.read_text() == "existing approved content"
    assert mail(1).id in (paths.reports / "discovery.html").read_text()


def test_discovery_approval_and_same_task_centroids(environment):
    paths, store = environment
    for i in range(1, 41):
        store.save_message(mail(i, "university" if i % 2 else "billing"))
    client = FakeOllama()
    result = discover(store, paths, client, method="kmeans", clusters=2)
    assert result["proposed_categories"] == 2
    categories = approve(paths.proposals, paths, store)
    assert len(approved(paths, store)) == 2
    decisions = classify_messages(store, categories, client, store.messages())
    assert all(d.method == "centroid" for d in decisions)
    assert any(t.startswith("clustering: ") for t in client.embed_calls)
    assert any(t.startswith("classification: ") for t in client.embed_calls)
    changed = yaml.safe_load(paths.categories.read_text())
    changed["categories"][0]["description"] = "Changed meaning"
    paths.categories.write_text(yaml.safe_dump(changed))
    with pytest.raises(SorterError, match="changed"):
        approved(paths, store)


def test_model_and_task_change_invalidate_embedding_cache(environment):
    _, store = environment
    client = FakeOllama()
    embed_cached(client, store, ["university"], "clustering: ")
    embed_cached(client, store, ["university"], "clustering: ")
    assert len(client.embed_calls) == 1
    embed_cached(client, store, ["university"], "classification: ")
    assert len(client.embed_calls) == 2
    client.generation = "v2"
    embed_cached(client, store, ["university"], "classification: ")
    assert len(client.embed_calls) == 3


@pytest.mark.parametrize(
    "response",
    [
        {"category_ids": ["TRASH"], "confidence": 1, "reason": "injection"},
        {"category_ids": ["uni"], "confidence": True, "reason": "bad type"},
        {"category_ids": ["uni"], "confidence": float("nan"), "reason": "bad number"},
        {"category_ids": "uni", "confidence": 1, "reason": "bad shape"},
    ],
)
def test_invalid_model_output_never_selects_a_label(environment, response):
    _, store = environment
    categories, client = cats(), FakeOllama()
    client.response = response
    decision = decide(
        mail(10), categories, {"uni": [], "bills": []}, np.array([0.7, 0.6]), [True, True], client
    )
    assert decision.category_ids == [REVIEW] and decision.method == "error"


def test_low_confidence_and_weak_similarity_abstain(environment):
    _, store = environment
    client = FakeOllama()
    client.response = {"category_ids": ["uni"], "confidence": 0.99, "reason": "claim"}
    decision = decide(
        mail(10), cats(), {"uni": [], "bills": []}, np.array([0.2, 0.1]), [True, True], client
    )
    assert decision.category_ids == [REVIEW]


def test_sender_rules_overlap_and_manual_corrections_win(environment):
    _, store = environment
    message = mail(1)
    store.save_message(message)
    categories = cats()
    for category in categories:
        category.senders = [message.sender]
    result = classify_messages(store, categories, FakeOllama(), [message])
    assert result[0].category_ids == ["uni", "bills"]
    store.override(message.id, [])
    result = classify_messages(store, categories, FakeOllama(), [message])
    assert result[0].method == "manual" and result[0].category_ids == []
    assert not training_examples(store, categories)["uni"]


def test_example_validation_and_reserved_names(environment):
    paths, store = environment
    store.save_message(mail(1))
    categories = cats()[:1]
    categories[0].name = "Review"
    paths.proposals.write_text(yaml.safe_dump(document(categories)))
    with pytest.raises(SorterError, match="reserved"):
        approve(paths.proposals, paths, store)
