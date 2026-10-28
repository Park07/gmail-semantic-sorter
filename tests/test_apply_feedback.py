import copy

import pytest

from sorter.apply_labels import apply_plan, recover
from sorter.categories import taxonomy_hash
from sorter.classify import classification_fingerprint
from sorter.config import Settings, SorterError
from sorter.feedback import observer
from sorter.models import Category
from conftest import FakeGmail, mail


def setup_plan(store, count=3):
    messages = [mail(i) for i in range(1, count + 1)]
    gmail = FakeGmail(messages)
    categories = [
        Category("uni", "University", "University correspondence", [], [], []),
        Category("bills", "Bills", "Billing correspondence", [], [], []),
    ]
    settings = Settings()
    store.bind_account(gmail.address)
    for message in messages:
        store.save_message(message)
    store.set_meta("approved_taxonomy", taxonomy_hash(categories))
    plan = {
        "id": "plan-one",
        "account": gmail.address,
        "taxonomy": taxonomy_hash(categories),
        "settings": settings.fingerprint(),
        "classification": classification_fingerprint(store, categories, settings),
        "items": [
            {
                "message_id": m.id,
                "category_ids": ["uni"],
                "labels": ["AI/University"],
                "content_hash": m.content_hash,
                "method": "centroid",
            }
            for m in messages
        ],
    }
    store.save_plan(plan)
    return gmail, categories, settings, plan


def test_dry_run_needs_no_gmail_client_and_writes_nothing(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    result = apply_plan(store, categories, settings, plan)
    assert result["dry_run"] and result["selected"] == 3
    assert not gmail.created and not gmail.added and not store.counts()["writes"]


def test_capped_apply_and_reruns_do_not_touch_processed_messages(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    first = apply_plan(store, categories, settings, plan, gmail, True, limit=2)
    assert first["applied"] == 2
    second = apply_plan(store, categories, settings, plan, gmail, True, limit=2)
    assert second["applied"] == 1
    third = apply_plan(store, categories, settings, plan, gmail, True)
    assert third["applied"] == 0
    touched = [i for batch, labels in gmail.added for i in batch]
    assert len(touched) == len(set(touched)) == 3


def test_all_targets_checked_before_any_mutation(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    tampered = copy.deepcopy(plan)
    tampered["items"][-1]["labels"] = ["TRASH"]
    with pytest.raises(SorterError):
        apply_plan(store, categories, settings, tampered, gmail, True)
    assert not gmail.created and not gmail.added
    store.set_meta("approved_taxonomy", None)
    with pytest.raises(SorterError):
        apply_plan(store, categories, settings, plan, gmail, True)
    assert not gmail.created


def test_account_mismatch_and_stale_plan_block_writes(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    gmail.address = "other@example.test"
    with pytest.raises(SorterError, match="different Gmail"):
        apply_plan(store, categories, settings, plan, gmail, True)
    assert not gmail.created
    gmail.address = plan["account"]
    store.override(mail(1).id, ["bills"])
    with pytest.raises(SorterError, match="corrections"):
        apply_plan(store, categories, settings, plan, gmail, True)
    assert not gmail.created


def test_live_eligibility_and_manual_labels_protect_messages(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    existing = gmail.create_ai_label("AI/Manual")
    gmail.created.clear()
    gmail.mail[mail(1).id].labels.append(existing["id"])
    gmail.mail[mail(2).id].labels = ["TRASH"]
    gmail.mail[mail(3).id].snippet = "Changed since the preview"
    result = apply_plan(store, categories, settings, plan, gmail, True)
    assert result["applied"] == 0 and result["protected"] == 1
    assert result["stale_or_missing"] == 2
    assert not gmail.created and not gmail.added


def test_timeout_after_acceptance_reconciles_without_second_write(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    gmail.fail_write = "after"
    with pytest.raises(TimeoutError):
        apply_plan(store, categories, settings, plan, gmail, True)
    assert store.write_record(mail(1).id)["status"] == "pending"
    gmail.fail_write = None
    result = apply_plan(store, categories, settings, plan, gmail, True)
    assert result["applied"] == 0
    assert recover(store, gmail)["confirmed"] == 3
    assert len(gmail.added) == 1


def test_missing_uncertain_write_requires_explicit_retry(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    gmail.fail_write = "before"
    with pytest.raises(TimeoutError):
        apply_plan(store, categories, settings, plan, gmail, True)
    gmail.fail_write = None
    assert recover(store, gmail)["held_for_review"] == 3
    assert apply_plan(store, categories, settings, plan, gmail, True)["applied"] == 0
    assert (
        apply_plan(store, categories, settings, plan, gmail, True, retry_held=True)["applied"] == 3
    )


def test_manual_gmail_corrections_become_examples_and_own_writes_do_not(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    apply_plan(store, categories, settings, plan, gmail, True)
    observe = observer(store, categories, gmail.labels())
    observe(gmail.mail[mail(1).id])
    assert not store.overrides()
    bills = gmail.create_ai_label("AI/Bills")
    gmail.mail[mail(1).id].labels = ["INBOX", bills["id"]]
    observer(store, categories, gmail.labels())(gmail.mail[mail(1).id])
    assert store.overrides()[mail(1).id] == ["bills"]
    gmail.mail[mail(2).id].labels = ["INBOX"]
    observer(store, categories, gmail.labels())(gmail.mail[mail(2).id])
    assert store.overrides()[mail(2).id] == []
    # The persistent write ledger protects both messages even after label removal.
    assert store.write_record(mail(2).id)["status"] == "done"


def test_model_errors_are_reported_without_labelling_review(environment):
    _, store = environment
    gmail, categories, settings, plan = setup_plan(store)
    for item in plan["items"]:
        item["method"] = "error"
    result = apply_plan(store, categories, settings, plan, gmail, True)
    assert result["model_errors"] == 3 and result["selected"] == 0
    assert not gmail.created and not gmail.added
