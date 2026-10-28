import csv
import json
import plistlib
from pathlib import Path

import pytest
import yaml

from sorter import cli
from sorter.categories import read_categories
from sorter.config import Paths, RunLock, Settings, SorterError
from sorter.evaluate import evaluate
from sorter.models import Category
from sorter.schedule import generate
from conftest import FakeGmail, FakeOllama, mail


def test_complete_cli_pipeline_with_mocked_external_services(tmp_path, monkeypatch, capsys):
    gmail = FakeGmail([mail(i, "university" if i % 2 else "billing") for i in range(1, 41)])

    def client(paths, write=False):
        gmail.write_allowed = write
        return gmail

    monkeypatch.setattr(cli, "gmail_client", client)
    monkeypatch.setattr(cli, "Ollama", FakeOllama)

    def run(*args):
        code = cli.main(["--root", str(tmp_path), *args])
        output = capsys.readouterr()
        assert code == 0, output.err
        return json.loads(output.out)

    run("init")
    assert (tmp_path / "data").stat().st_mode & 0o777 == 0o700
    assert run("ingest", "--limit", "40")["messages"] == 40
    assert run("discover", "--method", "kmeans", "--clusters", "2")["proposed_categories"] == 2
    run("approve")
    prediction = run("classify")
    assert prediction["messages"] == 40 and prediction["review"] == 0
    assert Path(prediction["report"]).exists()
    assert run("apply")["dry_run"] and not gmail.added
    sample = run("eval-sample", "--count", "8")
    categories = read_categories(Paths(tmp_path).categories)
    by_topic = {c.name.lower(): c.id for c in categories}
    csv_path = Path(sample["csv"])
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["expected_category_ids"] = by_topic[row["subject"].split()[0]]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert run("evaluate")["exact_match_accuracy"] == 1.0
    assert run("apply", "--apply", "--limit", "5")["applied"] == 5
    assert run("apply", "--apply", "--limit", "5")["applied"] == 5
    assert run("status")["writes"]["done"] == 10
    schedule = run("schedule")
    assert not schedule["installed"] and not schedule["applies_labels"]
    assert "--apply" not in plistlib.loads(Path(schedule["plist"]).read_bytes())["ProgramArguments"]


def test_evaluation_rejects_thread_leakage(environment):
    paths, store = environment
    first, second = mail(1), mail(2)
    second.thread_id = first.thread_id
    store.save_message(first)
    store.save_message(second)
    category = Category("uni", "Uni", "University", [first.id], [], [])
    file = paths.data / "holdout.csv"
    file.write_text("message_id,expected_category_ids\n" + second.id + ",uni\n")
    with pytest.raises(SorterError, match="overlaps"):
        evaluate(store, paths, [category], FakeOllama(), file)


def test_schedule_uses_exact_paths_and_apply_is_explicit(environment):
    paths, _ = environment
    result = generate(paths, apply=True)
    data = plistlib.loads(Path(result["plist"]).read_bytes())
    assert data["WorkingDirectory"] == str(paths.root)
    assert data["ProgramArguments"][-1] == "--apply"
    assert data["RunAtLoad"] is False
    assert data["Umask"] == 0o077


def test_concurrent_state_mutations_are_blocked(environment):
    paths, _ = environment
    with RunLock(paths.data / "run.lock"):
        with pytest.raises(SorterError, match="Another sorter"):
            with RunLock(paths.data / "run.lock"):
                pass


def test_bad_configuration_fails_before_network_or_writes(tmp_path):
    (tmp_path / "sorter.toml").write_text("similarity_threshold = nan\n")
    with pytest.raises(SorterError):
        Settings.load(tmp_path)
    assert not (tmp_path / "data").exists()


def test_category_approval_backs_up_previous_file(environment):
    from sorter.categories import approve, document

    paths, store = environment
    category = Category("uni", "Uni", "University", [], [], [])
    paths.proposals.write_text(yaml.safe_dump(document([category])))
    approve(paths.proposals, paths, store)
    category.name = "University"
    paths.proposals.write_text(yaml.safe_dump(document([category])))
    approve(paths.proposals, paths, store)
    backups = list((paths.data / "taxonomies").glob("*.yaml"))
    assert len(backups) == 1 and "name: Uni\n" in backups[0].read_text()
