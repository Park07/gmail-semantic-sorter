import ast
import json
from pathlib import Path
from urllib.parse import urlsplit

import httplib2
import pytest
from googleapiclient.discovery import build

from sorter.auth import READ_SCOPE, WRITE_SCOPE, check_scopes
from sorter.config import SorterError, Settings
from sorter.gmail import Gmail
from sorter.models import validate_label
from sorter.ollama import Ollama, NoRedirect
from sorter.reports import csv_cell, report


class NoWait:
    def wait(self, cost):
        pass


class RecordingHTTP:
    def __init__(self):
        self.calls = []

    def request(self, uri, method="GET", body=None, headers=None, **kwargs):
        self.calls.append((uri, method, body))
        response = {
            "labels": [
                {"id": "Label_1", "name": "AI/Bills", "type": "user"},
                {"id": "TRASH", "name": "TRASH", "type": "system"},
            ]
        }
        return httplib2.Response({"status": "200", "content-type": "application/json"}), json.dumps(
            response
        ).encode()


def test_real_google_sdk_request_adds_only_validated_user_labels():
    http = RecordingHTTP()
    service = build("gmail", "v1", http=http, static_discovery=True, cache_discovery=False)
    client = Gmail(service, write_allowed=True, limiter=NoWait())
    client.add_labels(["abcdef123456"], ["Label_1"])
    writes = [c for c in http.calls if c[1] == "POST"]
    assert len(writes) == 1 and urlsplit(writes[0][0]).path.endswith("/messages/batchModify")
    assert json.loads(writes[0][2]) == {"ids": ["abcdef123456"], "addLabelIds": ["Label_1"]}
    with pytest.raises(SorterError):
        client.add_labels(["abcdef123456"], ["TRASH"])
    assert len([c for c in http.calls if c[1] == "POST"]) == 1


def test_read_client_cannot_write_even_with_a_service_object():
    client = Gmail(object(), limiter=NoWait())
    with pytest.raises(SorterError, match="disabled"):
        client.create_ai_label("AI/Bills")
    with pytest.raises(SorterError, match="disabled"):
        client.add_labels(["abc"], ["Label_1"])


def test_source_never_calls_dangerous_gmail_methods():
    root = Path(__file__).parents[1] / "sorter"
    forbidden = {"trash", "untrash", "delete", "batchDelete", "send", "modify"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden, (path, node.lineno)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value != "removeLabelIds"


@pytest.mark.parametrize(
    "name", ["TRASH", "INBOX", "AI/", "AI/../Bills", "AI/Bills\n", "Other/Bills", "AI/Review/"]
)
def test_invalid_label_targets(name):
    with pytest.raises(SorterError):
        validate_label(name)


def test_scopes_are_separate_and_fail_closed():
    check_scopes([READ_SCOPE])
    check_scopes([WRITE_SCOPE], write=True)
    for scopes in ([WRITE_SCOPE], [READ_SCOPE, WRITE_SCOPE], ["https://mail.google.com/"], []):
        with pytest.raises(SorterError):
            check_scopes(scopes)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://192.168.1.2:11434",
        "http://localhost@example.com",
        "http://127.0.0.1:11434/redirect",
        "http://127.0.0.1:11434?x=y",
    ],
)
def test_model_endpoints_must_be_local(url):
    with pytest.raises(SorterError):
        Ollama(Settings(ollama_url=url))


def test_cloud_model_and_redirect_rejected():
    client = Ollama(Settings())
    with pytest.raises(SorterError):
        client.identity("qwen-cloud")
    with pytest.raises(SorterError):
        NoRedirect().redirect_request(None, None, None, None, None)


@pytest.mark.parametrize(
    "value", ['=IMPORTXML("bad")', "   +cmd", "-1+2", "@SUM(1)", "\tformula", "\nformula"]
)
def test_csv_formula_protection(value):
    assert csv_cell(value).startswith("'")


@pytest.mark.parametrize("value", ["\x00=cmd()", "\ufeff+cmd()", "\x01@SUM(1)"])
def test_csv_ignored_control_characters_are_safe(value):
    assert csv_cell(value).startswith("'")


def test_html_is_escaped_and_private(environment):
    paths, _ = environment
    attack = '<img src="https://bad.example" onerror="alert(1)"><script>alert(2)</script>'
    output = report(paths, "hostile", "Safe report", [{"subject": attack, "snippet": "=danger()"}])
    text = output.read_text()
    assert "<img src=" not in text and "&lt;img" in text
    assert "<script>alert(2)" not in text
    assert "Content-Security-Policy" in text
    assert output.stat().st_mode & 0o777 == 0o600
