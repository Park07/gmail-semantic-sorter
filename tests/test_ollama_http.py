import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from sorter.config import Settings, SorterError
from sorter.ollama import Ollama


@pytest.fixture
def server():
    calls = []
    state = {"remote": False, "redirect": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def respond(self, body):
            calls.append((self.path, body))
            if state["redirect"]:
                self.send_response(302)
                self.send_header("Location", "http://must-not-connect.invalid/collect")
                self.end_headers()
                return
            data = {}
            if self.path == "/api/tags":
                data = {
                    "models": [
                        {"name": "nomic-embed-text:latest", "digest": "embed-v1"},
                        {"name": "qwen3.5:0.8b", "digest": "chat-v1"},
                    ]
                }
            elif self.path == "/api/show":
                data = (
                    {"remote_host": "https://cloud.invalid"}
                    if state["remote"]
                    else {"details": {"format": "gguf"}}
                )
            elif self.path == "/api/embed":
                data = {"embeddings": [[3.0, 4.0] for _ in body["input"]]}
            elif self.path == "/api/chat":
                data = {
                    "message": {
                        "content": json.dumps(
                            {"name": "University", "description": "University mail"}
                        )
                    }
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield Settings(ollama_url=f"http://127.0.0.1:{http.server_port}"), calls, state
    http.shutdown()
    http.server_close()
    thread.join()


def test_http_payloads_use_local_models_json_and_bounded_context(server, monkeypatch):
    settings, calls, _ = server
    # System proxy settings must not redirect private prompts through a proxy.
    monkeypatch.setenv("HTTP_PROXY", "http://must-not-connect.invalid:99")
    client = Ollama(settings)
    assert client.embed(["classification: university"]).tolist() == [[0.6, 0.8]]
    result = client.chat("Name category", {"email": "university"}, {"type": "object"})
    assert result["name"] == "University"
    payload = next(data for path, data in calls if path == "/api/chat")
    assert payload["think"] is False and payload["keep_alive"] == 0 and payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert "tools" not in payload


def test_cloud_backed_models_never_receive_prompts(server):
    settings, calls, state = server
    state["remote"] = True
    with pytest.raises(SorterError, match="cloud"):
        Ollama(settings).embed(["private prompt must not be sent"])
    assert not any(path == "/api/embed" for path, _ in calls)


def test_redirects_never_forward_to_remote_hosts(server):
    settings, calls, state = server
    state["redirect"] = True
    with pytest.raises(SorterError):
        Ollama(settings).identity("nomic-embed-text:latest")
    assert len(calls) == 1
