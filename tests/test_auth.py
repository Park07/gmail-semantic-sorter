import json

import pytest

from sorter.auth import READ_SCOPE, WRITE_SCOPE, authorize, save_authorization
from sorter.config import SorterError


def test_desktop_oauth_uses_pkce_loopback_and_staged_scopes(environment, monkeypatch):
    paths, _ = environment
    paths.credentials.write_text(json.dumps({"installed": {"client_id": "synthetic-test-client"}}))
    captured = {}

    class Credentials:
        scopes = [READ_SCOPE]
        granted_scopes = [READ_SCOPE]

        def to_json(self):
            return json.dumps({"token": "synthetic-noncredential", "scopes": self.scopes})

    class Flow:
        @classmethod
        def from_client_config(cls, config, scopes, **kwargs):
            captured.update(scopes=scopes, **kwargs)
            return cls()

        def run_local_server(self, **kwargs):
            captured.update(kwargs)
            result = Credentials()
            result.scopes = result.granted_scopes = captured["scopes"]
            return result

    monkeypatch.setattr("google_auth_oauthlib.flow.InstalledAppFlow", Flow)
    _, payload = authorize(paths)
    assert captured["scopes"] == [READ_SCOPE]
    assert captured["host"] == "127.0.0.1" and captured["port"] == 0
    assert captured["autogenerate_code_verifier"] is True
    assert captured["include_granted_scopes"] == "false"
    assert not paths.token().exists()
    save_authorization(paths, payload)
    assert paths.token().stat().st_mode & 0o777 == 0o600
    assert not paths.token(True).exists()
    authorize(paths, write=True)
    assert captured["scopes"] == [WRITE_SCOPE]


def test_wrong_oauth_client_type_is_rejected_before_login(environment):
    paths, _ = environment
    paths.credentials.write_text('{"web": {}}')
    with pytest.raises(SorterError, match="Desktop"):
        authorize(paths)
