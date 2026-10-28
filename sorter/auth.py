from __future__ import annotations

import json

from .config import Paths, SorterError, atomic_write

READ_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
WRITE_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


def check_scopes(scopes, write=False):
    granted = set(scopes or [])
    expected = WRITE_SCOPE if write else READ_SCOPE
    gmail_scopes = {s for s in granted if "gmail" in s or s.startswith("https://mail.google.com")}
    identity_scopes = {
        "openid",
        "email",
        "profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    }
    allowed_gmail = {READ_SCOPE, WRITE_SCOPE} if write else {READ_SCOPE}
    if (
        expected not in gmail_scopes
        or not gmail_scopes <= allowed_gmail
        or granted - gmail_scopes - identity_scopes
    ):
        raise SorterError(
            "Unexpected Gmail permissions. Reauthorise using the correct read-only or label-write command; if Google combines old grants, use a separate Desktop OAuth client."
        )


def authorize(paths: Paths, write=False):
    from google_auth_oauthlib.flow import InstalledAppFlow

    credential_file = (
        paths.root / "credentials-write.json"
        if write and (paths.root / "credentials-write.json").exists()
        else paths.credentials
    )
    try:
        document = json.loads(credential_file.read_text())
    except (OSError, ValueError) as exc:
        raise SorterError(
            "Place the downloaded Desktop OAuth credentials.json in the project root."
        ) from exc
    if "installed" not in document:
        raise SorterError("credentials.json must be a Google Desktop app OAuth client.")
    credential_file.chmod(0o600)
    flow = InstalledAppFlow.from_client_config(
        document, scopes=[WRITE_SCOPE if write else READ_SCOPE], autogenerate_code_verifier=True
    )
    credentials = flow.run_local_server(
        host="127.0.0.1",
        port=0,
        prompt="consent",
        include_granted_scopes="false",
        success_message="Gmail Semantic Sorter is connected. You may close this tab.",
    )
    actual = credentials.granted_scopes or credentials.scopes
    check_scopes(actual, write)
    # Save the actual grant, not a guessed scope supplied while loading a token.
    payload = json.loads(credentials.to_json())
    payload["scopes"] = list(actual)
    return credentials, json.dumps(payload)


def save_authorization(paths, payload, write=False):
    atomic_write(paths.token(write), payload)


def load_credentials(paths: Paths, write=False):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError

    path = paths.token(write)
    if not path.exists():
        command = "gmail-sorter auth --enable-labels" if write else "gmail-sorter auth"
        raise SorterError(f"No saved credentials. Run {command} on your Mac.")
    try:
        credentials = Credentials.from_authorized_user_file(str(path))
        check_scopes(credentials.scopes, write)
        if not credentials.valid:
            if not credentials.refresh_token:
                raise SorterError("Google login has expired. Run auth again on your Mac.")
            credentials.refresh(Request())
            check_scopes(credentials.granted_scopes or credentials.scopes, write)
            atomic_write(path, credentials.to_json())
        return credentials
    except RefreshError as exc:
        raise SorterError(
            "Google login needs renewal. Run auth again on your Mac (Testing-mode tokens usually expire after seven days)."
        ) from exc
    except (ValueError, OSError) as exc:
        raise SorterError("The saved OAuth token is invalid. Run auth again on your Mac.") from exc


def service_for(credentials):
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build

    return build(
        "gmail",
        "v1",
        http=AuthorizedHttp(credentials, http=httplib2.Http(timeout=60)),
        cache_discovery=False,
    )
