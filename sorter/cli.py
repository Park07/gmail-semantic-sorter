from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys

from . import __version__
from . import auth
from .apply_labels import apply_plan, recover
from .categories import approve, approved
from .classify import preview
from .config import Paths, RunLock, Settings, SorterError, atomic_write
from .discover import discover
from .evaluate import evaluate, export_sample
from .feedback import correction, observer
from .gmail import Gmail
from .ingest import connect_account, drain, ingest, sync
from .ollama import Ollama
from .reports import audit
from .schedule import generate
from .storage import Store


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    result = argparse.ArgumentParser(
        description="Discover and review personal Gmail categories locally. Gmail writes always require --apply."
    )
    result.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project directory (default: current directory)",
    )
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create private data/report folders and local configuration")
    commands.add_parser(
        "doctor", help="Check dependencies, configuration, and downloaded local models"
    )
    commands.add_parser("status", help="Read local import, approval, and write status")
    login = commands.add_parser("auth", help="Google Desktop OAuth login on this Mac")
    login.add_argument(
        "--enable-labels",
        action="store_true",
        help="Request separate gmail.modify credentials for explicit apply runs",
    )
    importing = commands.add_parser(
        "ingest", help="Import Gmail metadata/snippets into local SQLite"
    )
    importing.add_argument("--limit", type=positive, default=1000)
    importing.add_argument("--query", help="Override the initial Gmail query")
    importing.add_argument(
        "--restart",
        action="store_true",
        help="Restart listing this query; stored messages are deduplicated",
    )
    syncing = commands.add_parser(
        "sync", help="Fetch new mail and label changes using durable history checkpoints"
    )
    syncing.add_argument("--limit", type=positive, default=1000)
    commands.add_parser("audit", help="Generate local inbox counts and rule coverage reports")
    discovery = commands.add_parser(
        "discover", help="Cluster imported mail and propose category names"
    )
    discovery.add_argument("--limit", type=positive, default=1000)
    discovery.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    discovery.add_argument(
        "--clusters",
        type=positive,
        default=10,
        help="Number of groups for explicit k-means comparison",
    )
    approval = commands.add_parser(
        "approve", help="Approve an edited category file; no Gmail access"
    )
    approval.add_argument("--file", type=Path, help="Default: data/categories.proposed.yaml")
    commands.add_parser("categories", help="List approved category IDs")
    classification = commands.add_parser(
        "classify", help="Create a saved preview and local reports"
    )
    classification.add_argument("--limit", type=positive, default=1000)
    classification.add_argument(
        "--include-processed",
        action="store_true",
        help="Include previously labelled messages for reporting only",
    )
    classification.add_argument("--refresh", action="store_true", help="Recompute cached decisions")
    applying = commands.add_parser("apply", help="Preview or apply an existing saved plan")
    applying.add_argument("--plan", help="Saved plan ID; default is the latest preview")
    applying.add_argument("--limit", type=positive, default=100)
    applying.add_argument(
        "--apply", action="store_true", help="Actually create/add approved AI labels in Gmail"
    )
    applying.add_argument(
        "--retry-held",
        action="store_true",
        help="Explicitly retry ambiguous writes after running recover and reviewing them",
    )
    commands.add_parser(
        "recover",
        help="Read Gmail to reconcile interrupted writes; missing labels are held for review",
    )
    feedback = commands.add_parser(
        "feedback", help="Refresh previously labelled messages and learn observed manual changes"
    )
    feedback.add_argument("--limit", type=positive, default=1000)
    correcting = commands.add_parser(
        "correct", help="Record a local correction/example without changing Gmail"
    )
    correcting.add_argument("message_id")
    choice = correcting.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--category", action="append", help="Approved ID or name; repeat for multiple categories"
    )
    choice.add_argument("--leave-unlabelled", action="store_true")
    choice.add_argument(
        "--clear",
        action="store_true",
        help="Remove the local override; existing Gmail labels and write protection remain",
    )
    sample = commands.add_parser("eval-sample", help="Export unseen threads for manual evaluation")
    sample.add_argument("--count", type=positive, default=100)
    evaluation = commands.add_parser("evaluate", help="Evaluate against a completed holdout CSV")
    evaluation.add_argument("--file", type=Path, default=Path("reports/evaluation-sample.csv"))
    running = commands.add_parser("run", help="Sync, classify, report, and optionally add labels")
    running.add_argument("--limit", type=positive, default=100)
    running.add_argument("--apply", action="store_true")
    scheduling = commands.add_parser(
        "schedule", help="Generate a macOS launchd plist; does not activate it"
    )
    scheduling.add_argument("--interval", type=positive, default=3600)
    scheduling.add_argument("--limit", type=positive, default=100)
    scheduling.add_argument(
        "--apply",
        action="store_true",
        help="Generate a schedule that adds labels, instead of dry runs",
    )
    return result


def location(paths, path):
    path = path.expanduser()
    return path if path.is_absolute() else paths.root / path


def gmail_client(paths, write=False):
    return Gmail(auth.service_for(auth.load_credentials(paths, write)), write_allowed=write)


def feedback_hook(paths, store, gmail):
    if not store.meta("approved_taxonomy"):
        return None
    try:
        categories = approved(paths, store)
    except SorterError:
        return None
    return observer(store, categories, gmail.labels())


def doctor(paths, settings):
    details = {
        "version": __version__,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "credentials_file": paths.credentials.exists(),
        "readonly_login": paths.token().exists(),
        "label_write_login": paths.token(True).exists(),
        "root": str(paths.root),
        "models": {},
    }
    client = Ollama(settings)
    for model in (settings.embedding_model, settings.chat_model):
        try:
            details["models"][model] = {"available": True, "digest": client.identity(model)}
        except SorterError as exc:
            details["models"][model] = {"available": False, "reason": str(exc)}
    return details


def status(paths):
    if not paths.database.exists():
        return {"initialised": False, "next": "Run init."}
    with sqlite3.connect(paths.database.as_uri() + "?mode=ro", uri=True) as connection:
        counts = {
            "messages": connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "pending": connection.execute("SELECT COUNT(*) FROM pending").fetchone()[0],
            "writes": dict(
                connection.execute("SELECT status,COUNT(*) FROM writes GROUP BY status")
            ),
        }
        for key in ("account", "approved_taxonomy", "latest_plan", "last_sync", "last_evaluation"):
            row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            counts[key] = json.loads(row[0]) if row else None
    return counts


def dispatch(args, paths, settings, store):
    command = args.command
    if command == "init":
        if not (paths.root / "sorter.toml").exists():
            atomic_write(paths.root / "sorter.toml", settings.example())
        ignore = paths.root / ".gitignore"
        content = ignore.read_text() if ignore.exists() else ""
        patterns = [
            "data/",
            "reports/",
            "credentials*.json",
            "token*.json",
            "client_secret*.json",
            ".env",
            ".env.*",
            "!.env.example",
            "categories.yaml",
            "sorter.toml",
            ".venv/",
        ]
        additions = [p for p in patterns if p not in content.splitlines()]
        if additions:
            atomic_write(
                ignore,
                content.rstrip()
                + "\n\n# Local Gmail Semantic Sorter data\n"
                + "\n".join(additions)
                + "\n",
            )
        return {
            "root": str(paths.root),
            "initialised": True,
            "next": "Add Desktop OAuth credentials.json, then run auth.",
        }
    if command == "auth":
        credentials, payload = auth.authorize(paths, args.enable_labels)
        profile = connect_account(Gmail(auth.service_for(credentials)), store)
        auth.save_authorization(paths, payload, args.enable_labels)
        return {
            "connected_account": profile["emailAddress"],
            "scope": "gmail.modify" if args.enable_labels else "gmail.readonly",
        }
    if command in {"ingest", "sync"}:
        gmail = gmail_client(paths)
        hook = feedback_hook(paths, store, gmail)
        if command == "ingest":
            result = ingest(
                gmail, store, args.query or settings.query, args.limit, args.restart, hook
            )
        else:
            result = sync(gmail, store, settings.query, args.limit, hook)
        return {**result, "report": str(audit(store, paths))}
    if command == "audit":
        return {"report": str(audit(store, paths))}
    if command == "discover":
        return discover(store, paths, Ollama(settings), args.limit, args.method, args.clusters)
    if command == "approve":
        categories = approve(
            location(paths, args.file) if args.file else paths.proposals, paths, store
        )
        return {
            "approved": [
                {"id": c.id, "name": c.name, "examples": len(c.example_ids)} for c in categories
            ]
        }
    if command == "recover":
        return recover(store, gmail_client(paths))
    if command == "schedule":
        return generate(paths, args.interval, args.apply, args.limit)
    categories = approved(paths, store)
    if command == "categories":
        return [{"id": c.id, "name": c.name, "description": c.description} for c in categories]
    if command == "classify":
        return preview(
            store,
            paths,
            categories,
            Ollama(settings),
            args.limit,
            args.include_processed,
            args.refresh,
        )
    if command == "apply":
        gmail = gmail_client(paths, write=True) if args.apply else None
        return apply_plan(
            store,
            categories,
            settings,
            store.plan(args.plan),
            gmail,
            args.apply,
            args.limit,
            args.retry_held,
        )
    if command == "correct":
        if args.clear:
            with store.db:
                store.db.execute("DELETE FROM overrides WHERE id=?", (args.message_id,))
            return {"local_override_cleared": args.message_id, "gmail_changed": False}
        return correction(store, args.message_id, args.category or [], categories)
    if command == "feedback":
        gmail = gmail_client(paths)
        connect_account(gmail, store)
        store.enqueue([r[0] for r in store.db.execute("SELECT id FROM writes WHERE status='done'")])
        count = drain(gmail, store, args.limit, observer(store, categories, gmail.labels()))
        return {
            "refreshed": count,
            "manual_examples": len(store.overrides()),
            "pending": store.counts()["pending"],
        }
    if command == "eval-sample":
        return export_sample(store, paths, categories, args.count)
    if command == "evaluate":
        return evaluate(store, paths, categories, Ollama(settings), location(paths, args.file))
    if command == "run":
        gmail = gmail_client(paths)
        synced = sync(
            gmail,
            store,
            settings.query,
            max(1000, args.limit),
            observer(store, categories, gmail.labels()),
        )
        predicted = preview(store, paths, categories, Ollama(settings), args.limit)
        result = {"sync": synced, "preview": predicted}
        if args.apply:
            result["apply"] = apply_plan(
                store,
                categories,
                settings,
                store.plan(),
                gmail_client(paths, True),
                True,
                args.limit,
            )
        return result
    raise SorterError("Unknown command.")


def main(argv=None):
    args = parser().parse_args(argv)
    # Private files created by libraries (SQLite WAL and OAuth state included).
    os.umask(0o077)
    try:
        paths = Paths(args.root)
        settings = Settings.load(paths.root)
        if args.command in {"doctor", "status"}:
            result = doctor(paths, settings) if args.command == "doctor" else status(paths)
        else:
            paths.prepare()
            with RunLock(paths.data / "run.lock"):
                store = Store(paths.database)
                try:
                    result = dispatch(args, paths, settings, store)
                finally:
                    store.close()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except SorterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Queued work and write records are retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Third-party exception text can include tokens, URLs, or email content.
        print(
            f"Operation failed ({type(exc).__name__}). Private error details were suppressed. Check doctor/status and retry; run recover after an interrupted apply.",
            file=sys.stderr,
        )
        return 1
