import plistlib
from pathlib import Path
import sys

from .config import atomic_write


def generate(paths, interval=3600, apply=False, limit=100):
    # Generation only. Activation is a separate, documented launchctl command.
    arguments = [
        str(Path(sys.executable).absolute()),
        "-m",
        "sorter",
        "--root",
        str(paths.root),
        "run",
        "--limit",
        str(limit),
    ]
    if apply:
        arguments.append("--apply")
    document = {
        "Label": "local.gmail-semantic-sorter",
        "ProgramArguments": arguments,
        "WorkingDirectory": str(paths.root),
        "StartInterval": interval,
        "RunAtLoad": False,
        "ProcessType": "Background",
        "StandardOutPath": str(paths.data / "schedule.log"),
        "StandardErrorPath": str(paths.data / "schedule-error.log"),
        "Umask": 0o077,
    }
    path = paths.data / "local.gmail-semantic-sorter.plist"
    atomic_write(path, plistlib.dumps(document).decode())
    return {
        "plist": str(path),
        "interval_seconds": interval,
        "applies_labels": apply,
        "installed": False,
        "next": "Review the plist and follow the launchctl instructions in README.md.",
    }
