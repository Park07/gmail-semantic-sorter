from __future__ import annotations

import csv
import html
import io
import json
from collections import Counter

from .config import atomic_write


def cell(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def csv_cell(value):
    text = cell(value)
    leading_controls = "".join(chr(i) for i in range(33)) + "\ufeff"
    if text.lstrip(leading_controls).startswith(("=", "+", "-", "@")) or (
        text and (ord(text[0]) < 32 or text[0] == "\ufeff")
    ):
        return "'" + text
    return text


def report(paths, stem, title, rows, summary=None, columns=None):
    columns = columns or list(rows[0] if rows else {"status": ""})
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows([csv_cell(row.get(k, "")) for k in columns] for row in rows)
    atomic_write(paths.reports / f"{stem}.csv", stream.getvalue())
    headers = "".join(f"<th>{html.escape(k)}</th>" for k in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell(row.get(k, '')))}</td>" for k in columns) + "</tr>"
        for row in rows
    )
    description = html.escape(json.dumps(summary or {}, ensure_ascii=False, indent=2))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{html.escape(title)}</title><style>
body{{font:15px system-ui,sans-serif;margin:32px;color:#172338;background:#fafbfc}}
h1{{font-size:25px}}pre{{white-space:pre-wrap;background:#edf2f7;padding:16px;border-radius:8px}}
input{{padding:12px;width:min(500px,90%);margin:16px 0;border:1px solid #aab8c5;border-radius:6px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;border:1px solid #dce3ea;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}}
th{{background:#edf2f7}}tr:nth-child(even){{background:#f6f8fa}}.table{{overflow:auto}}
</style><h1>{html.escape(title)}</h1><p>Private local report. Suggestions do not change Gmail.</p>
<pre>{description}</pre><label>Filter messages <input id="filter" placeholder="Type a subject, category, or sender"></label>
<div class="table"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>"""
    atomic_write(paths.reports / f"{stem}.html", page)
    return paths.reports / f"{stem}.html"


def rule_hints(message):
    result = []
    if (
        message.headers.get("list-unsubscribe")
        or message.headers.get("list-id")
        or "CATEGORY_PROMOTIONS" in message.labels
    ):
        result.append("mailing-list/promotions hint")
    if (
        any(part in message.sender for part in ("noreply", "no-reply", "notifications@"))
        or "CATEGORY_UPDATES" in message.labels
    ):
        result.append("notification hint")
    return result


def audit(store, paths):
    messages = store.messages()
    domains = Counter(m.domain for m in messages)
    gmail_categories = Counter(
        label for m in messages for label in m.labels if label.startswith("CATEGORY_")
    )
    hits = sum(bool(rule_hints(m)) for m in messages)
    summary = {
        "messages": len(messages),
        "rule_hint_coverage": hits / len(messages) if messages else 0,
        "note": "Hints describe mail type, not importance or a final topic assignment.",
        "gmail_categories": dict(gmail_categories),
        "top_domains": dict(domains.most_common(25)),
    }
    rows = [
        {
            "message_id": m.id,
            "sender": m.sender,
            "subject": m.subject,
            "snippet": m.snippet,
            "gmail_labels": m.labels,
            "rule_hints": rule_hints(m),
        }
        for m in messages
    ]
    return report(paths, "audit", "Inbox audit", rows, summary)
