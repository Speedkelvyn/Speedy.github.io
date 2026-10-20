"""InboxZero: triage a Gmail inbox in one run.

Finds every unread email, scores each one for priority (sender, keywords in the subject
and body, attachments, Gmail's own "important" flag), labels the priority ones, writes a
report, optionally emails you the report, and marks everything as read.

    python gmail_reader.py --dry-run          # analyse and report only, change nothing
    python gmail_reader.py                    # full run
    python gmail_reader.py --limit 50         # analyse only the 50 newest unread emails
    python gmail_reader.py --no-label         # don't apply the priority label

Setup: enable the Gmail API in Google Cloud, download the OAuth client as
``credentials.json`` next to this script, and ``pip install -r requirements.txt``.
The first run opens a browser to sign in and saves ``token.json`` (never commit either file).
Set ``INBOXZERO_EMAIL`` (or edit ``SUMMARY_EMAIL`` below) to receive the report by email.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import itertools
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any

__version__ = "2.1"

# --------------------------------------------------------------------------- settings

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

IMPORTANT_SENDERS = [
    "boss@company.com",
    "client@important.com",
    "@company.com",  # a domain matches every sender at that domain
]

IMPORTANT_KEYWORDS = [
    "urgent",
    "important",
    "asap",
    "deadline",
    "invoice",
    "payment",
    "contract",
    "meeting",
    "action required",
    "time-sensitive",
    "critical",
]

PRIORITY_LABEL = "PRIORITY_INBOX"
SUMMARY_EMAIL = os.environ.get("INBOXZERO_EMAIL", "")  # leave empty to skip the summary email
MAX_EMAILS_TO_ANALYZE = 200
BATCH_SIZE = 1000  # Gmail's batchModify limit

HERE = pathlib.Path(__file__).resolve().parent
CREDENTIALS_FILE = HERE / "credentials.json"
TOKEN_FILE = HERE / "token.json"
REPORT_DIR = HERE / "reports"

# Scoring weights
SCORE_SENDER = 3
SCORE_SUBJECT_KEYWORD = 2
SCORE_BODY_KEYWORD = 1
SCORE_ATTACHMENTS = 1
SCORE_GMAIL_IMPORTANT = 2


# --------------------------------------------------------------------------- terminal output


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for name in ("RED", "GREEN", "YELLOW", "BLUE", "CYAN", "BOLD", "RESET"):
            setattr(cls, name, "")


def setup_colors() -> None:
    """Use colour only on a real terminal; on Windows enable it through colorama when available."""
    if not sys.stdout.isatty():
        Colors.disable()
        return
    if os.name == "nt":
        try:
            import colorama

            colorama.just_fix_windows_console()
        except (ImportError, AttributeError):
            Colors.disable()


def success(msg: str) -> None:
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def error(msg: str) -> None:
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def warn(msg: str) -> None:
    print(f"{Colors.YELLOW}! {msg}{Colors.RESET}")


def info(msg: str) -> None:
    print(f"{Colors.BLUE}· {msg}{Colors.RESET}")


def header(msg: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{msg}{Colors.RESET}\n{'─' * 72}")


class Spinner:
    """A small terminal spinner, used as a context manager around slow API calls."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            print(f"\r{Colors.CYAN}{frame} {self.message}…{Colors.RESET}", end="", flush=True)
            time.sleep(0.1)

    def __enter__(self) -> Spinner:
        if sys.stdout.isatty():
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        print("\r" + " " * (len(self.message) + 4), end="\r", flush=True)


def progress(current: int, total: int, label: str, started: float) -> None:
    """One-line progress bar with an ETA."""
    width = 36
    filled = int(width * current / total) if total else width
    eta = ""
    elapsed = time.time() - started
    if current and elapsed > 0:
        remaining = (total - current) / (current / elapsed)
        eta = (
            f"  ETA {int(remaining)}s"
            if remaining < 60
            else f"  ETA {int(remaining // 60)}m {int(remaining % 60)}s"
        )
    pct = 100 * current / total if total else 100
    print(
        f"\r{label} |{Colors.GREEN}{'█' * filled}{'░' * (width - filled)}{Colors.RESET}| "
        f"{pct:5.1f}% ({current}/{total}){eta}   ",
        end="" if current < total else "\n",
        flush=True,
    )


def summary_box(title: str, rows: dict[str, str]) -> None:
    width = max(len(title), *(len(f"{k}: {v}") for k, v in rows.items())) + 4
    print(f"\n┌{'─' * width}┐")
    print(f"│ {Colors.BOLD}{title:<{width - 2}}{Colors.RESET} │")
    print(f"├{'─' * width}┤")
    for k, v in rows.items():
        print(f"│ {f'{k}: {v}':<{width - 2}} │")
    print(f"└{'─' * width}┘")


# --------------------------------------------------------------------------- scoring (pure functions)


@dataclass
class Email:
    id: str
    subject: str = ""
    sender: str = ""
    date: str = ""
    snippet: str = ""
    body: str = ""
    attachments: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    score: int = 0
    reasons: list[str] = field(default_factory=list)


def is_important_sender(sender: str, important: list[str] = IMPORTANT_SENDERS) -> bool:
    s = sender.lower()
    return any(item.lower() in s for item in important)


def find_keywords(text: str, keywords: list[str] = IMPORTANT_KEYWORDS) -> list[str]:
    t = text.lower()
    return [k for k in keywords if k.lower() in t]


def score_email(email: Email) -> Email:
    """Add up the priority score and record why. Mutates and returns the email."""
    score, reasons = 0, []
    if is_important_sender(email.sender):
        score += SCORE_SENDER
        reasons.append("important sender")
    if subject_kw := find_keywords(email.subject):
        score += SCORE_SUBJECT_KEYWORD * len(subject_kw)
        reasons.append(f"subject keywords: {', '.join(subject_kw[:3])}")
    if body_kw := find_keywords(email.body):
        score += SCORE_BODY_KEYWORD * len(body_kw)
        reasons.append(f"body keywords: {', '.join(body_kw[:2])}")
    if email.attachments:
        score += SCORE_ATTACHMENTS
        reasons.append(f"{len(email.attachments)} attachment(s)")
    if "IMPORTANT" in email.labels:
        score += SCORE_GMAIL_IMPORTANT
        reasons.append("marked important by Gmail")
    email.score, email.reasons = score, reasons
    return email


def decode_body(payload: dict[str, Any], limit: int = 500) -> str:
    """First `limit` characters of the plain-text body (single-part or the first text/plain part)."""

    def _decode(data: str) -> str:
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")[:limit]
        except (binascii.Error, ValueError):
            return ""

    if data := payload.get("body", {}).get("data"):
        return _decode(data)
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            return _decode(part.get("body", {}).get("data", ""))
    return ""


def parse_message(message: dict[str, Any]) -> Email:
    payload = message.get("payload", {})
    hdrs = {
        h["name"]: h["value"] for h in payload.get("headers", []) if h["name"] in ("Subject", "From", "Date")
    }
    return Email(
        id=message["id"],
        subject=hdrs.get("Subject", ""),
        sender=hdrs.get("From", ""),
        date=hdrs.get("Date", ""),
        snippet=message.get("snippet", ""),
        body=decode_body(payload),
        attachments=[p["filename"] for p in payload.get("parts", []) if p.get("filename")],
        labels=message.get("labelIds", []),
    )


def build_report(priority: list[Email], total_unread: int, dry_run: bool) -> str:
    lines = [
        "=" * 80,
        "GMAIL PRIORITY EMAIL REPORT",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 80,
        "",
        "SUMMARY",
        "-" * 80,
        f"Total unread emails: {total_unread}",
        f"Priority emails found: {len(priority)}",
        "DRY RUN: no emails were changed" if dry_run else "All unread emails were marked as read",
        "",
        "=" * 80,
    ]
    if not priority:
        lines += ["No high-priority emails found.", "=" * 80]
    else:
        lines += ["HIGH-PRIORITY EMAILS", "=" * 80, ""]
        for i, e in enumerate(priority, 1):
            lines += [f"{i}. [{e.score} points] {e.subject}", f"   From: {e.sender}", f"   Date: {e.date}"]
            if e.reasons:
                lines.append(f"   Why: {'; '.join(e.reasons)}")
            if e.attachments:
                lines.append(f"   Attachments: {', '.join(e.attachments)}")
            lines += [f"   Preview: {e.snippet[:150]}…", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- Gmail client


class InboxZero:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.service: Any = None
        self.unread_ids: list[str] = []
        self.priority: list[Email] = []
        self.others: list[Email] = []

    # -- API helpers
    def authenticate(self) -> None:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = (
            Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES) if TOKEN_FILE.exists() else None
        )
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CREDENTIALS_FILE.exists():
                    raise SystemExit(
                        f"Missing {CREDENTIALS_FILE.name}: download the OAuth client from Google Cloud."
                    )
                info("Opening your browser to sign in to Gmail…")
                creds = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), SCOPES
                ).run_local_server(port=0)
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        self.service = build("gmail", "v1", credentials=creds)
        success("Signed in to Gmail")

    def _messages(self) -> Any:
        return self.service.users().messages()

    def fetch_unread_ids(self) -> list[str]:
        ids: list[str] = []
        token = None
        with Spinner("Fetching unread emails"):
            while True:
                kw = {"userId": "me", "q": "is:unread", "maxResults": 500}
                if token:
                    kw["pageToken"] = token
                res = self._messages().list(**kw).execute()
                ids += [m["id"] for m in res.get("messages", [])]
                if not (token := res.get("nextPageToken")):
                    break
        success(f"Found {len(ids)} unread emails")
        return ids

    def label_id(self, name: str) -> str | None:
        labels = self.service.users().labels()
        for label in labels.list(userId="me").execute().get("labels", []):
            if label["name"] == name:
                return label["id"]
        body = {"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
        created = labels.create(userId="me", body=body).execute()
        success(f"Created label {name}")
        return created["id"]

    # -- steps
    def analyze(self) -> None:
        header("Analysing unread emails")
        self.unread_ids = self.fetch_unread_ids()
        if not self.unread_ids:
            return
        limit = self.args.limit or MAX_EMAILS_TO_ANALYZE or len(self.unread_ids)
        batch = self.unread_ids[:limit]
        info(f"Scoring the {len(batch)} newest of {len(self.unread_ids)} unread emails")
        started = time.time()
        for i, msg_id in enumerate(batch, 1):
            progress(i, len(batch), "Analysing", started)
            try:
                msg = self._messages().get(userId="me", id=msg_id, format="full").execute()
            except Exception as exc:  # one unreadable message should not stop the run
                warn(f"Skipped {msg_id}: {exc}")
                continue
            email = score_email(parse_message(msg))
            (self.priority if email.score > 0 else self.others).append(email)
        self.priority.sort(key=lambda e: e.score, reverse=True)
        success(f"{len(self.priority)} high-priority, {len(self.others)} other")

    def label_priority(self) -> None:
        if not self.priority or self.args.no_label or self.args.dry_run:
            return
        header(f"Labelling {len(self.priority)} priority emails")
        label = self.label_id(PRIORITY_LABEL)
        started = time.time()
        for i, e in enumerate(self.priority, 1):
            self._messages().modify(userId="me", id=e.id, body={"addLabelIds": [label]}).execute()
            progress(i, len(self.priority), "Labelling", started)

    def save_report(self) -> str:
        text = build_report(self.priority, len(self.unread_ids), self.args.dry_run)
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / f"gmail_priority_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path.write_text(text, encoding="utf-8")
        success(f"Report saved to {path.relative_to(HERE)}")
        return text

    def send_summary(self, report: str) -> None:
        if not SUMMARY_EMAIL or self.args.dry_run:
            return
        msg = MIMEText(report)
        msg["to"] = SUMMARY_EMAIL
        msg["subject"] = f"Gmail priority report: {len(self.priority)} important emails"
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        self._messages().send(userId="me", body={"raw": raw}).execute()
        success(f"Summary emailed to {SUMMARY_EMAIL}")

    def mark_all_read(self) -> int | None:
        """Mark every unread email as read in batches. Returns the unread count Gmail reports afterwards."""
        if self.args.dry_run:
            warn(f"Dry run: would mark {len(self.unread_ids)} emails as read")
            return None
        header(f"Marking {len(self.unread_ids)} emails as read")
        started, done = time.time(), 0
        for start in range(0, len(self.unread_ids), BATCH_SIZE):
            ids = self.unread_ids[start : start + BATCH_SIZE]
            try:
                self._messages().batchModify(
                    userId="me", body={"ids": ids, "removeLabelIds": ["UNREAD"]}
                ).execute()
                done += len(ids)
            except Exception as exc:
                error(f"Batch starting at {start} failed: {str(exc)[:80]}")
            progress(
                min(start + len(ids), len(self.unread_ids)), len(self.unread_ids), "Marking read", started
            )
            time.sleep(0.3)  # stay well under Gmail's rate limits
        time.sleep(2)  # give Gmail a moment before checking
        remaining = self._messages().list(userId="me", q="is:unread", maxResults=1).execute()
        left = int(remaining.get("resultSizeEstimate", 0))
        (success if left == 0 else warn)(f"{done} marked read; {left} still unread")
        return left

    def run(self) -> None:
        mode = "DRY RUN" if self.args.dry_run else "LIVE"
        print(f"{Colors.BOLD}InboxZero v{__version__}{Colors.RESET} · {mode}")
        started = time.time()
        self.authenticate()
        self.analyze()
        if not self.unread_ids:
            success("Inbox is already at zero. Nothing to do.")
            return
        self.label_priority()
        report = self.save_report()
        self.send_summary(report)
        left = self.mark_all_read()
        summary_box(
            "Dry run complete" if self.args.dry_run else "Inbox cleaned",
            {
                "Unread found": str(len(self.unread_ids)),
                "Priority": str(len(self.priority)),
                "Still unread": "unchanged (dry run)" if left is None else str(left),
                "Time": f"{int(time.time() - started)}s",
            },
        )


def main(argv: list[str] | None = None) -> int:
    setup_colors()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="analyse and report only; change nothing")
    ap.add_argument(
        "--limit", type=int, metavar="N", help=f"analyse only N emails (default {MAX_EMAILS_TO_ANALYZE})"
    )
    ap.add_argument("--no-label", action="store_true", help="don't apply the priority label")
    ap.add_argument("--version", action="version", version=f"InboxZero {__version__}")
    args = ap.parse_args(argv)
    try:
        InboxZero(args).run()
    except KeyboardInterrupt:
        warn("Stopped.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
