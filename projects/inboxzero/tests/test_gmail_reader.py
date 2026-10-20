"""Scoring, parsing and report logic (no Gmail account needed)."""

import base64
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gmail_reader as G


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def message(subject="Hello", sender="a@b.com", body="", parts=None, labels=None):
    payload = {
        "headers": [
            {"name": "Subject", "value": subject},
            {"name": "From", "value": sender},
            {"name": "Date", "value": "Mon, 1 Jan 2026"},
            {"name": "X-Other", "value": "x"},
        ]
    }
    if parts is not None:
        payload["parts"] = parts
    else:
        payload["body"] = {"data": b64(body)}
    return {"id": "m1", "payload": payload, "snippet": "snip", "labelIds": labels or []}


def test_parse_single_part():
    e = G.parse_message(message(body="Pay the invoice today"))
    assert (e.subject, e.sender, e.body) == ("Hello", "a@b.com", "Pay the invoice today")


def test_parse_multipart_and_attachments():
    parts = [
        {"mimeType": "text/html", "body": {"data": b64("<b>x</b>")}},
        {"mimeType": "text/plain", "body": {"data": b64("plain text")}},
        {"mimeType": "application/pdf", "filename": "contract.pdf", "body": {}},
    ]
    e = G.parse_message(message(parts=parts))
    assert e.body == "plain text" and e.attachments == ["contract.pdf"]


def test_bad_base64_is_ignored():
    assert G.decode_body({"body": {"data": "!!!not-base64"}}) == ""


def test_scoring_matches_documented_weights():
    e = G.score_email(
        G.parse_message(
            message(
                subject="URGENT: contract deadline",
                sender="Boss <boss@company.com>",
                body="see the invoice",
                labels=["IMPORTANT"],
            )
        )
    )
    # sender 3 + subject keywords (urgent, deadline, contract) 3×2 + body keyword (invoice) 1 + Gmail important 2
    assert e.score == 3 + 6 + 1 + 2
    assert any("important sender" in r for r in e.reasons)


def test_unimportant_email_scores_zero():
    assert G.score_email(G.parse_message(message(subject="Weekend photos", body="nice"))).score == 0


def test_report_lists_priority_first():
    a = G.Email(id="1", subject="Low", score=1, snippet="a")
    b = G.Email(id="2", subject="High", score=9, snippet="b", reasons=["important sender"])
    text = G.build_report(sorted([a, b], key=lambda e: e.score, reverse=True), total_unread=40, dry_run=True)
    assert text.index("High") < text.index("Low")
    assert "Total unread emails: 40" in text and "DRY RUN" in text


def test_cli_version(capsys):
    try:
        G.main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert G.__version__ in capsys.readouterr().out
