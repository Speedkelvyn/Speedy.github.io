"""End-to-end run against an in-memory fake of the Gmail API."""

import argparse
import base64
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gmail_reader as G


class Call:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeGmail:
    def __init__(self, n=1200):
        b = lambda s: base64.urlsafe_b64encode(s.encode()).decode()
        self.msgs = {}
        for i in range(n):
            subj = "Urgent invoice" if i % 10 == 0 else f"Newsletter {i}"
            self.msgs[f"m{i}"] = {
                "id": f"m{i}",
                "snippet": "s",
                "labelIds": ["UNREAD"],
                "payload": {
                    "headers": [{"name": "Subject", "value": subj}, {"name": "From", "value": "x@y.com"}],
                    "body": {"data": b("hello")},
                },
            }
        self.labels_store, self.sent = [], []

    # users().messages() / users().labels()
    def users(self):
        return self

    def messages(self):
        return self

    def labels(self):
        return self

    def list(self, userId, q=None, maxResults=None, pageToken=None):
        if q is None:  # labels().list
            return Call(lambda: {"labels": self.labels_store})
        unread = [k for k, m in self.msgs.items() if "UNREAD" in m["labelIds"]]
        start = int(pageToken or 0)
        page = unread[start : start + maxResults]
        res = {"messages": [{"id": k} for k in page], "resultSizeEstimate": len(unread)}
        if start + maxResults < len(unread):
            res["nextPageToken"] = str(start + maxResults)
        return Call(lambda: res)

    def get(self, userId, id, format):
        return Call(lambda: self.msgs[id])

    def create(self, userId, body):
        self.labels_store.append({"name": body["name"], "id": "L1"})
        return Call(lambda: {"id": "L1"})

    def modify(self, userId, id, body):
        return Call(lambda: self.msgs[id]["labelIds"].extend(body["addLabelIds"]))

    def batchModify(self, userId, body):
        def run():
            for i in body["ids"]:
                self.msgs[i]["labelIds"].remove("UNREAD")

        return Call(run)

    def send(self, userId, body):
        return Call(lambda: self.sent.append(body))


def make(tmp_path, monkeypatch, dry_run):
    fake = FakeGmail()
    monkeypatch.setattr(G, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(G, "HERE", tmp_path)
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    app = G.InboxZero(argparse.Namespace(dry_run=dry_run, limit=None, no_label=False))
    monkeypatch.setattr(app, "authenticate", lambda: setattr(app, "service", fake))
    return app, fake


def test_full_run_clears_inbox_and_labels_priority(tmp_path, monkeypatch):
    app, fake = make(tmp_path, monkeypatch, dry_run=False)
    app.run()
    assert len(app.unread_ids) == 1200  # paged past the first 500
    assert len(app.priority) == 20  # every 10th of the 200 analysed
    assert all("UNREAD" not in m["labelIds"] for m in fake.msgs.values())
    assert sum("L1" in m["labelIds"] for m in fake.msgs.values()) == 20
    assert len(list((tmp_path / "reports").glob("*.txt"))) == 1


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    app, fake = make(tmp_path, monkeypatch, dry_run=True)
    app.run()
    assert all("UNREAD" in m["labelIds"] for m in fake.msgs.values())
    assert not fake.labels_store and not fake.sent
