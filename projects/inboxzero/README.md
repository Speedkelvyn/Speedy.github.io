# InboxZero

Triage a Gmail inbox in one run: score every unread email for priority, label the ones that matter, save a report, and mark the rest as read.

## What it does

1. Finds every unread email (all pages, not just the first 100).
2. Scores each one: important sender (+3), each subject keyword (+2), each body keyword (+1), attachments (+1), flagged important by Gmail (+2).
3. Applies a `PRIORITY_INBOX` label to anything that scores above zero.
4. Saves a text report to `reports/` and, if configured, emails it to you.
5. Marks all unread emails as read in batches of 1,000, then checks Gmail's unread count.

## Setup (about 10 minutes)

```bash
pip install -r requirements.txt
```

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable the **Gmail API**.
2. Create **OAuth credentials** (Desktop app), download them, and save the file as `credentials.json` next to `gmail_reader.py`.
3. Under **OAuth consent screen → Test users**, add your Gmail address.
4. Edit `IMPORTANT_SENDERS` and `IMPORTANT_KEYWORDS` at the top of `gmail_reader.py`.
5. Optional: to receive the report by email, set `INBOXZERO_EMAIL`:
   ```bash
   export INBOXZERO_EMAIL="you@gmail.com"      # macOS / Linux
   set INBOXZERO_EMAIL=you@gmail.com           # Windows
   ```

## Usage

```bash
python gmail_reader.py --dry-run     # analyse and report only; nothing in Gmail changes
python gmail_reader.py               # full run
python gmail_reader.py --limit 50    # score only the 50 newest unread emails
python gmail_reader.py --no-label    # skip the priority label
```

The first run opens a browser to sign in and saves `token.json`; later runs reuse it.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Missing credentials.json` | Put the downloaded OAuth file next to the script, named exactly `credentials.json`. |
| `Error 403: access_denied` | Add your address as a test user on the OAuth consent screen. |
| `pip` not recognised | Use `python -m pip install -r requirements.txt`. |

## Privacy

Everything runs on your computer and talks only to Gmail's API. `credentials.json`, `token.json` and `reports/` are listed in `.gitignore`; never commit them.

## Tests

```bash
python -m pytest tests
```

The tests cover scoring, body decoding and the report, and need no Google account.
