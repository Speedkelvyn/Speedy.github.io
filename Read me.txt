# Gmail Inbox Cleaner

A simple Python script that cleans your Gmail inbox by marking all unread emails as read and highlighting the important ones.

## What It Does

- Reads all your unread emails
- Finds important messages (based on keywords and senders you choose)
- Labels the important ones
- Marks ALL emails as read (inbox zero!)
- Sends you a summary of what was important

## Requirements

- Python 3.8 or newer
- A Gmail account
- 10 minutes to set up

## Setup

### 1. Install Python Libraries

```bash
pip install -r requirements.txt
```

### 2. Get Gmail API Access

1. Go to https://console.cloud.google.com/
2. Create a new project
3. Enable Gmail API
4. Create OAuth credentials (Desktop app)
5. Download the credentials file and rename it to `credentials.json`
6. Put `credentials.json` in the same folder as the script

### 3. Add Your Email as Test User

1. In Google Cloud Console, go to "OAuth consent screen"
2. Scroll to "Test users"
3. Add your Gmail address

### 4. Edit Settings

Open the script and change these lines (around line 27-40):

```python
IMPORTANT_SENDERS = [
    'boss@company.com',      # Add your important people
    '@yourcompany.com',      # Or company domains
]

IMPORTANT_KEYWORDS = [
    'urgent', 'deadline', 'invoice'  # Add your keywords
]

YOUR_EMAIL = 'youremail@gmail.com'  # Your actual email
```

## How to Use

Run the script:

```bash
python gmail_reader.py
```

First time:
- A browser will open
- Sign in with Gmail
- Click "Allow"
- Close browser
- Script runs automatically

Next times:
- Just run the script
- No browser needed

## What You'll Get

- All unread emails marked as read
- Important emails labeled "PRIORITY_INBOX"
- A text file report saved locally
- Summary email sent to you (optional)

## Troubleshooting

**"credentials.json not found"**
- Make sure the file is in the same folder as the script
- Make sure it's named exactly `credentials.json`

**"Error 403: access_denied"**
- Go to OAuth consent screen in Google Cloud
- Add yourself as a test user

**"pip is not recognized"**
- Try: `python -m pip install -r requirements.txt`

## Security

- Your emails stay on Google's servers
- Nothing is uploaded anywhere
- Script runs on your computer only
- Keep `credentials.json` and `token.json` private!

## License

Free to use and modify however you want!