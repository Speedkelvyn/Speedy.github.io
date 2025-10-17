"""
Gmail Priority Email Reader and Analyzer - ENHANCED VERSION
Reads unread emails, identifies important ones, sends summary, and marks all as read

NEW FEATURES:
- Command-line options for flexibility
- Progress bars for better UX
- Improved performance with batch processing
- Beautiful colored output
- Spinning loaders
- ETA calculations
- Summary stats box
"""

import os.path
import base64
from datetime import datetime
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import re
import argparse
import sys
import threading

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# ============= CUSTOMIZE THESE SETTINGS =============

IMPORTANT_SENDERS = [
    'boss@company.com',
    'client@important.com',
    '@company.com',
]

IMPORTANT_KEYWORDS = [
    'urgent',
    'important',
    'asap',
    'deadline',
    'invoice',
    'payment',
    'contract',
    'meeting',
    'action required',
    'time-sensitive',
    'critical',
]

PRIORITY_LABEL = 'PRIORITY_INBOX'
YOUR_EMAIL = 'youremail@gmail.com'
MAX_EMAILS_TO_ANALYZE = 200

# ====================================================

# ANSI Color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.RED = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.MAGENTA = ''
        Colors.CYAN = ''
        Colors.WHITE = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.RESET = ''


class SpinnerLoader:
    def __init__(self, message="Loading"):
        self.message = message
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        
    def spin(self):
        idx = 0
        while self.running:
            print(f'\r{Colors.CYAN}{self.spinner_chars[idx]} {self.message}...{Colors.RESET}', end='', flush=True)
            idx = (idx + 1) % len(self.spinner_chars)
            time.sleep(0.1)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
    
    def stop(self, final_message=None):
        self.running = False
        if self.thread:
            self.thread.join()
        if final_message:
            print(f'\r{final_message}' + ' ' * 20)
        else:
            print('\r' + ' ' * 50, end='\r')


def print_progress_bar(current, total, prefix='', suffix='', length=40, start_time=None):
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    
    eta_str = ''
    if start_time and current > 0:
        elapsed = time.time() - start_time
        rate = current / elapsed
        if rate > 0:
            remaining = (total - current) / rate
            if remaining < 60:
                eta_str = f' - ETA: {int(remaining)}s'
            else:
                eta_str = f' - ETA: {int(remaining/60)}m {int(remaining%60)}s'
    
    print(f'\r{Colors.BLUE}{prefix}{Colors.RESET} |{Colors.GREEN}{bar}{Colors.RESET}| {Colors.YELLOW}{percent:.1f}%{Colors.RESET} {suffix}{Colors.CYAN}{eta_str}{Colors.RESET}', end='', flush=True)
    if current == total:
        print()


def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")


def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")


def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")


def print_header(message):
    print(f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}")


def print_summary_box(stats):
    width = 45
    
    print(f"\n{Colors.CYAN}┌{'─' * (width - 2)}┐{Colors.RESET}")
    print(f"{Colors.CYAN}│{Colors.BOLD}{Colors.GREEN}{'INBOX CLEANING COMPLETE! 🎉':^{width - 2}}{Colors.RESET}{Colors.CYAN}│{Colors.RESET}")
    print(f"{Colors.CYAN}├{'─' * (width - 2)}┤{Colors.RESET}")
    
    for key, value in stats.items():
        line = f"  {key}: {value}"
        padding = width - 2 - len(line)
        print(f"{Colors.CYAN}│{Colors.RESET}{line}{' ' * padding}{Colors.CYAN}│{Colors.RESET}")
    
    print(f"{Colors.CYAN}└{'─' * (width - 2)}┘{Colors.RESET}\n")


class GmailPriorityReader:
    def __init__(self, args):
        self.service = None
        self.priority_emails = []
        self.other_unread = []
        self.all_unread_ids = []
        self.args = args
        self.start_time = None
        
    def authenticate(self):
        spinner = SpinnerLoader("Authenticating with Gmail")
        spinner.start()
        
        creds = None
        
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                spinner.stop()
                print_info("Opening browser for authentication...")
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                spinner.start()
            
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        spinner.stop()
        print_success("Successfully authenticated with Gmail!")
        
    def get_or_create_label(self, label_name):
        try:
            results = self.service.users().labels().list(userId='me').execute()
            labels = results.get('labels', [])
            
            for label in labels:
                if label['name'] == label_name:
                    return label['id']
            
            label_object = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            created_label = self.service.users().labels().create(
                userId='me', body=label_object).execute()
            
            print_success(f"Created new label: {label_name}")
            return created_label['id']
            
        except Exception as e:
            print_error(f"Error with label: {e}")
            return None
    
    def is_important_sender(self, sender):
        sender_lower = sender.lower()
        for important in IMPORTANT_SENDERS:
            if important.lower() in sender_lower:
                return True
        return False
    
    def has_important_keywords(self, text):
        text_lower = text.lower()
        found_keywords = []
        for keyword in IMPORTANT_KEYWORDS:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        return found_keywords
    
    def get_email_body_quick(self, payload):
        body_text = ''
        
        if 'body' in payload and 'data' in payload['body']:
            try:
                body_data = payload['body']['data']
                body_text = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
                return body_text[:500]
            except:
                pass
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        try:
                            body_data = part['body']['data']
                            body_text = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
                            return body_text[:500]
                        except:
                            pass
                    break
        
        return body_text
    
    def get_email_details(self, message_id):
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full').execute()
            
            headers = message['payload']['headers']
            
            subject = ''
            sender = ''
            date = ''
            
            for header in headers:
                name = header['name']
                if name == 'Subject':
                    subject = header['value']
                elif name == 'From':
                    sender = header['value']
                elif name == 'Date':
                    date = header['value']
                
                if subject and sender and date:
                    break
            
            has_attachments = False
            attachment_names = []
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    filename = part.get('filename')
                    if filename:
                        has_attachments = True
                        attachment_names.append(filename)
            
            full_body = self.get_email_body_quick(message['payload'])
            snippet = message.get('snippet', '')
            
            return {
                'id': message_id,
                'subject': subject,
                'sender': sender,
                'date': date,
                'snippet': snippet,
                'full_body': full_body,
                'has_attachments': has_attachments,
                'attachment_names': attachment_names,
                'labels': message.get('labelIds', [])
            }
            
        except Exception as e:
            return None
    
    def get_all_unread_ids(self):
        print_info("Finding ALL unread emails...")
        
        spinner = SpinnerLoader("Fetching email list")
        spinner.start()
        
        all_message_ids = []
        page_token = None
        
        while True:
            if page_token:
                results = self.service.users().messages().list(
                    userId='me', 
                    q='is:unread',
                    pageToken=page_token,
                    maxResults=500
                ).execute()
            else:
                results = self.service.users().messages().list(
                    userId='me',
                    q='is:unread',
                    maxResults=500
                ).execute()
            
            messages = results.get('messages', [])
            
            for msg in messages:
                all_message_ids.append(msg['id'])
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
        spinner.stop()
        print_success(f"Found {Colors.BOLD}{len(all_message_ids)}{Colors.RESET}{Colors.GREEN} unread emails!{Colors.RESET}")
        return all_message_ids
    
    def analyze_emails(self):
        print_header("\n📊 Analyzing emails for priority...")
        print("=" * 80)
        
        try:
            self.all_unread_ids = self.get_all_unread_ids()
            
            if not self.all_unread_ids:
                print_warning("No unread emails found!")
                return
            
            total_unread = len(self.all_unread_ids)
            
            limit = self.args.limit if self.args.limit else (MAX_EMAILS_TO_ANALYZE or total_unread)
            emails_to_analyze = self.all_unread_ids[:limit]
            
            print(f"\n{Colors.CYAN}📬 Total unread emails:{Colors.RESET} {Colors.BOLD}{total_unread}{Colors.RESET}")
            print(f"{Colors.CYAN}🔍 Analyzing {Colors.BOLD}{len(emails_to_analyze)}{Colors.RESET}{Colors.CYAN} emails in detail...{Colors.RESET}\n")
            
            analysis_start = time.time()
            
            for idx, msg_id in enumerate(emails_to_analyze, 1):
                print_progress_bar(idx, len(emails_to_analyze), 
                                 prefix='Analyzing emails:',
                                 suffix=f'({idx}/{len(emails_to_analyze)})',
                                 start_time=analysis_start)
                
                email = self.get_email_details(msg_id)
                if not email:
                    continue
                
                priority_score = 0
                priority_reasons = []
                
                if self.is_important_sender(email['sender']):
                    priority_score += 3
                    priority_reasons.append('Important sender')
                
                keywords = self.has_important_keywords(email['subject'])
                if keywords:
                    priority_score += len(keywords) * 2
                    priority_reasons.append(f"Subject keywords: {', '.join(keywords[:3])}")
                
                body_keywords = self.has_important_keywords(email['full_body'])
                if body_keywords:
                    priority_score += len(body_keywords)
                    priority_reasons.append(f"Body keywords: {', '.join(body_keywords[:2])}")
                
                if email['has_attachments']:
                    priority_score += 1
                    priority_reasons.append(f"{len(email['attachment_names'])} attachment(s)")
                
                if 'IMPORTANT' in email['labels']:
                    priority_score += 2
                    priority_reasons.append('Gmail important')
                
                email['priority_score'] = priority_score
                email['priority_reasons'] = priority_reasons
                
                if priority_score > 0:
                    self.priority_emails.append(email)
                else:
                    self.other_unread.append(email)
            
            self.priority_emails.sort(key=lambda x: x['priority_score'], reverse=True)
            
            print(f"\n{'='*80}")
            print_success("Analysis complete!")
            print(f"   {Colors.RED}🔴 High priority emails:{Colors.RESET} {Colors.BOLD}{len(self.priority_emails)}{Colors.RESET}")
            print(f"   {Colors.WHITE}⚪ Other emails analyzed:{Colors.RESET} {len(self.other_unread)}")
            print(f"   {Colors.CYAN}📬 Total unread:{Colors.RESET} {total_unread}")
            
        except Exception as e:
            print_error(f"Error analyzing emails: {e}")
    
    def label_priority_emails(self):
        if not self.priority_emails:
            print_info("No priority emails to label.")
            return
        
        if self.args.no_label:
            print_warning("Skipping labeling (--no-label flag)")
            return
        
        print(f"\n{Colors.CYAN}🏷️  Labeling {Colors.BOLD}{len(self.priority_emails)}{Colors.RESET}{Colors.CYAN} priority emails...{Colors.RESET}")
        
        label_id = self.get_or_create_label(PRIORITY_LABEL)
        if not label_id:
            return
        
        try:
            for i, email in enumerate(self.priority_emails, 1):
                self.service.users().messages().modify(
                    userId='me',
                    id=email['id'],
                    body={'addLabelIds': [label_id]}
                ).execute()
                
                if i % 5 == 0 or i == len(self.priority_emails):
                    print(f"   {Colors.CYAN}Labeled {i}/{len(self.priority_emails)}...{Colors.RESET}", end='\r')
            
            print_success(f"Successfully labeled all {len(self.priority_emails)} priority emails!     ")
            
        except Exception as e:
            print_error(f"Error labeling emails: {e}")
    
    def create_report(self):
        print_info("Creating priority report...")
        
        report = []
        report.append("=" * 80)
        report.append("GMAIL PRIORITY EMAIL REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total unread emails: {len(self.all_unread_ids)}")
        report.append(f"Priority emails found: {len(self.priority_emails)}")
        if not self.args.dry_run:
            report.append(f"All emails will be marked as READ")
        else:
            report.append(f"DRY RUN - No emails will be marked as read")
        report.append("")
        
        if self.priority_emails:
            report.append("=" * 80)
            report.append("🔴 HIGH PRIORITY EMAILS (Action Required)")
            report.append("=" * 80)
            report.append("")
            
            for i, email in enumerate(self.priority_emails, 1):
                report.append(f"{i}. [{email['priority_score']} points] {email['subject']}")
                report.append(f"   From: {email['sender']}")
                report.append(f"   Date: {email['date']}")
                if email['priority_reasons']:
                    report.append(f"   Why important: {', '.join(email['priority_reasons'])}")
                if email['attachment_names']:
                    report.append(f"   Attachments: {', '.join(email['attachment_names'])}")
                report.append(f"   Preview: {email['snippet'][:150]}...")
                report.append("")
        else:
            report.append("=" * 80)
            report.append("No high priority emails found.")
            report.append("=" * 80)
            report.append("")
        
        report_text = "\n".join(report)
        
        filename = f"gmail_priority_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print_success(f"Report saved to: {Colors.BOLD}{filename}{Colors.RESET}")
        return report_text
    
    def send_summary_email(self, report_text):
        if YOUR_EMAIL == 'your.email@gmail.com' or YOUR_EMAIL == 'youremail@gmail.com':
            print_warning("Email notification disabled. Update YOUR_EMAIL in script.")
            return
        
        if self.args.dry_run:
            print_warning("Skipping email send (dry-run mode)")
            return
        
        print_info("Sending summary email...")
        
        try:
            message = MIMEText(report_text)
            message['to'] = YOUR_EMAIL
            message['subject'] = f"📬 Gmail Priority Report - {len(self.priority_emails)} Important Emails"
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            print_success(f"Summary email sent to {YOUR_EMAIL}")
            
        except Exception as e:
            print_error(f"Error sending email: {e}")
    
    def mark_all_as_read(self):
        if not self.all_unread_ids:
            print_info("No emails to mark as read.")
            return
        
        if self.args.dry_run:
            print(f"\n{Colors.YELLOW}🔍 DRY RUN:{Colors.RESET} Would mark {Colors.BOLD}{len(self.all_unread_ids)}{Colors.RESET} emails as read")
            print(f"   {Colors.YELLOW}(No emails were actually modified){Colors.RESET}")
            return
        
        total = len(self.all_unread_ids)
        print(f"\n{Colors.CYAN}📬 Marking ALL {Colors.BOLD}{total}{Colors.RESET}{Colors.CYAN} unread emails as READ...{Colors.RESET}")
        print(f"   {Colors.CYAN}Using batch API for speed...{Colors.RESET}\n")
        
        try:
            batch_size = 1000
            total_marked = 0
            num_batches = (len(self.all_unread_ids) + batch_size - 1) // batch_size
            
            mark_start = time.time()
            
            for i in range(0, len(self.all_unread_ids), batch_size):
                batch = self.all_unread_ids[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    self.service.users().messages().batchModify(
                        userId='me',
                        body={
                            'ids': batch,
                            'removeLabelIds': ['UNREAD']
                        }
                    ).execute()
                    
                    total_marked += len(batch)
                    
                    print_progress_bar(total_marked, total,
                                     prefix='Marking as read:',
                                     suffix=f'({total_marked}/{total})',
                                     start_time=mark_start)
                    
                    if batch_num < num_batches:
                        time.sleep(0.3)
                
                except Exception as e:
                    print(f"\n   {Colors.RED}✗ Error in batch {batch_num}: {str(e)[:50]}{Colors.RESET}")
                    continue
            
            print(f"\n{Colors.GREEN}✅ Successfully marked {Colors.BOLD}{total_marked}{Colors.RESET}{Colors.GREEN} emails as READ!{Colors.RESET}")
            
            print(f"\n{Colors.CYAN}🔄 Verifying changes...{Colors.RESET}")
            time.sleep(2)
            verify_results = self.service.users().messages().list(
                userId='me',
                q='is:unread',
                maxResults=1
            ).execute()
            
            remaining = verify_results.get('resultSizeEstimate', 0)
            
            if remaining == 0:
                print_success(f"VERIFIED: Your inbox shows {Colors.BOLD}0 unread messages!{Colors.RESET} 🎉")
            else:
                print_warning(f"{remaining} emails still showing as unread. Try refreshing Gmail.")
            
        except Exception as e:
            print_error(f"Error marking emails as read: {e}")
    
    def run(self):
        self.start_time = time.time()
        
        mode = f"{Colors.YELLOW}DRY RUN MODE{Colors.RESET}" if self.args.dry_run else f"{Colors.GREEN}LIVE MODE{Colors.RESET}"
        print_header(f"🚀 Starting Gmail Priority Email Reader & Cleaner [{mode}]")
        print("=" * 80)
        
        self.authenticate()
        
        self.analyze_emails()
        
        if not self.all_unread_ids:
            print(f"\n{Colors.GREEN}✨ Your inbox is empty! Nothing to do.{Colors.RESET}")
            return
        
        if self.priority_emails and not self.args.dry_run:
            self.label_priority_emails()
        
        report = self.create_report()
        
        if YOUR_EMAIL and YOUR_EMAIL != 'your.email@gmail.com':
            self.send_summary_email(report)
        
        self.mark_all_as_read()
        
        total_time = int(time.time() - self.start_time)
        
        print("\n")
        if self.args.dry_run:
            stats = {
                "Mode": f"{Colors.YELLOW}DRY RUN{Colors.RESET}",
                "Total Unread": f"{Colors.BOLD}{len(self.all_unread_ids)}{Colors.RESET}",
                "Priority Found": f"{Colors.BOLD}{len(self.priority_emails)}{Colors.RESET}",
                "Time Taken": f"{Colors.BOLD}{total_time}s{Colors.RESET}",
                "Status": f"{Colors.YELLOW}No Changes Made{Colors.RESET}"
            }
            print_summary_box(stats)
            print_info("DRY RUN COMPLETE! Run without --dry-run to actually clean inbox.")
        else:
            stats = {
                "Total Processed": f"{Colors.BOLD}{len(self.all_unread_ids)}{Colors.RESET} emails",
                "Priority Found": f"{Colors.BOLD}{len(self.priority_emails)}{Colors.RESET} emails",
                "Time Taken": f"{Colors.BOLD}{total_time}{Colors.RESET} seconds",
                "Inbox Status": f"{Colors.BOLD}0 Unread{Colors.RESET} ✅",
                "Report Saved": f"{Colors.GREEN}✓{Colors.RESET}"
            }
            print_summary_box(stats)
            print(f"{Colors.CYAN}💡 Refresh your Gmail to see the changes!{Colors.RESET}")
        
        print("=" * 80)


def main():
    if not sys.stdout.isatty() or (os.name == 'nt' and 'ANSICON' not in os.environ):
        try:
            import colorama
            colorama.init()
        except:
            Colors.disable()
    
    parser = argparse.ArgumentParser(
        description='Clean your Gmail inbox by marking emails as read and highlighting important ones.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gmail_reader.py                    # Normal run
  python gmail_reader.py --dry-run          # Preview without changes
  python gmail_reader.py --limit 50         # Only process 50 emails
  python gmail_reader.py --no-label         # Don't label priority emails
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview mode - analyze emails but do not mark as read')
    parser.add_argument('--limit', type=int, metavar='N',
                       help='Only analyze first N emails (default: 200)')
    parser.add_argument('--no-label', action='store_true',
                       help='Skip labeling priority emails')
    parser.add_argument('--version', action='version', version='Gmail Cleaner v2.0')
    
    args = parser.parse_args()
    
    reader = GmailPriorityReader(args)
    reader.run()


if __name__ == '__main__':
    main()