#!/usr/bin/env python3
"""Send email via SMTP (Office365). Usage:
    python3 perf_opt/send_mail.py "Subject" body_file.txt [attachment1 ...]
Credentials from env vars: MAIL_USER, MAIL_PASS
"""
import sys, smtplib, os
from email.message import EmailMessage

def send(subject, body, attachments=None):
    user = os.environ.get("MAIL_USER")
    pwd = os.environ.get("MAIL_PASS")
    if not user or not pwd:
        print("NO_MAIL_CREDENTIALS: set MAIL_USER and MAIL_PASS env vars")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = "yyc2660109300@outlook.com"
    msg.set_content(body)
    for f in (attachments or []):
        with open(f, "rb") as fh:
            filename = os.path.basename(f)
            msg.add_attachment(fh.read(), maintype="application",
                               subtype="octet-stream", filename=filename)
    s = smtplib.SMTP("smtp.office365.com", 587)
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
    s.quit()
    print("MAIL_SENT")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} SUBJECT BODY_FILE [ATTACHMENT...]")
        sys.exit(1)
    subj = sys.argv[1]
    with open(sys.argv[2]) as f:
        body = f.read()
    att = sys.argv[3:] if len(sys.argv) > 3 else None
    send(subj, body, att)