import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==========================================
# Configuration
# ==========================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "your_test_account@gmail.com"
APP_PASSWORD = "your_gmail_app_password"

TARGET_EMAIL = "your_agent_mailbox@gmail.com"

SEND_DELAY_SECONDS = 0.5  # Avoid Gmail throttling

# ==========================================
# Test Emails
# ==========================================

emails = [
    {
        "subject": "Refund Request for Order #1001",
        "body": """
Hello,

I would like to request a refund for Order #1001.
The product does not meet my expectations.

Thank you.
""",
        "category": "REFUND_REQUEST"
    },
    {
        "subject": "Refund Request for Order #1002",
        "body": """
Hello,

The item arrived damaged.
Please process a refund.

Regards.
""",
        "category": "REFUND_REQUEST"
    },
    {
        "subject": "Return Request for Wireless Mouse",
        "body": """
Hello,

I would like to return my wireless mouse.
Please send return instructions.

Thanks.
""",
        "category": "RETURN_REQUEST"
    },
    {
        "subject": "Return Request for Keyboard",
        "body": """
Hello,

The keyboard is incompatible with my system.
I would like to return it.

Thank you.
""",
        "category": "RETURN_REQUEST"
    },
    {
        "subject": "Very Disappointed",
        "body": """
Hello,

Your customer service has been extremely disappointing.
I expect a response immediately.

Regards.
""",
        "category": "COMPLAINT"
    },
    {
        "subject": "Poor Service Experience",
        "body": """
Hello,

I have contacted support multiple times and nobody helped me.

Regards.
""",
        "category": "COMPLAINT"
    },
    {
        "subject": "Special Summer Promotion",
        "body": """
Hello,

Check out our newest products and discounts.

Marketing Team
""",
        "category": "OTHER"
    },
    {
        "subject": "Question About Refund Policy",
        "body": """
Hello,

Before purchasing, I would like to know your refund policy.

Thank you.
""",
        "category": "OTHER"
    }
]

# ==========================================
# Send Emails
# ==========================================

server = None
sent_count = 0
failed_count = 0

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, APP_PASSWORD)
    print(f"Connected to {SMTP_SERVER}:{SMTP_PORT} successfully.\n")

    for idx, email_data in enumerate(emails):
        try:
            msg = MIMEMultipart()
            msg["From"] = SENDER_EMAIL
            msg["To"] = TARGET_EMAIL
            msg["Subject"] = email_data["subject"]
            msg.attach(MIMEText(email_data["body"], "plain"))

            server.sendmail(SENDER_EMAIL, TARGET_EMAIL, msg.as_string())

            print(
                f"[{idx + 1}/{len(emails)}] ✓ Sent {email_data['category']} : "
                f"{email_data['subject']}"
            )
            sent_count += 1
            time.sleep(SEND_DELAY_SECONDS)

        except smtplib.SMTPException as e:
            print(
                f"[{idx + 1}/{len(emails)}] ✗ FAILED '{email_data['subject']}': {e}"
            )
            failed_count += 1

except smtplib.SMTPAuthenticationError:
    print("Authentication failed. Check your APP_PASSWORD and SENDER_EMAIL.")
except smtplib.SMTPConnectError:
    print(f"Could not connect to {SMTP_SERVER}:{SMTP_PORT}. Check network or server settings.")
except smtplib.SMTPException as e:
    print(f"SMTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if server:
        try:
            server.quit()
        except Exception:
            pass

print(f"\nDone. {sent_count} sent, {failed_count} failed.")
