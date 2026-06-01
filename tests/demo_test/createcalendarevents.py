import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ==========================================
# Configuration
# ==========================================

TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CALENDAR_ID = "primary"
TIMEZONE = "Asia/Taipei"

# ==========================================
# Load & Refresh Credentials
# ==========================================

creds = Credentials.from_authorized_user_file(TOKEN_FILE, scopes=SCOPES)

if creds.expired and creds.refresh_token:
    creds.refresh(Request())
    # Persist the refreshed token for next run
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    print("Token refreshed and saved.\n")

# ==========================================
# Build Service
# ==========================================

service = build("calendar", "v3", credentials=creds)

# ==========================================
# Events to Create
# ==========================================

events = [
    {
        "summary": "Team Standup",
        "start": "2026-06-01T09:00:00",
        "end":   "2026-06-01T10:00:00"
    },
    {
        "summary": "Research Meeting",
        "start": "2026-06-01T14:00:00",
        "end":   "2026-06-01T15:00:00"
    },
    {
        "summary": "Project Review",
        "start": "2026-06-02T10:00:00",
        "end":   "2026-06-02T11:00:00"
    },
    {
        "summary": "Student Advising",
        "start": "2026-06-02T15:00:00",
        "end":   "2026-06-02T16:00:00"
    },
    {
        "summary": "Faculty Meeting",
        "start": "2026-06-03T09:00:00",
        "end":   "2026-06-03T10:30:00"
    },
    {
        "summary": "PhD Progress Review",
        "start": "2026-06-03T14:00:00",
        "end":   "2026-06-03T15:00:00"
    },
    {
        "summary": "Industry Collaboration Meeting",
        "start": "2026-06-04T11:00:00",
        "end":   "2026-06-04T12:00:00"
    },
    {
        "summary": "Lab Weekly Meeting",
        "start": "2026-06-04T15:00:00",
        "end":   "2026-06-04T16:00:00"
    },
    {
        "summary": "Grant Proposal Discussion",
        "start": "2026-06-05T09:00:00",
        "end":   "2026-06-05T10:00:00"
    },
    {
        "summary": "Research Seminar",
        "start": "2026-06-05T15:00:00",
        "end":   "2026-06-05T16:00:00"
    }
]

# ==========================================
# Create Events
# ==========================================

created_count = 0
failed_count = 0

for idx, e in enumerate(events):
    body = {
        "summary": e["summary"],
        "start": {
            "dateTime": e["start"],
            "timeZone": TIMEZONE
        },
        "end": {
            "dateTime": e["end"],
            "timeZone": TIMEZONE
        }
    }

    try:
        result = service.events().insert(
            calendarId=CALENDAR_ID,
            body=body
        ).execute()

        print(
            f"[{idx + 1}/{len(events)}] ✓ Created: {e['summary']} "
            f"(id: {result.get('id')})"
        )
        created_count += 1

    except HttpError as e:
        print(
            f"[{idx + 1}/{len(events)}] ✗ FAILED: {body['summary']} "
            f"— HTTP {e.resp.status}: {e.reason}"
        )
        failed_count += 1

    except Exception as ex:
        print(
            f"[{idx + 1}/{len(events)}] ✗ UNEXPECTED ERROR: {body['summary']} — {ex}"
        )
        failed_count += 1

print(f"\nDone. {created_count} created, {failed_count} failed.")
