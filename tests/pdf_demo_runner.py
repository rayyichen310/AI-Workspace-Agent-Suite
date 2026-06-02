import argparse
import asyncio
import os
import re
import smtplib
import sys
import time
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT_DIR / "project2" / ".env", override=True)
load_dotenv(ROOT_DIR / ".env", override=True)

import calendar_agent  # noqa: E402
import refund_agent  # noqa: E402
from mcp_config import get_mcp_config  # noqa: E402


CALENDAR_BASELINE_EVENTS = [
    ("Team Standup", "2026-06-01T09:00:00+08:00", "2026-06-01T10:00:00+08:00"),
    ("Research Meeting", "2026-06-01T14:00:00+08:00", "2026-06-01T15:00:00+08:00"),
    ("Project Review", "2026-06-02T10:00:00+08:00", "2026-06-02T11:00:00+08:00"),
    ("Student Advising", "2026-06-02T15:00:00+08:00", "2026-06-02T16:00:00+08:00"),
    ("Faculty Meeting", "2026-06-03T09:00:00+08:00", "2026-06-03T10:30:00+08:00"),
    ("PhD Progress Review", "2026-06-03T14:00:00+08:00", "2026-06-03T15:00:00+08:00"),
    ("Industry Collaboration Meeting", "2026-06-04T11:00:00+08:00", "2026-06-04T12:00:00+08:00"),
    ("Lab Weekly Meeting", "2026-06-04T15:00:00+08:00", "2026-06-04T16:00:00+08:00"),
    ("Grant Proposal Discussion", "2026-06-05T09:00:00+08:00", "2026-06-05T10:00:00+08:00"),
    ("Research Seminar", "2026-06-05T15:00:00+08:00", "2026-06-05T16:00:00+08:00"),
]

CALENDAR_PROMPTS = [
    "What's on my calendar?",
    "Schedule a team lunch for the coming Friday at noon for 1 hour.",
    "Find a free 30-minute slot for a call with john@example.com this week.",
]

REFUND_EMAILS = [
    (
        "Refund Request for Order #1001",
        "Hello,\n\nI would like to request a refund for Order #1001.\nThe product does not meet my expectations.\n\nThank you.\n",
    ),
    (
        "Refund Request for Order #1002",
        "Hello,\n\nThe item arrived damaged.\nPlease process a refund.\n\nRegards.\n",
    ),
    (
        "Return Request for Wireless Mouse",
        "Hello,\n\nI would like to return my wireless mouse.\nPlease send return instructions.\n\nThanks.\n",
    ),
    (
        "Return Request for Keyboard",
        "Hello,\n\nThe keyboard is incompatible with my system.\nI would like to return it.\n\nThank you.\n",
    ),
    (
        "Very Disappointed",
        "Hello,\n\nYour customer service has been extremely disappointing.\nI expect a response immediately.\n\nRegards.\n",
    ),
    (
        "Poor Service Experience",
        "Hello,\n\nI have contacted support multiple times and nobody helped me.\n\nRegards.\n",
    ),
    (
        "Special Summer Promotion",
        "Hello,\n\nCheck out our newest products and discounts.\n\nMarketing Team\n",
    ),
    (
        "Question About Refund Policy",
        "Hello,\n\nBefore purchasing, I would like to know your refund policy.\n\nThank you.\n",
    ),
]

REFUND_PROMPT = (
    "Process all unread customer service emails, including refunds, returns, complaints, "
    "and other messages. Follow the configured workflow and produce a summary."
)


MANUAL_PROMPTS = [
    (
        "Calendar 1",
        "Show all calendar events from June 1, 2026 through June 5, 2026.",
    ),
    (
        "Calendar 2",
        "Schedule a team lunch for Friday, June 5, 2026 at 12:00 PM for 1 hour.",
    ),
    (
        "Calendar 3",
        "Find a free 30-minute slot this week. If john@example.com's calendar is unavailable, list my available slots.",
    ),
    (
        "Refund",
        REFUND_PROMPT,
    ),
]


def require_env() -> None:
    missing = [
        key
        for key in (
            "GOOGLE_API_KEY",
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "USER_GOOGLE_EMAIL",
        )
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_manual_prompts() -> None:
    print_section("Manual prompts")
    for label, prompt in MANUAL_PROMPTS:
        print(f"{label}:")
        print(prompt)
        print()


async def get_tool(client: MultiServerMCPClient, name: str):
    tools = await client.get_tools()
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    raise RuntimeError(f"MCP tool not available: {name}")


async def initialize_calendar_data(client: MultiServerMCPClient, reset_team_lunch: bool) -> None:
    print_section("Initializing calendar demo data")
    get_events = await get_tool(client, "get_events")
    manage_event = await get_tool(client, "manage_event")

    event_list = await get_events.ainvoke(
        {
            "calendar_id": "primary",
            "time_min": "2026-06-01T00:00:00+08:00",
            "time_max": "2026-06-06T00:00:00+08:00",
            "max_results": 50,
        }
    )
    event_text = str(event_list)

    if reset_team_lunch:
        ids_to_delete = re.findall(
            r'"Team Lunch".*?Starts: 2026-06-05T12:00:00\+08:00.*?ID: ([^ |]+)',
            event_text,
        )
        for event_id in ids_to_delete:
            print(f"Deleting previous run event: Team Lunch ({event_id})")
            await manage_event.ainvoke(
                {
                    "action": "delete",
                    "event_id": event_id,
                    "calendar_id": "primary",
                    "send_updates": "none",
                }
            )

    for summary, start_time, end_time in CALENDAR_BASELINE_EVENTS:
        if f'"{summary}"' in event_text and f"Starts: {start_time}" in event_text:
            print(f"Already present: {summary}")
            continue

        print(f"Creating missing baseline event: {summary}")
        await manage_event.ainvoke(
            {
                "action": "create",
                "summary": summary,
                "start_time": start_time,
                "end_time": end_time,
                "calendar_id": "primary",
                "timezone": "Asia/Taipei",
                "send_updates": "none",
            }
        )


async def initialize_refund_data(
    client: MultiServerMCPClient,
    send_missing: bool,
    force_send: bool,
) -> None:
    print_section("Initializing refund demo data")
    search = await get_tool(client, "search_gmail_messages")
    missing_emails = []

    for subject, body in REFUND_EMAILS:
        result = await search.ainvoke({"query": f'subject:"{subject}"'})
        if "Found 0 messages" in str(result):
            print(f"Missing: {subject}")
            missing_emails.append((subject, body))
        else:
            print(f"Already present: {subject}")

    if force_send:
        print("Force-send enabled. Sending a fresh copy of all refund demo emails.")
        missing_emails = REFUND_EMAILS

    if not missing_emails:
        if send_missing:
            print("All refund email fixtures are already present. No emails were sent.")
        return

    if not send_missing:
        print("Refund email fixtures are missing. Re-run with --send-missing-refund-emails to send them.")
        return

    sender_email = os.environ.get("DEMO_SENDER_EMAIL") or os.environ.get("USER_GOOGLE_EMAIL")
    app_password = os.environ.get("DEMO_GMAIL_APP_PASSWORD")
    target_email = os.environ.get("DEMO_TARGET_EMAIL") or os.environ.get("USER_GOOGLE_EMAIL")
    if not sender_email or not app_password or not target_email:
        raise RuntimeError(
            "Set DEMO_GMAIL_APP_PASSWORD before using --send-missing-refund-emails. "
            "DEMO_SENDER_EMAIL and DEMO_TARGET_EMAIL default to USER_GOOGLE_EMAIL."
        )

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, app_password)
        for subject, body in missing_emails:
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = target_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            server.sendmail(sender_email, target_email, msg.as_string())
            print(f"Sent missing fixture email: {subject}")
            time.sleep(0.5)


async def run_calendar_pdf_prompts(client: MultiServerMCPClient) -> None:
    print_section("Running Calendar Agent PDF prompts")
    agent = await calendar_agent.build_agent(client)
    history = []

    for index, prompt in enumerate(CALENDAR_PROMPTS, 1):
        print_section(f"Calendar Prompt {index}: {prompt}")
        history.append(HumanMessage(content=prompt))
        display_text, pending_action = await calendar_agent.process_agent_stream(
            agent,
            history,
            return_pending=True,
        )
        print(display_text.strip())
        history.append(AIMessage(content=display_text))

        if pending_action:
            print("A pending destructive action was produced. This runner does not auto-confirm it.")


async def run_refund_pdf_prompt(client: MultiServerMCPClient) -> None:
    print_section("Running Refund Agent PDF prompt")
    agent = await refund_agent.build_agent(client)
    display_text = await refund_agent.process_agent_stream(
        agent,
        [HumanMessage(content=REFUND_PROMPT)],
    )
    print(display_text.strip())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PDF demo flow without assertions.")
    parser.add_argument("--skip-calendar-init", action="store_true")
    parser.add_argument("--skip-refund-init", action="store_true")
    parser.add_argument("--skip-calendar-agent", action="store_true")
    parser.add_argument("--skip-refund-agent", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--keep-team-lunch", action="store_true")
    parser.add_argument("--send-missing-refund-emails", action="store_true")
    parser.add_argument("--force-send-refund-emails", action="store_true")
    args = parser.parse_args()

    require_env()

    run_calendar_agent = not args.init_only and not args.skip_calendar_agent
    run_refund_agent = not args.init_only and not args.skip_refund_agent

    if not args.skip_calendar_init or run_calendar_agent:
        calendar_client = MultiServerMCPClient(get_mcp_config("calendar"))
        try:
            if not args.skip_calendar_init:
                await initialize_calendar_data(calendar_client, not args.keep_team_lunch)
            if run_calendar_agent:
                await run_calendar_pdf_prompts(calendar_client)
        finally:
            if hasattr(calendar_client, "close"):
                await calendar_client.close()

    if not args.skip_refund_init or run_refund_agent:
        refund_client = MultiServerMCPClient(get_mcp_config("refund"))
        try:
            if not args.skip_refund_init:
                await initialize_refund_data(
                    refund_client,
                    args.send_missing_refund_emails,
                    args.force_send_refund_emails,
                )
            if run_refund_agent:
                await run_refund_pdf_prompt(refund_client)
        finally:
            if hasattr(refund_client, "close"):
                await refund_client.close()

    if args.init_only:
        print_manual_prompts()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        message = str(exc)
        print("\nDemo runner failed.")
        if "ACTION REQUIRED: Google Authentication Needed" in message:
            print("\nGoogle authorization is required before this demo can continue.")
            print(message)
        else:
            print(traceback.format_exc())
        raise SystemExit(1)
