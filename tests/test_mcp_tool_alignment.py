import re
import unittest
from dataclasses import dataclass

import calendar_agent
import refund_agent


ACTUAL_CALENDAR_MCP_TOOLS = {
    "start_google_auth",
    "list_calendars",
    "get_events",
    "manage_event",
    "manage_out_of_office",
    "manage_focus_time",
    "query_freebusy",
    "create_calendar",
}

ACTUAL_REFUND_MCP_TOOLS = {
    "start_google_auth",
    "search_gmail_messages",
    "get_gmail_message_content",
    "get_gmail_messages_content_batch",
    "get_gmail_attachment_content",
    "send_gmail_message",
    "draft_gmail_message",
    "get_gmail_thread_content",
    "get_gmail_threads_content_batch",
    "list_gmail_labels",
    "manage_gmail_label",
    "list_gmail_filters",
    "manage_gmail_filter",
    "modify_gmail_message_labels",
    "batch_modify_gmail_message_labels",
}


@dataclass
class FakeTool:
    name: str


def backticked_tool_names(prompt: str) -> set[str]:
    return {
        name
        for name in re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", prompt)
        if "_" in name
    }


class McpToolAlignmentTests(unittest.TestCase):
    def test_calendar_prompt_only_mentions_existing_tools(self):
        prompt = calendar_agent.build_system_prompt(
            user_email="user@example.com",
            current_date="2026-06-01, Monday",
        )

        self.assertLessEqual(backticked_tool_names(prompt), ACTUAL_CALENDAR_MCP_TOOLS)
        self.assertIn("get_events", prompt)
        self.assertIn("manage_event", prompt)
        self.assertIn("query_freebusy", prompt)
        self.assertNotIn("create_event", prompt)
        self.assertNotIn("create_focus_time_event", prompt)

    def test_refund_prompt_only_mentions_existing_tools(self):
        prompt = refund_agent.build_system_prompt("support@example.com")

        self.assertLessEqual(backticked_tool_names(prompt), ACTUAL_REFUND_MCP_TOOLS)
        self.assertIn("get_gmail_thread_content", prompt)
        self.assertIn("draft_gmail_message", prompt)
        self.assertIn("modify_gmail_message_labels", prompt)
        self.assertNotIn("create_gmail_draft", prompt)

    def test_calendar_mcp_whitelist_contains_only_spec_tools(self):
        tools = [FakeTool(name) for name in ACTUAL_CALENDAR_MCP_TOOLS]

        filtered = calendar_agent.filter_calendar_mcp_tools(tools)

        self.assertEqual(
            [tool.name for tool in filtered],
            ["list_calendars", "get_events", "manage_event", "query_freebusy"],
        )

    def test_refund_mcp_whitelist_contains_only_spec_tools(self):
        tools = [FakeTool(name) for name in ACTUAL_REFUND_MCP_TOOLS]

        filtered = refund_agent.filter_refund_mcp_tools(tools)

        self.assertEqual(
            [tool.name for tool in filtered],
            [
                "search_gmail_messages",
                "get_gmail_message_content",
                "get_gmail_messages_content_batch",
                "send_gmail_message",
                "draft_gmail_message",
                "get_gmail_thread_content",
                "list_gmail_labels",
                "modify_gmail_message_labels",
            ],
        )


if __name__ == "__main__":
    unittest.main()
