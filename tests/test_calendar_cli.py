import unittest
from datetime import datetime as RealDateTime
from datetime import timedelta, timezone
from unittest.mock import patch

import calendar_agent
from mcp_config import get_mcp_config


WORKSPACE_CLI_URL = "http://127.0.0.1:8765/mcp"


class FixedDateTime(RealDateTime):
    @classmethod
    def now(cls, tz=None):
        fixed = RealDateTime(2026, 6, 1, 15, 30, 0)
        if tz is not None:
            return fixed.replace(tzinfo=timezone(timedelta(hours=8))).astimezone(tz)
        return fixed


class CalendarCliToolTests(unittest.TestCase):
    def capture_commands(self, callback):
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)

            class Result:
                returncode = 0
                stdout = "{}"
                stderr = ""

            return Result()

        with patch.object(calendar_agent.subprocess, "run", fake_run):
            callback()

        return commands

    def test_cli_list_calendars_uses_workspace_cli_key_value_call(self):
        commands = self.capture_commands(lambda: calendar_agent.cli_list_calendars.invoke({}))

        self.assertEqual(
            commands,
            [["workspace-cli", "--url", WORKSPACE_CLI_URL, "call", "list_calendars"]],
        )

    def test_cli_today_events_queries_only_today_in_taipei(self):
        with patch.object(calendar_agent, "datetime", FixedDateTime):
            commands = self.capture_commands(
                lambda: calendar_agent.cli_today_events.invoke({"calendar_id": "primary"})
            )

        self.assertEqual(
            commands,
            [
                [
                    "workspace-cli",
                    "--url",
                    WORKSPACE_CLI_URL,
                    "call",
                    "get_events",
                    "calendar_id=primary",
                    "time_min=2026-06-01T00:00:00+08:00",
                    "time_max=2026-06-02T00:00:00+08:00",
                    "max_results=25",
                ]
            ],
        )

    def test_cli_list_events_defaults_to_next_7_days_in_taipei(self):
        with patch.object(calendar_agent, "datetime", FixedDateTime):
            commands = self.capture_commands(
                lambda: calendar_agent.cli_list_events.invoke({"calendar_id": "primary"})
            )

        self.assertEqual(
            commands,
            [
                [
                    "workspace-cli",
                    "--url",
                    WORKSPACE_CLI_URL,
                    "call",
                    "get_events",
                    "calendar_id=primary",
                    "time_min=2026-06-01T15:30:00+08:00",
                    "time_max=2026-06-08T15:30:00+08:00",
                    "max_results=10",
                ]
            ],
        )

    def test_cli_get_event_uses_get_events_with_event_id(self):
        commands = self.capture_commands(
            lambda: calendar_agent.cli_get_event.invoke(
                {"event_id": "evt-123", "calendar_id": "primary"}
            )
        )

        self.assertEqual(
            commands,
            [
                [
                    "workspace-cli",
                    "--url",
                    WORKSPACE_CLI_URL,
                    "call",
                    "get_events",
                    "calendar_id=primary",
                    "event_id=evt-123",
                    "detailed=true",
                ]
            ],
        )

    def test_calendar_mcp_config_enables_workspace_cli_sidecar(self):
        config = get_mcp_config("calendar")

        self.assertEqual(
            config["workspace"]["env"]["WORKSPACE_MCP_HTTP_PORT"],
            "8765",
        )


if __name__ == "__main__":
    unittest.main()
