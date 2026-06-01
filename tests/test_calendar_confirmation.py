import unittest

from langchain_core.messages import AIMessage

import calendar_agent


class CalendarConfirmationTests(unittest.TestCase):
    def tool_call_response(self, action: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "manage_event",
                    "args": {"action": action, "event_id": "event-123"},
                    "id": f"call-{action}",
                }
            ],
        )

    def test_calendar_prompt_requires_confirmation_before_mutations(self):
        prompt = calendar_agent.build_system_prompt(
            user_email="user@example.com",
            current_date="2026-06-01, Monday",
        ).lower()

        self.assertIn("explicit confirmation", prompt)
        self.assertIn("update", prompt)
        self.assertIn("delete", prompt)
        self.assertIn("rsvp", prompt)
        self.assertIn("before calling `manage_event`", prompt)

    def test_destructive_manage_event_routes_to_confirmation(self):
        message = self.tool_call_response("delete")

        self.assertEqual(calendar_agent.calendar_tool_route(message), "confirmation")

    def test_update_and_rsvp_are_destructive_manage_event_actions(self):
        for action in ("update", "rsvp"):
            with self.subTest(action=action):
                message = self.tool_call_response(action)
                pending_action = calendar_agent.destructive_manage_event_tool_call(message)

                self.assertEqual(pending_action["name"], "manage_event")
                self.assertEqual(pending_action["args"]["action"], action)
                self.assertEqual(pending_action["id"], f"call-{action}")

    def test_create_manage_event_routes_to_tools_without_confirmation(self):
        message = self.tool_call_response("create")

        self.assertEqual(calendar_agent.calendar_tool_route(message), "tools")

    def test_confirmation_prompt_stores_complete_tool_call(self):
        message = self.tool_call_response("delete")
        pending_action = calendar_agent.destructive_manage_event_tool_call(message)
        prompt = calendar_agent.format_pending_action_message(pending_action)

        self.assertIn('"name": "manage_event"', prompt)
        self.assertIn('"action": "delete"', prompt)
        self.assertIn('"event_id": "event-123"', prompt)
        self.assertIn('"id": "call-delete"', prompt)
        self.assertIn("1. Confirm", prompt)
        self.assertIn("2. Cancel", prompt)

    def test_next_turn_accepts_only_structured_confirmation_choice(self):
        self.assertEqual(calendar_agent.parse_pending_action_choice("1"), "confirm")
        self.assertEqual(calendar_agent.parse_pending_action_choice("1. Confirm"), "confirm")
        self.assertEqual(calendar_agent.parse_pending_action_choice("2"), "cancel")
        self.assertEqual(calendar_agent.parse_pending_action_choice("2. Cancel"), "cancel")
        self.assertIsNone(calendar_agent.parse_pending_action_choice("yes, confirm"))
        self.assertIsNone(calendar_agent.parse_pending_action_choice("confirm"))


if __name__ == "__main__":
    unittest.main()
