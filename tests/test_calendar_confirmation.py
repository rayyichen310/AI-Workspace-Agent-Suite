import unittest

from langchain_core.messages import AIMessage, HumanMessage

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

    def test_delete_requires_confirmation_before_tool_call(self):
        state = {"messages": [HumanMessage(content="Delete event event-123")]}

        guarded = calendar_agent.enforce_calendar_mutation_confirmation(
            state,
            self.tool_call_response("delete"),
        )

        self.assertEqual(guarded.tool_calls, [])
        self.assertIn("confirm", guarded.content.lower())
        self.assertIn("delete", guarded.content.lower())

    def test_update_requires_confirmation_before_tool_call(self):
        state = {"messages": [HumanMessage(content="Move event event-123 to 3 PM")]}

        guarded = calendar_agent.enforce_calendar_mutation_confirmation(
            state,
            self.tool_call_response("update"),
        )

        self.assertEqual(guarded.tool_calls, [])
        self.assertIn("confirm", guarded.content.lower())
        self.assertIn("update", guarded.content.lower())

    def test_rsvp_requires_confirmation_before_tool_call(self):
        state = {"messages": [HumanMessage(content="Decline event event-123")]}

        guarded = calendar_agent.enforce_calendar_mutation_confirmation(
            state,
            self.tool_call_response("rsvp"),
        )

        self.assertEqual(guarded.tool_calls, [])
        self.assertIn("confirm", guarded.content.lower())
        self.assertIn("rsvp", guarded.content.lower())

    def test_confirmed_mutation_tool_call_is_allowed(self):
        state = {
            "messages": [
                HumanMessage(content="Delete event event-123"),
                AIMessage(content="[calendar-confirmation-required]\nPlease confirm delete."),
                HumanMessage(content="yes, confirm"),
            ]
        }
        response = self.tool_call_response("delete")

        guarded = calendar_agent.enforce_calendar_mutation_confirmation(state, response)

        self.assertIs(guarded, response)
        self.assertEqual(guarded.tool_calls[0]["args"]["action"], "delete")

    def test_create_does_not_require_confirmation(self):
        state = {"messages": [HumanMessage(content="Create a lunch event tomorrow")]}
        response = self.tool_call_response("create")

        guarded = calendar_agent.enforce_calendar_mutation_confirmation(state, response)

        self.assertIs(guarded, response)
        self.assertEqual(guarded.tool_calls[0]["args"]["action"], "create")


if __name__ == "__main__":
    unittest.main()
