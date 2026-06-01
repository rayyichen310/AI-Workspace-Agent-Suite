import os
import sys
import json
import asyncio
import subprocess
from typing import TypedDict, Annotated, Sequence, Any
from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_config import setup_environment, get_mcp_config
setup_environment("calendar_mcp.log")

TAIPEI_TZ = timezone(timedelta(hours=8))
DEFAULT_WORKSPACE_CLI_PORT = "8765"
CALENDAR_MCP_TOOL_NAMES = (
    "list_calendars",
    "get_events",
    "manage_event",
    "query_freebusy",
)
DESTRUCTIVE_CALENDAR_ACTIONS = {"update", "delete", "rsvp"}
CONFIRMATION_MARKER = "[calendar-confirmation-required]"
CONFIRMATION_WORDS = {
    "yes",
    "confirm",
    "confirmed",
    "ok",
    "okay",
    "proceed",
    "確定",
    "確認",
    "可以",
    "同意",
}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def _print_setup_guide() -> None:
    print("Missing required environment variables. Please follow the setup guide:")
    print("1. Install workspace-mcp: `uv tool install google_workspace_mcp`")
    print("2. Set up Google Cloud Project and enable Calendar API.")
    print("3. Configure OAuth consent screen and create Desktop App credentials.")
    print("4. Export the following variables:")
    print("   export GOOGLE_API_KEY=<your-google-ai-studio-key>")
    print("   export GOOGLE_OAUTH_CLIENT_ID=<your-client-id>")
    print("   export GOOGLE_OAUTH_CLIENT_SECRET=<your-client-secret>")
    print("   export USER_GOOGLE_EMAIL=<your-email>")

def filter_calendar_mcp_tools(tools):
    tools_by_name = {tool.name: tool for tool in tools}
    return [tools_by_name[name] for name in CALENDAR_MCP_TOOL_NAMES if name in tools_by_name]

def build_system_prompt(user_email: str, current_date: str) -> str:
    return f"""You are a highly capable Google Calendar Agent managing the calendar for {user_email}.
    Today's date is {current_date}. 
    The user's timezone is Asia/Taipei (UTC+8).
    Your goal is to help the user manage their schedule.
    
    TOOL SELECTION RULES:
    1. For simple, read-only queries like "What's on today?" or "List my calendars", prefer using the CLI tools (cli_today_events, cli_list_calendars) for speed.
    2. For direct MCP calendar reads, use `list_calendars` for calendars and `get_events` for event lookup or time-range search.
    3. For creating, updating, deleting, or RSVP responses, use `manage_event` with action "create", "update", "delete", or "rsvp".
    4. For finding available meeting slots across attendees, use `query_freebusy`.
    5. For update, delete, or RSVP actions, ask for explicit confirmation before calling `manage_event`.
    6. Always format times into human-readable strings (e.g., "3:00 PM" instead of raw ISO format) when replying to the user.
    7. CRITICAL TIMEZONE RULE: When creating or updating events, ALWAYS include the timezone offset in the ISO timestamp (e.g., "2026-05-29T13:00:00+08:00"). Do not use 'Z' (UTC) unless explicitly calculating a timezone difference.
    8. You already know the user's email is {user_email}. Do not ask for it.
    """

def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)

def _latest_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""

def _has_pending_calendar_confirmation(messages: Sequence[BaseMessage]) -> bool:
    return any(
        isinstance(message, AIMessage) and CONFIRMATION_MARKER in _message_text(message)
        for message in messages
    )

def _is_confirmation_text(text: str) -> bool:
    normalized = text.strip().lower()
    return any(word in normalized for word in CONFIRMATION_WORDS)

def _destructive_manage_event_action(response: BaseMessage) -> str | None:
    for tool_call in getattr(response, "tool_calls", []) or []:
        if tool_call.get("name") != "manage_event":
            continue
        action = str(tool_call.get("args", {}).get("action", "")).strip().lower()
        if action in DESTRUCTIVE_CALENDAR_ACTIONS:
            return action
    return None

def enforce_calendar_mutation_confirmation(state: AgentState, response: BaseMessage) -> BaseMessage:
    action = _destructive_manage_event_action(response)
    if action is None:
        return response

    messages = list(state["messages"])
    if _has_pending_calendar_confirmation(messages) and _is_confirmation_text(_latest_human_text(messages)):
        return response

    return AIMessage(
        content=(
            f"{CONFIRMATION_MARKER}\n"
            f"Please confirm before I {action} this calendar event. "
            "Reply with an explicit confirmation to proceed."
        )
    )

def _run_cli(args: list[str], timeout: int = 15) -> dict[str, Any]:
    # Run workspace-cli subprocess with a timeout handler
    try:
        result = subprocess.run(
            ["workspace-cli", "--url", _workspace_cli_url()] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            return {"error": result.stderr}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"result": result.stdout}
    except Exception as e:
        return {"error": str(e)}

def _workspace_cli_url() -> str:
    if os.environ.get("WORKSPACE_MCP_URL"):
        return os.environ["WORKSPACE_MCP_URL"]
    port = os.environ.get("WORKSPACE_MCP_HTTP_PORT", DEFAULT_WORKSPACE_CLI_PORT)
    return f"http://127.0.0.1:{port}/mcp"

def _cli_arg(name: str, value: Any) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"{name}={value}"

@tool
def cli_list_calendars():
    """What calendars do I have? Call this to list all calendars and get their IDs."""
    return _run_cli(["call", "list_calendars"])

@tool
def cli_today_events(calendar_id: str = "primary"):
    """What's on today? Fast CLI tool to list today's events."""
    today_start = datetime.now(TAIPEI_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    return _run_cli([
        "call",
        "get_events",
        _cli_arg("calendar_id", calendar_id),
        _cli_arg("time_min", today_start.isoformat()),
        _cli_arg("time_max", today_end.isoformat()),
        _cli_arg("max_results", 25),
    ])

@tool
def cli_list_events(time_min: str = None, time_max: str = None, max_results: int = 10, calendar_id: str = "primary"):
    """Show me this week. Fast CLI tool for listing events."""
    now = datetime.now(TAIPEI_TZ).replace(microsecond=0)
    if not time_min:
        time_min = now.isoformat()
    if not time_max:
        time_max = (now + timedelta(days=7)).isoformat()
    
    args = [
        "call",
        "get_events",
        _cli_arg("calendar_id", calendar_id),
        _cli_arg("time_min", time_min),
        _cli_arg("time_max", time_max),
        _cli_arg("max_results", max_results),
    ]
    
    return _run_cli(args)

@tool
def cli_get_event(event_id: str, calendar_id: str = "primary"):
    """Get details for that meeting. Fast CLI tool to fetch a single event."""
    return _run_cli([
        "call",
        "get_events",
        _cli_arg("calendar_id", calendar_id),
        _cli_arg("event_id", event_id),
        _cli_arg("detailed", True),
    ])

@tool
def cli_tool_list():
    """Debug/tool discovery. Fast CLI tool to enumerate all workspace-cli tools."""
    return _run_cli(["list"])

async def build_agent(mcp_client: MultiServerMCPClient):
    mcp_tools = filter_calendar_mcp_tools(await mcp_client.get_tools())
    # Merge MCP tools with all CLI tools
    cli_tools = [cli_list_calendars, cli_today_events, cli_list_events, cli_get_event, cli_tool_list]
    all_tools = mcp_tools + cli_tools

    llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    current_date = datetime.now().strftime("%Y-%m-%d, %A")
    user_email = os.environ.get("USER_GOOGLE_EMAIL", "your primary email")

    SYSTEM_PROMPT = build_system_prompt(user_email=user_email, current_date=current_date)

    def agent_node(state: AgentState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
        response = llm_with_tools.invoke(messages)
        response = enforce_calendar_mutation_confirmation(state, response)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(all_tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

async def process_agent_stream(agent, initial_messages) -> str:
    # Process Agent stream, accurately parse and print detailed progress of each tool execution
    final_message_content = None
    
    async for event in agent.astream({"messages": initial_messages}):
        for node_name, state_update in event.items():
            if node_name == "agent":
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        name = tool_call["name"]
                        args = tool_call.get("args", {})
                        
                        # Print the AI's intended action
                        if "cli_" in name:
                            sys.stdout.write(f"[Action] Executing fast CLI tool '{name}'...\n")
                        elif "create" in name:
                            sys.stdout.write(f"[Action] Creating calendar event...\n")
                        elif "list" in name:
                            sys.stdout.write(f"[Action] Searching calendar events...\n")
                        else:
                            sys.stdout.write(f"[Action] Running tool '{name}'...\n")
                        sys.stdout.flush()
                else:
                    final_message_content = last_msg.content
                    
            elif node_name == "tools":
                sys.stdout.write("[System] Action executed successfully.\n")
                sys.stdout.flush()
                
    display_text = ""
    if isinstance(final_message_content, list):
        for block in final_message_content:
            if isinstance(block, dict) and block.get("type") == "text":
                display_text += block.get("text", "")
    else:
        display_text = str(final_message_content)
        
    return display_text

async def run_demo(agent) -> None:
    # Iterate over three pre-written demo queries without requiring user input
    demo_queries = [
        "What calendars do I have?",
        "What's on my calendar today?",
        "Show me my events for the next 7 days."
    ]
    
    print("\n" + "="*50)
    print("[INFO] Starting Calendar Agent Demo Mode")
    print("="*50)
    
    for idx, query in enumerate(demo_queries, 1):
        print(f"\n[Demo Query {idx}] {query}")
        print("-" * 50)
        messages = [HumanMessage(content=query)]
        display_text = await process_agent_stream(agent, messages)
        print(f"\nAgent:\n{display_text.strip()}")
        print("=" * 50)

async def run_interactive_chat(agent):
    print("\n" + "="*50)
    print("[INFO] Calendar Agent Ready!")
    print("Commands: 'quit' or 'exit' to leave")
    print("="*50)
    
    # Maintain full conversation history
    chat_history = []
    
    while True:
        sys.stdout.write("\nYou: ")
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()
        
        if not user_input or user_input.lower() in ['quit', 'exit']:
            break
            
        chat_history.append(HumanMessage(content=user_input))
        display_text = await process_agent_stream(agent, chat_history)
            
        print(f"\nAgent:\n{display_text.strip()}")
        chat_history.append(BaseMessage(content=display_text, type="ai"))

async def main():
    required_envs = ["GOOGLE_API_KEY", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "USER_GOOGLE_EMAIL"]
    if not all(os.environ.get(env) for env in required_envs):
        _print_setup_guide()
        sys.exit(1)

    print("\n[INFO] Loading MCP Server configuration...")
    mcp_config = get_mcp_config("calendar")
    
    print("[INFO] Starting Workspace MCP Server (Transport: stdio)...")
    mcp_client = MultiServerMCPClient(mcp_config)
    
    try:
        print("[INFO] Connecting to Calendar API and building AI Agent...")
        agent = await build_agent(mcp_client)
        print("[INFO] Initialization complete!")
        print("-" * 50)
        
        # Select mode before starting the session
        sys.stdout.write("Select mode - (1) Interactive Mode (2) Demo Mode: ")
        sys.stdout.flush()
        mode = sys.stdin.readline().strip()
        
        if mode == "1":
            await run_interactive_chat(agent)
        else:
            await run_demo(agent)
    finally:
        if hasattr(mcp_client, 'close'):
            print("\n[INFO] Shutting down MCP Server...")
            await mcp_client.close()
            print("[INFO] Disconnected. Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
