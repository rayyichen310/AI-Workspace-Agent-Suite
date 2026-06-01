import os
import sys
import json
import asyncio
import subprocess
from typing import TypedDict, Annotated, Sequence, Any, Union
from datetime import datetime, timedelta, timezone

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_config import setup_environment, get_mcp_config

# Initialize environment
setup_environment("calendar_mcp.log")

CALENDAR_MCP_TOOL_NAMES = (
    "list_calendars",
    "get_events",
    "manage_event",
    "query_freebusy",
)
DESTRUCTIVE_MANAGE_EVENT_ACTIONS = {"update", "delete", "rsvp"}


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pending_action: dict[str, Any] | None
    confirmed_action: dict[str, Any] | None

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
        Current Date: {current_date}
        Timezone: Asia/Taipei (UTC+8)

        Your primary goal is to help the user efficiently manage their schedule.

        TOOL SELECTION GUIDE (CLI vs MCP)
        1. READ-ONLY / FAST QUERIES: Prefer CLI tools for simple scheduling questions like "What is on my calendar today?" or "List my events".
        2. DIRECT MCP READS: Use `list_calendars` for calendars, `get_events` for event lookup or time-range search, and `query_freebusy` for availability.
        3. MUTATIONS: Use `manage_event` with action "create", "update", "delete", or "rsvp".

        SAFETY AND CONFIRMATION RULES
        - DESTRUCTIVE OPERATIONS: For update, delete, or rsvp actions, ask for explicit confirmation before calling `manage_event`.
        - EVENT CREATION: Use `manage_event` with action "create" for scheduling meetings.

        FORMATTING AND DATA RULES
        - HUMAN-READABLE TIME: Always convert raw ISO timestamps from tool outputs into natural, human-readable formats (e.g., "3:00 PM", "Tomorrow at 10:00 AM") when replying to the user.
        - TIMEZONE HANDLING: When passing time data to tools for creating or updating events, ALWAYS append the correct timezone offset (e.g., "2026-05-29T13:00:00+08:00"). Do not use 'Z' (UTC) unless explicitly calculating a timezone difference.
        - USER CONTEXT: You already know the user's email is {user_email}. Do not ask for it.
        """


def destructive_manage_event_tool_call(message: BaseMessage) -> dict[str, Any] | None:
    for tool_call in getattr(message, "tool_calls", []) or []:
        if tool_call.get("name") != "manage_event":
            continue
        action = str(tool_call.get("args", {}).get("action", "")).strip().lower()
        if action in DESTRUCTIVE_MANAGE_EVENT_ACTIONS:
            return dict(tool_call)
    return None


def format_pending_action_message(tool_call: dict[str, Any]) -> str:
    action = str(tool_call.get("args", {}).get("action", "modify")).strip() or "modify"
    serialized_tool_call = json.dumps(tool_call, ensure_ascii=False, indent=2)
    return (
        "This calendar action requires confirmation before it can run.\n\n"
        f"Pending action: manage_event action={action}\n"
        "Stored tool call:\n"
        f"{serialized_tool_call}\n\n"
        "Choose one option:\n"
        "1. Confirm\n"
        "2. Cancel"
    )


def parse_pending_action_choice(text: str) -> str | None:
    normalized = " ".join(text.strip().lower().replace(".", " ").split())
    if normalized in {"1", "1 confirm"}:
        return "confirm"
    if normalized in {"2", "2 cancel"}:
        return "cancel"
    return None


def calendar_tool_route(message: BaseMessage) -> str:
    if destructive_manage_event_tool_call(message):
        return "confirmation"
    if hasattr(message, "tool_calls") and message.tool_calls:
        return "tools"
    return END


def _run_cli(command: Union[str, list[str]], timeout: int = 15, **kwargs) -> dict[str, Any]:
    # Parse command and **kwargs into proper CLI arguments
    args = command if isinstance(command, list) else [command]
    
    for key, value in kwargs.items():
        if value is not None:
            # Convert snake_case to kebab-case for CLI flags
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                # If boolean is True, append flag (e.g., --detailed)
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

    # Run workspace-cli subprocess with a timeout handler
    try:
        result = subprocess.run(
            ["workspace-cli"] + args,
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

@tool  
def cli_today_events(calendar_id: str = "primary"):
    """
    Calculates today's UTC start and end timestamps and calls get_events to fetch today's schedule.
    No date arguments needed from the user.
    """
    now_utc = datetime.now(timezone.utc)
    
    # Get 00:00:00 UTC for today
    start_of_today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    # Get 23:59:59 (equivalent to 00:00:00 UTC tomorrow)
    end_of_today = start_of_today + timedelta(days=1)
    
    # Convert to RFC3339 format accepted by calendar_tools.py
    time_min = start_of_today.strftime('%Y-%m-%dT%H:%M:%SZ')
    time_max = end_of_today.strftime('%Y-%m-%dT%H:%M:%SZ')

    return _run_cli(
        "get_events", 
        calendar_id=calendar_id, 
        time_min=time_min, 
        time_max=time_max
    )

@tool
def cli_list_events(time_min: str = None, time_max: str = None, max_results: int = 25, calendar_id: str = "primary"):
    """
    General purpose event lister.
    If time_max is not provided, defaults to fetching events for 7 days from now (or from time_min).
    """
    now_utc = datetime.now(timezone.utc)
    
    if not time_min:
        time_min = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        
    if not time_max:
        # Default to 7 days later if no time_max is provided
        time_max_dt = now_utc + timedelta(days=7)
        time_max = time_max_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    return _run_cli(
        "get_events",
        calendar_id=calendar_id,
        time_min=time_min,
        time_max=time_max,
        max_results=max_results
    )

@tool
def cli_list_calendars():
    """
    Calls list_calendars tool with no arguments, returning all calendar metadata (ID, name, etc.).
    """
    return _run_cli("list_calendars")

@tool
def cli_get_event(event_id: str, calendar_id: str = "primary"):
    """
    Fetches full detailed information for a single event.
    """
    return _run_cli(
        "get_events",
        event_id=event_id,
        calendar_id=calendar_id,
        detailed=True
    )

@tool
def cli_tool_list():
    """
    Debug and tool discovery. Fast CLI tool to enumerate all workspace-cli tools.
    """
    return _run_cli(["list"])

async def build_agent(mcp_client: MultiServerMCPClient):
    # 1. Retrieve all MCP tools
    raw_mcp_tools = await mcp_client.get_tools()
    filtered_mcp_tools = filter_calendar_mcp_tools(raw_mcp_tools)
            
    # Extract the names from the filtered tool objects
    tool_names = [getattr(t, "name", "unknown") for t in filtered_mcp_tools]
    
    # Print the count and the joined list of names
    print(f"[INFO] Filtered and loaded {len(filtered_mcp_tools)} calendar-related MCP tools: {', '.join(tool_names)}")
    
    # 3. Merge MCP tools with all CLI tools
    cli_tools = [cli_list_calendars, cli_today_events, cli_list_events, cli_get_event, cli_tool_list]
    all_tools = filtered_mcp_tools + cli_tools

    # 4. Initialize LLM and bind combined tools
    llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    current_date = datetime.now().strftime("%Y-%m-%d, %A")
    user_email = os.environ.get("USER_GOOGLE_EMAIL", "your primary email")

    SYSTEM_PROMPT = build_system_prompt(user_email, current_date)

    def agent_node(state: AgentState):
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
            
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def confirmed_action_node(state: AgentState):
        return {
            "messages": [AIMessage(content="", tool_calls=[state["confirmed_action"]])],
            "confirmed_action": None,
        }

    def confirmation_node(state: AgentState):
        last_message = state["messages"][-1]
        pending_action = destructive_manage_event_tool_call(last_message)
        return {
            "messages": [AIMessage(content=format_pending_action_message(pending_action))],
            "pending_action": pending_action,
        }

    def start_route(state: AgentState) -> str:
        if state.get("confirmed_action"):
            return "confirmed_action"
        return "agent"

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return calendar_tool_route(last_message)

    def should_execute_confirmed_action(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("confirmed_action", confirmed_action_node)
    workflow.add_node("confirmation", confirmation_node)
    workflow.add_node("tools", ToolNode(all_tools))

    workflow.add_conditional_edges(START, start_route)
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_conditional_edges("confirmed_action", should_execute_confirmed_action)
    workflow.add_edge("confirmation", END)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

async def process_agent_stream(
    agent,
    initial_messages,
    confirmed_action: dict[str, Any] | None = None,
    return_pending: bool = False,
):
    # Process Agent stream, parse accurately and print tool execution progress
    final_message_content = None
    pending_action = None
    initial_state = {"messages": initial_messages}
    if confirmed_action:
        initial_state["confirmed_action"] = confirmed_action
    
    async for event in agent.astream(initial_state):
        for node_name, state_update in event.items():
            if node_name in {"agent", "confirmed_action", "confirmation"}:
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        name = tool_call["name"]
                        
                        # Print the AI's intended action without emojis
                        if "cli_" in name:
                            sys.stdout.write(f"[Action] Executing fast CLI tool '{name}'...\n")
                        elif "create" in name:
                            sys.stdout.write(f"[Action] Creating calendar event...\n")
                        elif "list" in name or "get" in name:
                            sys.stdout.write(f"[Action] Searching calendar events...\n")
                        else:
                            sys.stdout.write(f"[Action] Running tool '{name}'...\n")
                        sys.stdout.flush()
                else:
                    final_message_content = last_msg.content
                    if node_name == "confirmation":
                        pending_action = state_update.get("pending_action")
                    
            elif node_name == "tools":
                sys.stdout.write("[System] Action executed successfully.\n")
                sys.stdout.flush()
                
    display_text = ""
    if isinstance(final_message_content, list):
        for block in final_message_content:
            if isinstance(block, dict) and block.get("type") == "text":
                display_text += block.get("text", "")
            elif isinstance(block, str):
                display_text += block
    else:
        display_text = str(final_message_content)

    if return_pending:
        return display_text, pending_action
    return display_text

async def run_demo(agent) -> None:
    # Iterate over three pre-written demo queries without user input
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
    pending_action = None
    
    while True:
        sys.stdout.write("\nYou: ")
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()
        
        if not user_input or user_input.lower() in ['quit', 'exit']:
            break

        if pending_action:
            choice = parse_pending_action_choice(user_input)
            if choice == "confirm":
                chat_history.append(HumanMessage(content=user_input))
                display_text, _ = await process_agent_stream(
                    agent,
                    chat_history,
                    confirmed_action=pending_action,
                    return_pending=True,
                )
                pending_action = None
            elif choice == "cancel":
                chat_history.append(HumanMessage(content=user_input))
                pending_action = None
                display_text = "Cancelled. I did not modify the calendar event."
            else:
                print("\nAgent:\nPlease choose one option: 1. Confirm or 2. Cancel.")
                continue
        else:
            chat_history.append(HumanMessage(content=user_input))
            display_text, pending_action = await process_agent_stream(
                agent,
                chat_history,
                return_pending=True,
            )
            
        print(f"\nAgent:\n{display_text.strip()}")
        chat_history.append(AIMessage(content=display_text))

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
