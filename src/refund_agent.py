import os
import sys
import asyncio
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_config import setup_environment, get_mcp_config
setup_environment("refund_mcp.log")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

REFUND_MCP_TOOL_NAMES = (
    "search_gmail_messages",
    "get_gmail_message_content",
    "get_gmail_messages_content_batch",
    "send_gmail_message",
    "draft_gmail_message",
    "get_gmail_thread_content",
    "list_gmail_labels",
    "modify_gmail_message_labels",
)

def _print_setup_guide() -> None:
    print("Missing required environment variables. Please follow the setup guide:")
    print("1. Install workspace-mcp: `uv tool install google_workspace_mcp`")
    print("2. Set up Google Cloud Project and enable Gmail API.")
    print("3. Configure OAuth consent screen and create Desktop App credentials.")
    print("4. Export the following variables:")
    print("   export GOOGLE_API_KEY=<your-google-ai-studio-key>")
    print("   export GOOGLE_OAUTH_CLIENT_ID=<your-client-id>")
    print("   export GOOGLE_OAUTH_CLIENT_SECRET=<your-client-secret>")
    print("   export USER_GOOGLE_EMAIL=<your-email>")

def filter_refund_mcp_tools(tools):
    tools_by_name = {tool.name: tool for tool in tools}
    return [tools_by_name[name] for name in REFUND_MCP_TOOL_NAMES if name in tools_by_name]

def build_system_prompt(user_email: str) -> str:
    return f"""
    You are an autonomous customer service agent handling a Gmail inbox for refund, return requests, and customer complaints.

    Your authorized Gmail address is: {user_email}
    CRITICAL RULE: Whenever a tool requires an email address, you MUST use EXACTLY {user_email}. NEVER use placeholder emails like 'customer_service@example.com'.

    Your workflow is:
    1. SEARCH inbox for relevant customer service emails using `search_gmail_messages` (e.g., search for unread messages).
    2. READ THE FULL THREAD using `get_gmail_thread_content` to understand the conversation history.
    3. EVALUATE HISTORY: Combine the historical conversation context carefully. Review who sent the last message. If the last message was sent by you ({user_email}), skip it. If the customer replied, address their new concern based on the history.
    4. CLASSIFY intent into one of the categories below.
    5. DRAFT or SEND threaded reply using `draft_gmail_message` when uncertain or `send_gmail_message` when confident.
    6. MARK AS READ: After successfully sending a reply or creating a draft, you MUST immediately use `modify_gmail_message_labels` to remove the 'UNREAD' label from the processed message or thread. This prevents reprocessing.
    7. REPORT summary.

    Email Classifications and Actions:
    - REFUND REQUEST: Customer wants money back. Action: Send refund approval reply (3-5 day processing).
    - RETURN REQUEST: Customer wants to return product. Action: Send return instructions with prepaid label steps.
    - COMPLAINT: General dissatisfaction. Action: Send empathetic acknowledgement, 24hr follow-up promise.
    - OTHER: Unrelated content. Action: Skip, no reply sent.

    Hard Rules:
    - ALWAYS thread replies using the correct thread_id.
    - NEVER double-reply: Always verify the last sender in the thread. If you ({user_email}) are the last sender, take no action and skip to the next email.
    - NEVER reply to OTHER emails.
    - ALWAYS remove the 'UNREAD' label using tools after processing a valid email to clean up the queue.
    - When uncertain, prefer using `draft_gmail_message` over `send_gmail_message` to allow human review.
    """

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

async def build_agent(mcp_client: MultiServerMCPClient):
    tools = filter_refund_mcp_tools(await mcp_client.get_tools())
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it", 
        temperature=0,
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    llm_with_tools = llm.bind_tools(tools)

    user_email = os.environ.get("USER_GOOGLE_EMAIL", "your email")

    SYSTEM_PROMPT = build_system_prompt(user_email)

    def agent_node(state: AgentState) -> AgentState:
        messages = list(state["messages"])
        
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent_node")
    workflow.add_conditional_edges(
        "agent_node",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent_node")

    return workflow.compile()

async def process_agent_stream(agent, initial_messages) -> str:
    final_message_content = None
    
    async for event in agent.astream({"messages": initial_messages}):
        for node_name, state_update in event.items():
            if node_name == "agent_node":
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        name = tool_call["name"]
                        args = tool_call.get("args", {})
                        
                        if "search" in name:
                            sys.stdout.write(f"[Action] Searching Inbox using query: '{args.get('query', '')}'...\n")
                        elif "get_gmail_thread" in name:
                            sys.stdout.write(f"[Action] Reading full Email Thread (ID: {args.get('thread_id', args.get('id', ''))})...\n")
                        elif "get_gmail_message" in name:
                            sys.stdout.write(f"[Action] Reading Email Content (ID: {args.get('message_id', args.get('id', ''))})...\n")
                        elif "draft" in name:
                            sys.stdout.write(f"[Action] Classifying intent and Creating a GMAIL DRAFT...\n")
                        elif "send" in name:
                            sys.stdout.write(f"[Action] Classifying intent and SENDING a threaded reply...\n")
                        elif "modify" in name:
                            sys.stdout.write(f"[Action] Modification triggered: Removing UNREAD label from message...\n")
                        else:
                            sys.stdout.write(f"[Action] Running tool '{name}'...\n")
                        sys.stdout.flush()
                else:
                    final_message_content = last_msg.content
                    
            elif node_name == "tools":
                for msg in state_update.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        content_str = str(msg.content)
                        if "search" in getattr(msg, "name", "").lower() or "search" in getattr(msg, "tool_call_id", "").lower():
                            count = content_str.count("'id':") + content_str.count('"id":')
                            sys.stdout.write(f"[System] Total email(s) found matching criteria: {count}\n")
                        else:
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

async def run_auto_refund_processing(agent) -> None:
    print("\n" + "="*50)
    print("[INFO] Starting Auto-Processing Mode...")
    print("="*50)
    
    initial_message = [HumanMessage(content="Process all unread customer service emails (including refunds, returns, and complaints) using the 6-step workflow.")]
    display_text = await process_agent_stream(agent, initial_message)
        
    print("\n" + "="*50)
    print("[Agent Summary]")
    print("-" * 50)
    print(display_text.strip())
    print("="*50 + "\n")

async def run_interactive_chat(agent):
    print("\n" + "="*50)
    print("[INFO] Refund Agent Interactive Mode Ready! (Type 'quit' or 'exit' to exit)")
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

async def main() -> None:
    required_envs = ["GOOGLE_API_KEY", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "USER_GOOGLE_EMAIL"]
    if not all(os.environ.get(env) for env in required_envs):
        _print_setup_guide()
        sys.exit(1)

    print("\n[INFO] Loading MCP Server configuration...")
    mcp_config = get_mcp_config("refund")    
    
    print("[INFO] Starting Workspace MCP Server (Transport: stdio)...")
    mcp_client = MultiServerMCPClient(mcp_config)
    
    try:
        print("[INFO] Connecting to Gmail API and building AI Agent...")
        agent = await build_agent(mcp_client)
        print("[INFO] Initialization complete!")
        print("-" * 50)
        
        # Route to appropriate run mode based on user input
        sys.stdout.write("Select mode - (1) Auto Processing (2) Interactive Chat: ")
        sys.stdout.flush()
        mode = sys.stdin.readline().strip()
        
        if mode == "1":
            await run_auto_refund_processing(agent)
        else:
            await run_interactive_chat(agent)
            
    finally:
        if hasattr(mcp_client, 'close'):
            print("\n[INFO] Shutting down MCP Server...")
            await mcp_client.close()
            print("[INFO] Disconnected. Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
