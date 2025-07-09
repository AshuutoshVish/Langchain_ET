import os
import ast
import logging
from typing import List, Any
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.agent_toolkits.load_tools import load_tools

from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_tool

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_environment() -> None:
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")

class SafePythonREPLTool(PythonREPLTool):
    name: str = "SafePythonREPL"

    def _run(self, query: str) -> str:
        forbidden_keywords = ["os", "subprocess", "eval", "exec", "open", "__import__"]
        lowered_query = query.lower()
        if any(keyword in lowered_query for keyword in forbidden_keywords):
            return "Error: Restricted Python operation."

        try:
            tree = ast.parse(query, mode="exec")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return "Error: Import statements are restricted."
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "open", "__import__"}:
                        return f"Error: Use of '{node.func.id}' is not allowed."
            return super()._run(query)
        except Exception as e:
            return f"Execution Error: {str(e)}"

def initialize_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_retries=3,
        max_tokens=1000,
    )

def load_agent_tools(llm: ChatOpenAI) -> List[Any]:
    tools = []
    try:
        tools.append(TavilySearch(api_key=os.getenv("TAVILY_API_KEY")))
        logger.info("Tavily Search tool loaded.")
    except Exception as e:
        logger.warning(f"Failed to load Tavily tool: {e}")

    try:
        tools += load_tools(["llm-math", "requests_all"], llm=llm, allow_dangerous_tools=True)
        tools.append(SafePythonREPLTool())
        tool_names = [getattr(tool, "name", type(tool).__name__) for tool in tools]
        logger.info(f"Tools loaded: {tool_names}")

    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        raise RuntimeError("Critical tools failed to load.")
    return tools

# LangGraph agent creation
def create_agent(llm: ChatOpenAI, tools: List[Any]) -> Runnable:
    try:
        converted_tools = [convert_to_openai_tool(tool) for tool in tools]
        react_agent = create_react_agent(llm, converted_tools)

        class AgentState(dict): pass
        graph = StateGraph(AgentState)
        graph.add_node("agent", react_agent)
        graph.set_entry_point("agent")
        graph.set_finish_point("agent")
        return graph.compile()
    except Exception as e:
        logger.error(f"Failed to create LangGraph agent: {e}")
        raise

# Handle user input and get agent output
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def process_user_input(agent: Runnable, query: str, history: InMemoryChatMessageHistory) -> str:
    try:
        logger.info(f"Processing query: {query}")

        result = agent.invoke({
            "input": query,
            "chat_history": history.messages
        })

        messages = result.get("messages", [])
        if not messages:
            raise ValueError("No response from agent.")

        final_message = messages[-1]
        if not isinstance(final_message, AIMessage):
            raise ValueError("Final message is not an AIMessage.")

        history.add_user_message(query)
        history.add_ai_message(final_message.content)

        return final_message.content

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"Error: {str(e)}"

# Main loop
def main():
    validate_environment()
    llm = initialize_llm()
    tools = load_agent_tools(llm)
    agent = create_agent(llm, tools)
    history = InMemoryChatMessageHistory()

    # Add initial messages to avoid empty history
    history.add_user_message("Hi")
    history.add_ai_message("Hello! How can I assist you?")

    print("🤖 Agent initialized successfully!")
    print("\n💬 Ask your questions (type 'exit' to quit):")

    while True:
        user_input = input("\n🔹 You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = process_user_input(agent, user_input, history)
        print(f"🤖 Agent: {response}")

# ✅ Entry point
if __name__ == "__main__":
    main()
