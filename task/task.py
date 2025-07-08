import os
import logging
from typing import List, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools.python.tool import PythonREPLTool

# LangGraph Imports
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def validate_environment() -> None:
    """Check if required API keys are present."""
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")


class SafePythonREPLTool(PythonREPLTool):
    """A safer version of PythonREPLTool that restricts dangerous operations."""
    def _run(self, query: str) -> str:
        forbidden_keywords = ["os.system", "subprocess", "open(", "exec(", "eval(", "import os"]
        if any(keyword in query.lower() for keyword in forbidden_keywords):
            return "Error: Restricted Python operation."
        return super()._run(query)

def initialize_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
        api_key=os.getenv("OPENAI_API_KEY")
    )

def load_agent_tools(llm: OpenAI) -> List[Any]:
    tools = []

    # Prefering SerpAPI → Tavily → DuckDuckGo --> As i dont have serAPI Key
    if os.getenv("SERPAPI_API_KEY"):
        try:
            tools.extend(load_tools(["serpapi"], llm=llm, serpapi_api_key=os.getenv("SERPAPI_API_KEY")))
        except Exception as e:
            logger.warning(f"Failed to load SerpAPI: {e}")
    elif os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tools.append(TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY")))
        except Exception as e:
            logger.warning(f"Failed to load Tavily: {e}")
    else:
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            tools.append(DuckDuckGoSearchResults())
        except Exception as e:
            logger.warning(f"Failed to load DuckDuckGo: {e}")

    # LLM Math, Requests, Safe Python REPL
    try:
        tools.extend(load_tools(["llm-math", "requests_all"], llm=llm, allow_dangerous_tools=True))
        tools.append(SafePythonREPLTool())
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        raise RuntimeError("Critical tools failed to load.")
    
    return tools


def create_agent(llm: OpenAI, tools: List[Any]) -> Runnable:
    converted_tools = [convert_to_openai_tool(tool) for tool in tools]
    react_agent = create_react_agent(llm, converted_tools)

    class AgentState(dict): pass

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", react_agent)
    graph_builder.set_entry_point("agent")
    graph_builder.set_finish_point("agent")

    return graph_builder.compile()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)


def process_user_input(agent: Runnable, query: str, history: InMemoryChatMessageHistory) -> str:
    try:
        logger.info(f"Processing query: {query}")
        result = agent.invoke({"messages": history.messages + [HumanMessage(content=query)]})
        response_msg = result.get("messages", [])[-1]

        history.add_user_message(query)
        history.add_ai_message(response_msg.content)

        return response_msg.content
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error: {str(e)}"



def main():
    validate_environment()
    llm = initialize_llm()
    tools = load_agent_tools(llm)
    agent = create_agent(llm, tools)
    # history = ChatMessageHistory()
    history = InMemoryChatMessageHistory()


    test_queries = [
        "What is the GDP of Germany and its square root?",
        "What's the current weather in New York?",
        "Calculate 15% of 2500",
        "Open a file called test.txt",  # Should be blocked
    ]

    for query in test_queries:
        print(f"\n🔹 Query: {query}")
        response = process_user_input(agent, query, history)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
