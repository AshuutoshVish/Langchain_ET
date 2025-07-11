
import os
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import TypedDict, List, Any
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

load_dotenv()

class SafePythonREPLTool(PythonREPLTool):
    def _run(self, query: str) -> str:
        forbidden = ["import os", "import sys", "open(", "__import__", "eval(", "exec(", "subprocess", "shutil", "rm ", "del ", "write("]
        query_lower = query.lower()
        if any(cmd in query_lower for cmd in forbidden):
            logging.warning("Blocked potentially dangerous Python code: %s", query)
            return "Error: Potentially dangerous operation blocked for security reasons."
        logging.info("Executing safe Python code: %s", query)
        return super()._run(query)

def validate_environment() -> None:
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        logging.error("Missing environment variables: %s", ', '.join(missing_keys))
        raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")
    logging.info("All required environment variables are present.")

def get_tools():
    try:
        validate_environment()
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        tools = load_tools(["requests_get", "requests_post", "requests_patch", "requests_delete", "llm-math"],
                           llm=llm,
                           allow_dangerous_tools=True
                           )
        tools.append(TavilySearch())
        # tools.append(TavilySearch(include_answer=True, include_raw_content=False, search_depth="basic"))
        tools.append(SafePythonREPLTool())

        tool_descriptions = {
            "requests_get": "retrieve data from a URL via GET request",
            "requests_post": "send data to a URL via POST request",
            "requests_patch": "partially update a resource via PATCH request",
            "requests_delete": "delete a resource via DELETE request",
            "llm-math": "perform mathematical calculations",
            "tavily_search": "search the web for current information",
            "python_repl": "execute Python code (restricted for safety)"
        }

        for tool in tools:
            if hasattr(tool, 'description'):
                tool.description = f"{tool.description} Use this tool when you need to {tool_descriptions.get(tool.name, 'perform this specific action')}."
        logging.info("Tools initialized: %s", [tool.name for tool in tools if hasattr(tool, 'name')])
        return tools
    except Exception as e:
        logging.exception("Error initializing tools")
        raise

class AgentState(TypedDict):
    input: str
    chat_history: List[Any]
    agent_scratchpad: List[Any]

# Creating LangChain agent
def create_agent():
    tools = get_tools()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that uses tools to answer user questions. "
                   "Be concise but thorough in your responses. Always consider security and privacy when using tools."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
    logging.info("Agent created successfully.")
    return executor

# Add a simple rate limiter
class RateLimitedExecutor:
    def __init__(self, executor):
        self.executor = executor
        self.last_call = None
        self.min_interval = timedelta(seconds=1)

    def invoke(self, input_dict):
        now = datetime.now()
        if self.last_call and (now - self.last_call) < self.min_interval:
            wait_time = (self.min_interval - (now - self.last_call)).total_seconds()
            logging.info("Rate limiting: sleeping for %.2f seconds", wait_time)
            time.sleep(wait_time)
        self.last_call = datetime.now()
        logging.info("Invoking agent with input: %s", input_dict.get("input"))

        return self.executor.invoke(input_dict)

# Wrapping agent in LangGraph graph
def create_graph_agent() -> Runnable:
    builder = StateGraph(AgentState)
    builder.add_node("agent_executor", create_agent())
    builder.set_entry_point("agent_executor")
    builder.add_edge("agent_executor", END)
    logging.info("Graph agent created and compiled.")
    return builder.compile()

# Main REPL loop
if __name__ == "__main__":
    try:
        agent = create_graph_agent()
        rate_limited_agent = RateLimitedExecutor(agent)
        logging.info("rate_limited_agent:  %s", rate_limited_agent)
        
        logging.info("🤖 The Smart Task Agent is ready!")
        print("\n🤖 The Smart Task Agent is ready!")
        print("• Type 'exit' or 'quit' to end the session")
        print("• Press Ctrl+C to cancel a running operation\n")
        chat_history = []
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye! Have a great day!")
                    logging.info("Session ended by user.")
                    break

                logging.info("User input: %s", user_input)
                response = rate_limited_agent.invoke({"input": user_input,
                                                      "chat_history": chat_history})
                logging.info("Agent output: %s", response)
                output = response.get("output", "I didn't get a response.")
                logging.info("Agent output: %s", output)
                print(f"\nAssistant: {output}\n")

                chat_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": output}
                ])

            except KeyboardInterrupt:
                print("\nOperation cancelled. You can continue with your next question.")
                logging.info("Operation cancelled by user (KeyboardInterrupt).")
                continue

            except Exception as e:
                logging.exception("Error during agent interaction")
                print(f"\nError: {str(e)}")
                print("Please try again or rephrase your question.\n")

    except Exception as e:
        logging.exception("Failed to initialize agent")
        print(f"Failed to initialize agent: {str(e)}")
        print("Please check your environment variables and try again.")