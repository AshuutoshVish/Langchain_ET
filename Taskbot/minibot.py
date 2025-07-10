import os
import math
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool, AgentExecutor
# from langchain.memory import ConversationBufferMemory
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.tools.requests.tool import (RequestsGetTool, RequestsPostTool, RequestsPatchTool,RequestsPutTool,RequestsDeleteTool)
from langchain_community.utilities.requests import RequestsWrapper


import warnings
warnings.filterwarnings("ignore")

load_dotenv()
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    raise EnvironmentError("Required environment variables are missing.")


#Llm initialisation
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    top_p=1.0,
    max_retries=3,
    request_timeout=15,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Function to calculate mathematical expressions safely
def calculate_expression(input_str: str) -> str:
    safe_dict = {
        "sqrt": math.sqrt,
        "log": math.log,
        "pow": pow,
        "abs": abs,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e
    }
    try:
        result = eval(input_str, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    func=calculate_expression,
    description="Useful for math problems including sqrt(x), log(x), power(**), parentheses, etc."
)

# For Reading and writing file
def read_file(file_path: str) -> str:
    try:
        path = file_path.strip().strip('"').strip("'")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(100)
            return content + ("\n\n... [truncated]" if len(content) == 100 else "...")
    except Exception as e:
        return f"File reading error: {str(e)}"

file_tool = Tool(
    name="ReadFile",
    func=read_file,
    description=(
        "Reads text files from the local disk. "
        "The input should be a valid full file path like 'C:/Users/Ashu/ET/Langchain/Taskbot/test.txt'. "
        "If the user hasn't provided a file path, ask them to specify the file path explicitly."
    )
)


# Tavily Search Tool for web search
search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), 
                      top_k=1, 
                      max_results=1, 
                      include_answers=False, 
                      include_raw_content=False,)

search_tool = Tool(
    name="Search",
    func=search.invoke,
    description="Useful for answering questions about current topics, events, facts, or data like GDP, weather, etc."
)


# REPL Tool for executing Python code
python_tool = Tool(
    name="PythonREPL",
    func=PythonREPLTool().run,
    description="Executes Python code. Use this for general Python calculations or list operations."
)

# Requests Tools for making HTTP requests
requests_wrapper = RequestsWrapper()

requests_tools = [
    Tool(
        name="requests_get",
        func=RequestsGetTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run,
        description="Make GET requests to fetch data from APIs."
    ),
    Tool(
        name="requests_post",
        func=RequestsPostTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run,
        description="Make POST requests to send data to APIs."
    ),
    Tool(
        name="requests_patch",
        func=RequestsPatchTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run,
        description="Make PATCH requests to update part of a resource."
    ),
    Tool(
        name="requests_put",
        func=RequestsPutTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run,
        description="Make PUT requests to fully replace a resource."
    ),
    Tool(
        name="requests_delete",
        func=RequestsDeleteTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run,
        description="Make DELETE requests to remove a resource."
    )
]


# Assembling tools
tools = [calculator_tool, file_tool, search_tool, python_tool,]  + requests_tools



agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)


def main():
    print("\nLangChain Agent with Tavily is ready. Ask your question (type 'exit or quit' to quit)")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'stop', 'bye', 'goodbye', 'esc']:
                print("Exiting the program...!!")
                break

            response = agent_executor.invoke({"input": user_input})
            output = response.get('output', '').strip()
            if not output:
                print("\nAgent: I couldn't generate a response for that. Try rephrasing your query.")
            else:
                print(f"\nAgent: {output}\n")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting the program...")
            break

        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()