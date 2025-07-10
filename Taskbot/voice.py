import os
import math
import warnings
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.tools.requests.tool import (RequestsGetTool, RequestsPostTool, RequestsPatchTool,RequestsPutTool, RequestsDeleteTool)
from langchain_community.utilities.requests import RequestsWrapper


warnings.filterwarnings("ignore")
load_dotenv()

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    raise EnvironmentError("Required environment variables are missing.")

#  directories exist for file operations
def ensure_dir(file_path: str):
    dir_name = os.path.dirname(file_path.strip())
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

# LLM initialization
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    top_p=1.0,
    max_retries=3,
    request_timeout=15,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Calculator tool using safe eval
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
        "e": math.e,
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

# CRUD on file
def read_file(file_path: str) -> str:
    try:
        path = file_path.strip().strip('"').strip("'")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(100)
            return content + ("\n\n... [truncated]" if len(content) == 100 else "...")
    except Exception as e:
        return f"File reading error: {str(e)}"

def write_file(input_str: str) -> str:
    try:
        path, content = input_str.split("||", 1)
        path = path.strip()
        ensure_dir(path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        return f"File written successfully at: {path}"
    except Exception as e:
        return f"File write error: {str(e)}"

def update_file(input_str: str) -> str:
    try:
        path, content = input_str.split("||", 1)
        path = path.strip()
        ensure_dir(path)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content.strip())
        return f"Content appended to file: {path}"
    except Exception as e:
        return f"File update error: {str(e)}"

def delete_file(file_path: str) -> str:
    try:
        os.remove(file_path.strip())
        return f"File deleted: {file_path.strip()}"
    except Exception as e:
        return f"File deletion error: {str(e)}"

file_tools = [
    Tool(name="ReadFile", func=read_file, description="Reads the content of a local text file. Input: path only."),
    Tool(name="WriteFile", func=write_file, description="Writes text to a file. Input: '<path>||<content>'."),
    Tool(name="UpdateFile", func=update_file, description="Appends text to a file. Input: '<path>||<content>'."),
    Tool(name="DeleteFile", func=delete_file, description="Deletes a file from local storage disk. Input: path only.")
]

search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    top_k=1,
    max_results=1,
    include_answers=False,
    include_raw_content=False
)

search_tool = Tool(
    name="Search",
    func=search.invoke,
    description="Useful for answering questions about current topics, events, facts, or data like GDP, weather, etc."
)

python_tool = Tool(
    name="PythonREPL",
    func=PythonREPLTool().run,
    description="Executes Python code. Use this for general Python calculations or list operations."
)

requests_wrapper = RequestsWrapper()

requests_tools = [
    Tool(name="requests_get", func=RequestsGetTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run, description="Make GET requests to fetch data from APIs."),
    Tool(name="requests_post", func=RequestsPostTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run, description="Make POST requests to send data to APIs."),
    Tool(name="requests_patch", func=RequestsPatchTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run, description="Make PATCH requests to update part of a resource."),
    Tool(name="requests_put", func=RequestsPutTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run, description="Make PUT requests to fully replace a resource."),
    Tool(name="requests_delete", func=RequestsDeleteTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True).run, description="Make DELETE requests to remove a resource.")
]

tools = [calculator_tool, search_tool, python_tool] + file_tools + requests_tools

def transcribe_voice_to_text() -> str:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Speak now...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

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
    print("\nLangChain Agent with Voice Support is ready...!!")
    print("Type inputs or type 'voice' to use voice input......!!")
    print("Type 'exit' or 'quit' to end the conversation and exit....!!")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'stop', 'bye', 'goodbye', 'esc']:
                print("Exiting the conversation...!!")
                break

            if user_input.lower() == "voice":
                user_input = transcribe_voice_to_text()

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