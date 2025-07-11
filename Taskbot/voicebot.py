#Voice AI Agentic bot with Tavily Search and Calculator
#Use streamlit run voicebot.py

import os
import math
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import ReadFileTool, WriteFileTool, DeleteFileTool

load_dotenv()
st.set_page_config(page_title="AI Agent", layout="centered")

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("OPENAI_API_KEY or TAVILY_API_KEY is missing in your .env file.")
    st.stop()

# Initializing LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Custom safe calculator tool
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
    description="Evaluate mathematical expressions including operations like addition, subtraction, multiplication, division, square root, power, logarithm, and constants (π, e). Use functions like sqrt(x), pow(x, y), log(x), etc.")

# Tavily Search Tool
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
    description="Use this tool to perform real-time web searches and retrieve up-to-date information on topics such as news, facts, statistics, global events, or public data (e.g., GDP, population, weather). Best for answering current or factual queries."
)

# Python REPL
python_tool = Tool(
    name="PythonREPL",
    func=PythonREPLTool().run,
    description="Execute Python code for calculations, logic, or data manipulation. Use it for tasks like computing values, checking conditions, running loops, or evaluating expressions in real-time."
)

file_tools = [ReadFileTool(), WriteFileTool(), DeleteFileTool()]

# HTTP Requests tools via RequestsToolkit
requests_wrapper = TextRequestsWrapper()
requests_toolkit = RequestsToolkit(requests_wrapper=requests_wrapper,allow_dangerous_requests=True)
requests_tools = requests_toolkit.get_tools()

# Combining tools
tools = [calculator_tool, search_tool, python_tool] + file_tools + requests_tools

# Initialize agent using OPENAI_FUNCTIONS for multi-input tool support
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# Agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=tools,
    llm=llm,
    memory=memory,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)

def transcribe_voice_to_text() -> str:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Listening... Please speak.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

# Streamlit UI
st.title("Voice Agent")
st.markdown("Ask anything, manage files, interact with APIs, or use voice input.")

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Ask me anything:")
with col2:
    voice_triggered = st.button("Voice Input")

if voice_triggered:
    user_input = transcribe_voice_to_text()
    st.success(f"You said: {user_input}")

if user_input:
    user_input = user_input.strip()
    if not user_input or "sorry" in user_input.lower():
        st.warning("Input unclear. Try again.")
        st.stop()

    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        output = response.get("output", "No response generated.")
        st.markdown(f"### Agent Response:\n{output}")
        with st.expander("View Full Agent Output"):
            st.json(response)