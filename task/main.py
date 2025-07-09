import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType

import warnings
warnings.filterwarnings("ignore")



load_dotenv()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    max_retries=3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def calculate_expression(input_str: str) -> str:
    try:
        result = eval(input_str)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"
    

calculator_tool = Tool(
    name="Calculator",
    func=calculate_expression,
    description="Useful for math problems like addition, multiplication, or square roots."
)

def read_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"File reading error: {str(e)}"


file_tool = Tool(
    name="ReadFile",
    func=read_file,
    description="Reads text files from local disk. Input must be a valid local file path like 'test.txt'."
)

#Tavily Search Tool
search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
search_tool = Tool(
    name="Search",
    func=search.invoke,
    description="Useful for answering questions about current events, facts, or data like GDP, weather, etc."
)

# Assembling tools
tools = [calculator_tool, file_tool, search_tool]

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=tools,
    verbose=True,
    max_iterations=5, # Maximum number of iterations for the agent to run
    early_stopping_method="generate"
)

def main():
    print("\nLangChain Agent with Tavily is ready. Ask your question (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        try:
            response = agent_executor.invoke(user_input)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()