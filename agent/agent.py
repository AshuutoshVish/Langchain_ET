from langchain.agents import initialize_agent, AgentType
from tools.calculator_tool import calculator_tool
from tools.search_tool import search_tool
from utils.llm_initializer import get_llm

llm = get_llm()

tools = [calculator_tool, search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    while True:
        user_input = input("\n🧠 Ask something: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print(f"🤖 Response: {response}")
