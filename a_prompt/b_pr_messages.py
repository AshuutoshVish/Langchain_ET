from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

# Define the chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []
with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

query = "where is my refund?"
prompt = chat_template.invoke({"chat_history": chat_history,"query": query})
print(prompt)