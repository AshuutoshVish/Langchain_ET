from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

chat = ChatOpenAI(model_name="gpt-4")
system_message = SystemMessage(content="You are a helpful, friendly assistant.")

# Create memory to keep track of conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create conversation chain with chat model and memory
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    prompt=None  # Default prompt uses messages in memory + user input
)

# Start conversation with the system message included in the initial memory
memory.chat_memory.add_message(system_message)

# User sends first message
user_message_1 = HumanMessage(content="Hi, who won the World Cup in 2018?")
response_1 = conversation.predict(input=user_message_1.content)
print("Assistant:", response_1)

# User sends a follow-up message
user_message_2 = HumanMessage(content="And who was the top scorer?")
response_2 = conversation.predict(input=user_message_2.content)
print("Assistant:", response_2)

# To see all messages in the memory so far
print("\nConversation History:")
for msg in memory.chat_memory.messages:
    print(f"{msg.type}: {msg.content}")
