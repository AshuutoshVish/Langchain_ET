import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.7)

st.title("Bot")
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input" not in st.session_state:
    st.session_state.input = ""

# Input handler function
def handle_input():
    user_input = st.session_state.input
    if user_input:
        response = conversation.predict(input=user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.input = ""

st.text_input("You:", key="input", on_change=handle_input)


conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=True,
)


for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")

with st.expander("Conversation History"):
    st.write(st.session_state.memory.buffer)