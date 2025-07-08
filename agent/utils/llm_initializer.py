import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
