from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage 
load_dotenv()

def get_name_generator(user_input):
    """
    Generates a creative name for the user-provided product description.
    """
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    prompt = f"Generate a unique and creative name for {user_input}."
    messages = [HumanMessage(content=prompt)]
    response = chat.invoke(messages)
    return response.content

if __name__ == "__main__":
    user_input = input("What is your product or idea? Describe it: ")
    name = get_name_generator(user_input)
    print(f"Generated Name: {name}")
