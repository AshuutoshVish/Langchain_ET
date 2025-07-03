from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=100,
    request_timeout=30,
    max_retries=3
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a knowledgeable assistant. Answer the following question clearly:\n\n{question}"
)

user_question = input("Ask a question: ")
formatted_prompt = prompt.format(question=user_question)
response = llm.invoke(formatted_prompt)
print("\nAnswer:", response.content)