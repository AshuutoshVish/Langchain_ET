from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = ChatGroq(model="llama3-8b-8192",
               temperature=0.0
               )
prompt = PromptTemplate(input_variables=["question"],
                        template="You are a knowledgeable assistant. Answer the question: {question}"
                        )

user_question = input("Ask a question: ")
formatted_prompt = prompt.format(question=user_question)

response = llm.invoke(formatted_prompt)
print("Answer:", response.content)
print("Answer:", response)
