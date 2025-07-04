from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
prompt = PromptTemplate(input_variables=["question"],template="You are a knowledgeable assistant. Answer the question: {question}")
user_question = input("Ask a relevant question from your model: ")
formatted_prompt = prompt.format(question=user_question,
                                #  question="Who is the current president of the United States?"
                                 )
response = llm.invoke(formatted_prompt)
model_used = response.response_metadata.get("model_name")

print("Answer:", response.content)
print("Model Used:", model_used)

