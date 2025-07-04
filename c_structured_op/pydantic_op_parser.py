from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_output_tokens=100)

class PersonInfo(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="Enter the city name of the person")
    email: EmailStr = Field(description="Email of the person")

parser = PydanticOutputParser(pydantic_object=PersonInfo)

template = PromptTemplate(
    template="Generate the name, age, city, and email of a fictional {place} person.\n{format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

place = input("Enter the name of country : ")
formatted_prompt = template.format(place=place)
llm_response = llm.invoke(formatted_prompt)


result = parser.parse(llm_response.content)
print("Parsed Output: ", result)

print("\nRaw LLM Response: ", llm_response)

