from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.7,
                             max_output_tokens=200
                             )

schema = [
    ResponseSchema(name="fact", description="A relevant fact about the topic"),
    ResponseSchema(name="joke", description="A funny joke about the topic"),
    ResponseSchema(name="summary", description="A summary of the joke")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate( template=("You are a helpful assistant.\n"
                                     "Generate the following for {topic}:"
                                     "1. A relevant fact\n"
                                     "2. A funny joke\n"
                                     "3. A short summary of the joke\n\n"
                                     "{format_instruction}"),
                                     input_variables=["topic"],
                                     partial_variables={'format_instruction': parser.get_format_instructions()}
                                     )

chain = template | llm | parser

topic = input("Enter the topic : ")
result = chain.invoke({"topic": topic})

# Print the result in a human-readable format

print("The Topic is :", topic)
print(f"\nThe Fact is : {result['fact']}")
print(f"\nJoke about {topic} is : {result['joke']}")
print(f"\nSummary: {result['summary']}")

print(f"\n\nresults: {result}")