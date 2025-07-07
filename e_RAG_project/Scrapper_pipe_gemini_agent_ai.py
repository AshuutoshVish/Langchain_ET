import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
load_dotenv()

# document loader
def load_documents(url: str):
    print(f"Loading website content from {url}")
    loader = WebBaseLoader([url])
    return loader.load()

# Text splitter for chunking
def split_documents(docs):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_documents(docs)

#Generating embeddings for similarity and vector store for storing data
def embed_documents(splits):
    print("Embedding documents using Google Generative AI...")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    return vectorstore

#Retrieval chain
def build_retrieval_chain(vectorstore):
    print("Building RetrievalQA chain using Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_output_tokens=100)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=False
    )
    return qa_chain


def run_chat_loop(qa_chain, vectorstore):
    print("\nAsk questions about the site (type 'exit' to quit):")
    chat_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.....!")
            break

        # Fallback if nothing is found
        relevant_docs = vectorstore.similarity_search(user_input, k=2)
        if not relevant_docs:
            print("AI: Sorry, no relevant info found on that topic.")
            continue

        try:
            result = qa_chain.invoke({"query": user_input})
            ai_response = result['result']
            chat_history.append((user_input, ai_response))

            print("\nChat History:")
            for idx, (q, a) in enumerate(chat_history, 1):
                print(f"\n🧑 Q{idx}: {q}\n🤖 A{idx}: {a}")

        except Exception as e:
            print("Error:", e)


def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    url = "https://excellencetechnologies.in/"
    docs = load_documents(url)
    splits = split_documents(docs)
    vectorstore = embed_documents(splits)
    qa_chain = build_retrieval_chain(vectorstore)
    run_chat_loop(qa_chain, vectorstore)


if __name__ == "__main__":
    main()