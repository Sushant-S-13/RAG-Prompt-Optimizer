import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set your Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"

pdf_path = "reader.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="rag_db")
vectorstore.persist()

def prompt_formatter(query, context_docs):
    context = "\n".join([doc.page_content for doc in context_docs])
    return f"""
You are an expert at writing clear, detailed, optimized prompts.

Use the following context to enhance the original query. 
If the context isn't relevant or is insufficient:
1. Generate 2-3 follow-up questions to ask the user.
2. Format each question on a new line starting with "Question:".
3. Focus on clarifying ambiguous terms and identifying specific requirements.

Context:
{context}

Original Prompt:
{query}

Respond EITHER with an optimized prompt OR follow-up questions (never both).
"""

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

def retrieve_context(query, k=5):
    query_embedding = embedding_model.embed_query(query)
    return vectorstore.similarity_search_by_vector(query_embedding, k=k)

def generate_optimized_prompt(query):
    context_docs = retrieve_context(query)
    formatted_prompt = prompt_formatter(query, context_docs)
    response = llm.invoke(formatted_prompt)
    response_text = response.content.strip()


    if "Question:" in response_text:
        print("\n🔍 Need more information to optimize your prompt.")
        questions = [line.replace("Question:", "").strip()
                     for line in response_text.split("\n") if line.strip().startswith("Question:")]
        answers = []
        for i, question in enumerate(questions, 1):
            answer = input(f"{i}. {question}\n   Your answer: ")
            answers.append(f"{question} {answer}")
        # Append user answers to the original query for more context
        updated_query = f"{query}\nAdditional Context:\n" + "\n".join(answers)
        return generate_optimized_prompt(updated_query)
    else:
        return response_text

if __name__ == "__main__":
    user_query = input("Enter your original prompt: ")
    optimized = generate_optimized_prompt(user_query)
    print("\n🔧 Optimized Prompt:\n", optimized)
