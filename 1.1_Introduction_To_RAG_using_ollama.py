
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import bs4

"""
| Parameter | Impact |
|:---|:---|
| `chunk_size` too small | → Less context for LLM, may miss meaning |
| `chunk_size` too large | → Slower search, noisy embeddings, risk of truncation |
| `chunk_overlap` too small | → Risk of broken thoughts at chunk boundaries |
| `chunk_overlap` too large | → Redundant context, more memory usage, slower indexing |

"""
# ---- INDEXING ----

# Define the persistent directory for Chroma
PERSIST_DIRECTORY = "./database/chroma_db"

# Check if the database already exists
if not os.path.exists(PERSIST_DIRECTORY):
    # If the database doesn't exist, process and save the data
    loader = WikipediaLoader(query="Cancer", lang="en", load_max_docs=3)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY
    )
else:
    # If the database exists, load it directly
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model
    )

retriever = vectorstore.as_retriever()

# ---- RETRIEVAL + GENERATION ----

prompt_template = """
Use the following context to answer the question. 
If the context does not help or is unrelated, you may answer from your own knowledge.

Context:
{context}

Question: {question}
"""

llm = OllamaLLM(model="llama3.2")  # or any model you have installed


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(lambda x: prompt_template.format(**x))
    | llm
    | StrOutputParser()
)
question = "What are the main types of cancer?"
print(f"\n=== User Question ===")
print(question)

response = rag_chain.invoke(question)

print(f"\n=== Final Answer ===")
print(response)
