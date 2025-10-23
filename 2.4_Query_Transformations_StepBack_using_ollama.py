import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4

# ---- STEP 1: LOAD & INDEX ----

loader = WikipediaLoader(query="Obesity", lang="en", load_max_docs=3)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# ---- STEP 2: STEPBACK QUERY GENERATION ----

stepback_template = """
You are an AI assistant. Given a specific question, generate a more general, high-level question that would help understand the broader context.
The stepback question should focus on the underlying concepts and principles.

Specific question: {question}

Stepback question:"""

prompt = PromptTemplate.from_template(stepback_template)
llm = OllamaLLM(model="llama3.2")

generate_stepback_chain = (
    {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# ---- STEP 3: RAG ANSWERING ----

rag_prompt = PromptTemplate.from_template(
    """
You are an intelligent assistant.

Use the following context to answer the question. If the context is unclear, incomplete, or unrelated, feel free to use your own knowledge to give the best answer.

Be clear, accurate, and to the point.

Context:
{context}

Question: {question}
"""
)


format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])


def retrieve_with_stepback(original_question):
    # Step 1: generate stepback question
    stepback_question = generate_stepback_chain.invoke(original_question)

    print("\n=== Original Question ===")
    print(original_question)
    print("\n=== Generated Stepback Question ===")
    print(stepback_question)

    # Step 2: retrieve docs for both questions
    original_docs = retriever.invoke(original_question)
    stepback_docs = retriever.invoke(stepback_question)

    # Step 3: combine and remove duplicates
    all_docs = original_docs + stepback_docs
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs


# ---- STEP 4: RUN ----

question = "What are the health risks associated with obesity?"
relevant_docs = retrieve_with_stepback(question)

# Format + LLM call
final_prompt = rag_prompt.format(context=format_docs(relevant_docs), question=question)
answer = llm.invoke(final_prompt)

print("\n=== User Question ===")
print(question)
print("\n=== Final Answer ===")
print(answer)
