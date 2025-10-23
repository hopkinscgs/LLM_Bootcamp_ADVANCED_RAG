import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4

# ---- STEP 1: LOAD & INDEX ----

loader = WikipediaLoader(query="Heart disease", lang="en", load_max_docs=3)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# ---- STEP 2: QUERY DECOMPOSITION SETUP ----

decomposition_template = """
You are an AI assistant.

Given a complex question, break it down into 3 or 4 simpler sub-questions that each explore one part of the topic.
Make the sub-questions easy to understand and focused on specific concepts. Each should help answer the main question when combined.

Complex question: {question}

Sub-questions:
"""


prompt = PromptTemplate.from_template(decomposition_template)
llm = OllamaLLM(model="llama3.2")

generate_subquestions_chain = (
    {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# ---- STEP 3: RAG ANSWERING WITH DECOMPOSITION ----

# RAG prompt for individual sub-questions
rag_prompt = PromptTemplate.from_template(
    """
You are a helpful assistant.

Use the following context to answer the question. If the context is unclear, incomplete, or unrelated, use your own knowledge to provide the best answer.

Context:
{context}

Question: {question}
"""
)


# Final synthesis prompt
synthesis_template = """
You are a helpful assistant.

Here is a list of sub-questions and their answers related to a complex question.

Use these to write a single, clear, well-organized answer to the original question. 
If some sub-answers are weak or unrelated, rely on your general knowledge to fill in the gaps.

Q+A Pairs:
{context}

Original Question: {question}
"""


synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


def retrieve_and_rag(question):
    """RAG on each sub-question"""

    # Generate sub-questions
    subquestions_text = generate_subquestions_chain.invoke(question)
    subquestions = [
        q.strip() for q in subquestions_text.strip().split("\n") if q.strip()
    ]

    print("\n=== User Question ===")
    print(question)
    print("\n=== Generated Sub-questions ===")
    for idx, q in enumerate(subquestions, 1):
        print(f"{idx}. {q}")

    # Initialize a list to hold RAG chain results
    rag_results = []

    # Answer each sub-question
    for sub_question in subquestions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.invoke(sub_question)

        # Format context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Use retrieved documents and sub-question in RAG chain
        answer = (rag_prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": sub_question}
        )
        rag_results.append(answer)

    return rag_results, subquestions


# ---- STEP 4: RUN ----

question = "What are the main causes of heart disease?"

# Get answers for each sub-question
answers, questions = retrieve_and_rag(question)

# Format Q+A pairs for synthesis
context = format_qa_pairs(questions, answers)

# Synthesize final answer
final_rag_chain = synthesis_prompt | llm | StrOutputParser()

answer = final_rag_chain.invoke({"context": context, "question": question})

print("\n=== Final Answer ===")
print(answer)
