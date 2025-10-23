import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pprint import pprint
import bs4

# ---- Load and index docs ----

docs = WikipediaLoader(query="Chronic pain", lang="en", load_max_docs=3).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(doc_splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3.2")

# ---- Prompt templates ----

grade_doc_prompt = PromptTemplate.from_template(
    """
You are a grader assessing whether retrieved content is relevant to a user question.

Context:
{context}

Question:
{question}

Answer with "yes" or "no" and explain briefly.
"""
)

rag_prompt = PromptTemplate.from_template(
    """
You are a helpful and knowledgeable assistant.

Use the following context to answer the question. If the context is irrelevant or incomplete, use your own knowledge to give the best possible answer.

Context:
{context}

Question:
{question}
"""
)

grade_generation_prompt = PromptTemplate.from_template(
    """
You are evaluating whether the assistant's answer is helpful and appropriate.

Even if the context does not directly support the answer, consider whether the answer still makes sense and is accurate based on the question.

Context:
{context}

Answer:
{generation}

Question:
{question}

Reply with one of:
- "useful"
- "not useful"
- "not supported"

Then briefly explain your reasoning.
"""
)

# ---- LangGraph State ----


class GraphState(dict):
    question: str
    context: str
    generation: str
    attempts: int


# ---- Nodes ----


def retrieve(state):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    print("\nRetrieved context.")
    return {
        "question": state["question"],
        "context": context,
        "attempts": state.get("attempts", 0),
    }


def grade_documents(state):
    prompt = grade_doc_prompt.format(
        context=state["context"], question=state["question"]
    )
    result = llm.invoke(prompt).lower()
    print("\nDocument Grading Result:", result)

    if "yes" in result:
        return {"grade_documents": "generate"}

    # NEW: fallback after 1 bad document attempt
    if state.get("attempts", 0) >= 1:
        print("Context still not helpful. Using LLM to answer directly.")
        fallback_answer = llm.invoke(
            f"Answer this using your own knowledge:\n{state['question']}"
        )
        return {
            "grade_documents": "done",
            "question": state["question"],
            "generation": fallback_answer,
            "context": "No relevant context found.",
            "done": True,
        }

    return {"grade_documents": "transform_query"}


def generate(state):
    if state.get("done"):
        print("\nFinal Fallback Answer (No relevant context found):")
        print(state["generation"])
        return {**state, "grade_generation_result": "useful"}

    prompt = rag_prompt.format(context=state["context"], question=state["question"])
    answer = llm.invoke(prompt)
    print("\n Generated Answer:\n", answer)
    return {
        "generation": answer,
        "context": state["context"],
        "question": state["question"],
        "attempts": state.get("attempts", 0),
    }


def transform_query(state):
    attempt = state.get("attempts", 0) + 1
    print(f"\n Rephrasing query (attempt {attempt})...")

    return {"question": state["question"], "attempts": attempt}


def grade_generation_v_documents_and_question(state):
    if state.get("done"):
        return "useful"

    prompt = grade_generation_prompt.format(
        context=state["context"],
        generation=state["generation"],
        question=state["question"],
    )
    result = llm.invoke(prompt).lower()
    print("\n Self-Reflection Result:", result)

    if "not supported" in result:
        return "not supported"
    elif "not useful" in result:
        return "not useful"
    return "useful"


# ---- Build LangGraph ----

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    lambda x: x["grade_documents"],
    {
        "generate": "generate",
        "transform_query": "transform_query",
        "done": END,  # Exit early with LLM-only answer
    },
)

workflow.add_edge("transform_query", "retrieve")


def condition_after_generate(state):
    if state.get("done"):
        return "useful"
    elif state.get("grade_generation_result") == "useful":
        return "useful"
    else:
        return grade_generation_v_documents_and_question(state)


workflow.add_conditional_edges(
    "generate",
    condition_after_generate,
    {"useful": END, "not useful": "transform_query", "not supported": "generate"},
)

app = workflow.compile()

# ---- Run It ----

question = "What are common treatments for chronic pain?"
inputs = {"question": question}

print("\n=== Running Self-Reflection RAG ===")
for step in app.stream(inputs):
    for node, val in step.items():
        print(f"\n Node: {node}")
        pprint(val)
        print("\n---")
print("\n Final Answer:")
pprint(val.get("generation"))
