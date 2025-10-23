# pip install -U langchain langchain-community langchain-ollama langgraph chromadb beautifulsoup4

import os

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict
from pprint import pprint
import bs4

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from pprint import pprint

"""
Adaptive RAG

The pipeline follows these steps:

Router Decision: 
Based on the user's question, decide whether to retrieve information from the vectorstore or perform a web search.
Retrieve Context:
If the router selects "vectorstore," retrieve relevant documents from the pre-indexed vector database.
If the router selects "web_search," simulate a web search to retrieve external information.
Generate Answer: Use the retrieved context to generate an answer to the user's question.
Output Answer: Return the final answer to the user.

"""

"""
ChainGraph
+------------------+
|   Entry Point    |
|  Route Question  |
+------------------+
         |
   +-----+-----+
   |           |
   v           v
+------------------+       +------------------+
| Retrieve Vector- |       |  Retrieve Web    |
|     store        |       |   (Simulated)    |
+------------------+       +------------------+
         |                       |
         +-----------+-----------+
                     |
                     v
         +-----------------------+
         |    Grade Context      |
         +-----------------------+
                     |
           +---------+---------+
           |                   |
           v                   v
   +----------------+   +----------------+
   |   Generate     |   |    Fallback    |
   |   Answer       |   | (LLM-Only)     |
   +----------------+   +----------------+
           |                   |
           +---------+---------+
                     |
                     v
             +---------------+
             |      END      |
             +---------------+
"""
from langgraph.graph import StateGraph, END

# ---- STEP 1: Load & Split Documents ----

loader = WikipediaLoader(query="Diabetes", lang="en", load_max_docs=3)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(docs)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# ---- STEP 2: Simulated Web Search ----


def fake_web_search(query: str):
    return f"Simulated web result for query: '{query}' — [This would be dynamic info from the web.]"


# ---- STEP 3: LLM and Chains ----

llm = OllamaLLM(model="llama3.2", temperature=0)

router_prompt = ChatPromptTemplate.from_template(
    """
You are a smart router. Choose one of the following based on the user question:
- "vectorstore"
- "web_search"

Question: {question}

Answer with just one word.
"""
)

router_chain = (
    {"question": lambda x: x["question"]} | router_prompt | llm | StrOutputParser()
)

answer_prompt = ChatPromptTemplate.from_template(
    """
Use the following context to answer the user's question.

Context:
{context}

Question:
{question}
"""
)

generate_chain = (
    {"question": lambda x: x["question"], "context": lambda x: x["context"]}
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 4: LangGraph State Definition ----


class AgentState(TypedDict):
    question: str
    context: str
    context_relevant: str
    generation: str


# ---- STEP 5: Node Functions ----


def route_question(state: AgentState):
    route = router_chain.invoke({"question": state["question"]}).strip().lower()
    print("Routing decision:", route)
    return "web_search" if "web" in route else "vectorstore"


def retrieve_vectorstore(state: AgentState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"question": state["question"], "context": context}


def retrieve_web(state: AgentState):
    context = fake_web_search(state["question"])
    return {"question": state["question"], "context": context}


def grade_context_node(state: AgentState):
    context = state["context"]
    question = state["question"]
    prompt = f"""Is the following context relevant to the question?

Question: {question}
Context: {context}

Reply only with yes or no."""
    result = llm.invoke(prompt).strip().lower()
    print("Context Relevance:", result)
    return {**state, "context_relevant": "yes" if "yes" in result else "no"}


def check_relevance_condition(state: AgentState):
    return "generate" if state["context_relevant"] == "yes" else "fallback"


def generate_answer(state: AgentState):
    answer = generate_chain.invoke(
        {"question": state["question"], "context": state["context"]}
    )
    return {
        "question": state["question"],
        "context": state["context"],
        "generation": answer,
    }


def fallback_answer(state: AgentState):
    question = state["question"]
    print("Fallback to LLM-only answer.")
    answer = llm.invoke(f"Answer this using your own knowledge:\n{question}")
    return {
        "question": question,
        "context": "No relevant context found.",
        "generation": answer,
    }


# ---- STEP 6: Build LangGraph ----

graph = StateGraph(AgentState)

graph.add_node("retrieve_vectorstore", retrieve_vectorstore)
graph.add_node("retrieve_web", retrieve_web)
graph.add_node("grade_context", grade_context_node)
graph.add_node("generate", generate_answer)
graph.add_node("fallback", fallback_answer)

graph.set_conditional_entry_point(
    route_question,
    {"vectorstore": "retrieve_vectorstore", "web_search": "retrieve_web"},
)

graph.add_edge("retrieve_vectorstore", "grade_context")
graph.add_edge("retrieve_web", "grade_context")

graph.add_conditional_edges(
    "grade_context",
    check_relevance_condition,
    {"generate": "generate", "fallback": "fallback"},
)

graph.add_edge("generate", END)
graph.add_edge("fallback", END)

app = graph.compile()

# ---- STEP 7: Run the Agent ----

# question = "What are the main symptoms of diabetes?"

question = "What is the largest city in Southern California based on their population?"

print("\n=== Running Adaptive RAG with Fallback ===")
for step in app.stream({"question": question}):
    for node, output in step.items():
        print(f"\n Node: {node}")
        pprint(output)
        print("\n---")

print("\nFinal Answer:")
pprint(output.get("generation"))
