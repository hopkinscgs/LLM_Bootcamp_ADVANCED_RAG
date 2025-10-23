import os
from typing import TypedDict, List
from pprint import pprint
import bs4

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from typing import TypedDict, List
from pprint import pprint

"""
Overview of Agentic RAG
The pipeline follows these steps:
Agent Initialization: 
Extract the user's question from the input.
Retrieve Documents: Retrieve relevant documents for the question.
Grade Documents: Evaluate whether the retrieved documents are relevant to the question.
Rewrite Query: If the documents are not relevant, rewrite the query to make it more specific or clear.
Generate Answer: Use the retrieved documents to generate an answer.
Iterative Improvement: Repeat the process until relevant documents are found or a satisfactory answer is generated.

"""
"""
+------------------+
|   Entry Point    |
|      Agent       |
+------------------+
         |
         v
+------------------+
|    Retrieve      |
|   Documents      |
+------------------+
         |
         v
+------------------+
|   Grade Docs     |
+------------------+
         |
   +-----+-----+
   |           |
   v           v
+-------+   +-------+
|Generate|  |Rewrite|
+-------+   +-------+
   |           |
   |           v
   |     +------------------+
   |     |    Retrieve      |
   |     |   Documents      |
   |     +------------------+
   |           |
   +-----------+
         |
         v
+------------------+
|     Fallback     |
| (LLM Knowledge)  |
+------------------+
         |
         v
+------------------+
|       END        |
+------------------+

"""
# ----- STEP 1: Load, Split & Index Docs -----

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = []
for url in urls:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embedding_model, collection_name="agentic-rag")
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3.2", temperature=0)

# ----- STEP 2: Define Graph State -----

class AgentState(TypedDict):
    messages: List[HumanMessage]
    docs: List[str]
    question: str
    generation: str

# ----- STEP 3: Prompts -----

doc_grader_prompt = PromptTemplate.from_template("""
You are grading whether retrieved context is relevant to the user's question.

Question: {question}
Context:
{context}

Reply with "yes" or "no" and explain briefly.
""")

rewrite_prompt = PromptTemplate.from_template("""
Rewrite the following question to be more specific and clear:

Original question: {question}

Rewritten question:
""")

rag_prompt = PromptTemplate.from_template("""
Use the following context to answer the question.

Context:
{context}

Question: {question}
""")

# ----- STEP 4: Define Graph Nodes -----

def agent(state: AgentState):
    question = state["messages"][0].content
    return {"question": question}

def retrieve(state: AgentState):
    question = state["question"]
    docs = retriever.invoke(question)
    return {"question": question, "docs": docs}

def grade_documents(state: AgentState):
    question = state["question"]
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    result = llm.invoke(doc_grader_prompt.format(question=question, context=context)).lower()

    print("\n Document grading result:", result)
    return "yes" if "yes" in result else "no"

def rewrite(state: AgentState):
    original = state["question"]
    rewritten = llm.invoke(rewrite_prompt.format(question=original))
    print("\n Rewritten question:", rewritten)
    return {"question": rewritten}


def generate(state: AgentState):
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    prompt = rag_prompt.format(context=context, question=state["question"])
    answer = llm.invoke(prompt)
    return {"generation": answer}

def fallback(state: AgentState):
    question = state["question"]
    print("⚠️ Using LLM knowledge to answer directly (no relevant context).")
    answer = llm.invoke(f"Answer this using your own knowledge:\n{question}")
    return {"generation": answer}

# ----- STEP 5: Build LangGraph -----

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("rewrite", rewrite)
graph.add_node("generate", generate)
graph.add_node("fallback", fallback)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", lambda _: "continue", {
    "continue": "retrieve"
})

graph.add_conditional_edges("retrieve", grade_documents, {
    "yes": "generate",
    "no": "rewrite",
    "fallback": "fallback"
})

graph.add_conditional_edges("rewrite", lambda _: "continue", {
    "continue": "retrieve"
})

graph.add_edge("generate", END)
graph.add_edge("fallback", END)

app = graph.compile()

# ----- STEP 6: Run the Agent -----

inputs = {
    "messages": [
        # HumanMessage(content="What does Lilian Weng say about the types of agent memory?")
        HumanMessage(content="which city is the biggest in southern California")
    ]
}

print("\n=== Running Agentic RAG ===")

for step in app.stream(inputs):
    for key, value in step.items():
        print(f"\n Node: {key}")
        pprint(value)
        print("\n---")

print("\n Final Answer:")
print(value.get("generation"))
