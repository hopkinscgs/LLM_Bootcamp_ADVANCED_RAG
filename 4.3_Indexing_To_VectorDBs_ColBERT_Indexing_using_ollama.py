import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
import bs4
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and split documents
print("Step 1: Loading and splitting documents...")
loader = WikipediaLoader(query="Tuberculosis", lang="en", load_max_docs=3)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
splits = text_splitter.split_documents(docs)
print(f"Split documents into {len(splits)} chunks")

# Step 2: Create token-level embeddings
print("\nStep 2: Creating token-level embeddings...")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


def create_token_embeddings(text: str) -> List[np.ndarray]:
    # Split text into tokens (words)
    tokens = text.split()
    # Get embeddings for each token
    token_embeddings = []
    for token in tokens:
        embedding = embedding_model.embed_query(token)
        token_embeddings.append(np.array(embedding))
    return token_embeddings


# Step 3: Create document representations
def create_document_representations(
    docs: List[Document],
) -> Dict[str, List[np.ndarray]]:
    doc_representations = {}
    for i, doc in enumerate(docs):
        print(f"Processing document {i+1}/{len(docs)}...")
        doc_representations[f"doc_{i}"] = create_token_embeddings(doc.page_content)
    return doc_representations


doc_representations = create_document_representations(splits)
print("Token embeddings complete")


# Step 4: Create ColBERT-style retrieval function
def colbert_retrieval(query: str, k: int = 3) -> List[Document]:
    # Get query token embeddings
    query_tokens = query.split()
    query_embeddings = [
        np.array(embedding_model.embed_query(token)) for token in query_tokens
    ]

    # Calculate max similarity scores for each document
    doc_scores = []
    for doc_id, doc_tokens in doc_representations.items():
        # Calculate token-level similarities
        token_similarities = []
        for q_emb in query_embeddings:
            max_sim = max(
                cosine_similarity([q_emb], [d_emb])[0][0] for d_emb in doc_tokens
            )
            token_similarities.append(max_sim)

        # Calculate document score (sum of max similarities)
        doc_score = sum(token_similarities)
        doc_scores.append((doc_id, doc_score))

    # Sort documents by score and get top k
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, _ in doc_scores[:k]]

    # Get the actual documents
    retrieved_docs = [splits[int(doc_id.split("_")[1])] for doc_id in top_doc_ids]

    return retrieved_docs


# Step 5: Create the full chain
print("\nStep 5: Creating the full chain...")
llm = OllamaLLM(model="llama3.2")

template = """Answer the question based on the following context. \
The context has been retrieved using a token-level matching approach. \
Use the most relevant information to provide a comprehensive answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that provides comprehensive answers based on token-level matched context.",
        ),
        ("user", template),
    ]
)

chain = (
    {
        "context": RunnableLambda(
            lambda x: "\n\n".join(
                [doc.page_content for doc in colbert_retrieval(x["question"])]
            )
        ),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6: Test the implementation
print("\nStep 6: Testing the implementation...")
questions = [
    "What are the key components of an autonomous agent?",
    "How does memory work in autonomous agents?",
    "What is the role of planning in autonomous agents?",
]

for question in questions:
    print("\n=== User Question ===")
    print(question)

    response = chain.invoke({"question": question})

    print("\n=== Final Answer ===")
    print(response)
    print("-" * 80)
