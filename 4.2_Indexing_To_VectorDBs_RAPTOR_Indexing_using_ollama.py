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
from concurrent.futures import ThreadPoolExecutor
import time

# Step 1: Load and split documents
print("Step 1: Loading and splitting documents...")
loader = WikipediaLoader(query="Asthma", lang="en", load_max_docs=3)
docs = loader.load()
print(f"Loaded {len(docs)} documents")


# Step 2: Create hierarchical text splitter with larger chunks
def create_hierarchical_splits(
    docs: List[Document],
    chunk_sizes: List[int] = [2000, 1000],  # Increased chunk sizes
    chunk_overlaps: List[int] = [200, 100],
) -> Dict[str, List[Document]]:
    splits = {}
    for size, overlap in zip(chunk_sizes, chunk_overlaps):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
        )
        splits[f"level_{size}"] = splitter.split_documents(docs)
    return splits


hierarchical_splits = create_hierarchical_splits(docs)
print(f"Created hierarchical splits with {len(hierarchical_splits)} levels")

# Step 3: Create abstractive summaries for each level
print("\nStep 3: Creating abstractive summaries...")
llm = OllamaLLM(model="llama3.2")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Ultra-concise prompt for faster summarization
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the single most important point from this text in one sentence.",
        ),
        ("user", "{text}"),
    ]
)


def summarize_doc(doc: Document, level: str) -> Document:
    try:
        # Only summarize if the text is long enough
        if len(doc.page_content.split()) > 100:  # Increased threshold
            summary = llm.invoke(summary_prompt.format_messages(text=doc.page_content))
            return Document(
                page_content=summary,
                metadata={
                    **doc.metadata,
                    "original_content": doc.page_content,
                    "level": level,
                },
            )
        else:
            # For short texts, use them as is
            return Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "original_content": doc.page_content,
                    "level": level,
                },
            )
    except Exception as e:
        print(f"Error summarizing document: {e}")
        return doc


def create_abstractive_summaries(
    splits: Dict[str, List[Document]],
) -> Dict[str, List[Document]]:
    summarized_splits = {}
    total_docs = sum(len(docs) for docs in splits.values())
    processed_docs = 0
    start_time = time.time()

    for level, docs in splits.items():
        print(f"\nProcessing {level} ({len(docs)} documents)...")
        # Process in larger batches
        batch_size = 10  # Increased batch size
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=5) as executor:  # Increased workers
                summarized_docs = list(
                    executor.map(lambda doc: summarize_doc(doc, level), batch)
                )
            if level not in summarized_splits:
                summarized_splits[level] = []
            summarized_splits[level].extend(summarized_docs)

            processed_docs += len(batch)
            elapsed_time = time.time() - start_time
            avg_time_per_doc = elapsed_time / processed_docs
            remaining_docs = total_docs - processed_docs
            estimated_time = remaining_docs * avg_time_per_doc

            print(f"Progress: {processed_docs}/{total_docs} documents")
            print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")

    return summarized_splits


summarized_splits = create_abstractive_summaries(hierarchical_splits)
print("Summarization complete")

# Step 4: Create vector stores for each level
print("\nStep 4: Creating vector stores...")
vector_stores = {}
for level, docs in summarized_splits.items():
    print(f"Creating vector store for {level}...")
    vector_stores[level] = Chroma.from_documents(
        documents=docs, embedding=embedding_model, collection_name=f"raptor_{level}"
    )
print(f"Created {len(vector_stores)} vector stores")

def dedup(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on their original content.
    """
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.metadata["original_content"] not in seen:
            seen.add(doc.metadata["original_content"])
            unique_docs.append(doc)
    return unique_docs

# Step 5: Create hierarchical retrieval function
def hierarchical_retrieval(query: str, k: int = 3) -> List[Document]:
    all_docs = []

    # Start with the most abstract level
    for level in sorted(vector_stores.keys(), reverse=True):
        docs = vector_stores[level].similarity_search(query, k=k)
        all_docs.extend(docs)

        # Deduplicate documents
        all_docs = dedup(all_docs)

        # If we have enough high-level context, break
        if len(all_docs) >= k:
            break

    print("Selected docs:", )
    return all_docs


# Step 6: Create the full chain
template = """Answer the question based on the following context. \
The context is organized in a hierarchical structure, with different levels of abstraction. \
Use the most relevant information from each level to provide a comprehensive answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that provides comprehensive answers based on hierarchical context.",
        ),
        ("user", template),
    ]
)

chain = (
    {
        "context": RunnableLambda(
            lambda x: "\n\n".join(
                [doc.page_content for doc in hierarchical_retrieval(x["question"])]
            )
        ),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Step 7: Test the implementation
print("\nStep 7: Testing the implementation...")
question = "What triggers asthma attacks?"

print("\n=== User Question ===")
print(question)

response = chain.invoke({"question": question})

print("\n=== Final Answer ===")
print(response)
print("-" * 80)
