# RAG Data Preprocessing & Embedding Pipeline

This project provides a robust pipeline for preparing and embedding text data for Retrieval-Augmented Generation (RAG) and other NLP tasks. It includes data cleanup, chunking, embedding using the `nomic-embed-text` model from Ollama, and advanced retrieval mechanisms.

---

## Features
- **Data Cleanup:** Remove HTML, normalize text, strip unwanted characters, and more.
- **Chunking:** Split large documents into overlapping chunks for better context retention.
- **Embedding:** Generate high-quality vector embeddings using Ollama's `nomic-embed-text` model.
- **Similarity Search:** Find the most relevant text chunks for a given query.
- **Advanced RAG Techniques:** Includes query transformations, routing, indexing, agentic RAG, and more.

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/IDEAS-Incubator/LLM_Bootcamp_ADVANCED_RAG
   ```
2. **Create conda env**
   ```bash
   conda create -n adrag python=3.12
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your environment**
   ```bash
   cp .env.example .env
   ```

---

## Usage

The main scripts are located in the project root. Each script covers a specific aspect of the RAG pipeline or advanced RAG techniques:

| Script | Description |
|--------|-------------|
| `1.1_Introduction_To_RAG_using_ollama.py` | Basic RAG introduction and workflow |
| `2.1_...` to `2.5_...` | Query transformation techniques (MultiQuery, RAG Fusion, Decomposition, StepBack, HyDE) |
| `3.1_...` to `3.3_...` | Routing to data sources (base, semantic, multi-source) |
| `4.1_...` to `4.3_...` | Indexing to vector DBs (multi-representation, RAPTOR, ColBERT) |
| `5.1_Retrieval_Mechanisms_using_ollama.py` | Retrieval mechanisms |
| `6.1_Self_Reflection_RAG_using_ollama.py` | Self-reflection in RAG |
| `7.1_Agentic_Rag_using_ollama.py` | Agentic RAG |
| `8.1_Adaptive_Rag_Agent_using_ollama.py` | Adaptive RAG agent |
| `9.1_Corrective_Rag_Agent_using_ollama.py` | Corrective RAG agent |
| `10.1_LLAMA_3_Rag_Agent_Local_using_ollama.py` | LLAMA 3 RAG agent (local) |

---

### Example: Generating Embeddings

Use the following code to generate embeddings for a text chunk using Ollama:
```python
import ollama
response = ollama.embeddings(model='nomic-embed-text', prompt="your text chunk here")
embedding = response['embedding']
```

---

### Example: Similarity Search
Use cosine similarity (e.g., with scikit-learn) to compare query embeddings to your chunk embeddings.

---

## Requirements
See `requirements.txt` for all dependencies.

---
