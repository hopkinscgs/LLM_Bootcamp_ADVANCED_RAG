import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Define your prompts
physics_template = """You are a brilliant physics professor known for making complex physics topics easy to understand.

Answer the following question clearly and accurately. If you are unsure or the question isn't related to physics, say so honestly.

Question:
{query}"""


math_template = """You are a skilled mathematician who explains math problems step-by-step.

Break down the problem, solve the parts if needed, and then provide a complete answer. If the question isn't math-related, it's okay to say you don't know.

Question:
{query}"""

general_template = """You are a knowledgeable generalist who can provide information on a wide range of topics.

Break down the problem, solve the parts if needed, and then provide a complete answer.

Question:
{query}"""

prompt_templates = [general_template, physics_template, math_template]

# Step 2: Embed the prompts
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
prompt_embeddings = embedding_model.embed_documents(prompt_templates)


# Step 3: Create router logic
def prompt_router(input):
    query_embedding = embedding_model.embed_query(input["query"])
    similarity_scores = cosine_similarity([query_embedding], prompt_embeddings)[0]
    best_index = np.argmax(similarity_scores)
    chosen_prompt = prompt_templates[best_index]

    print("\n=== Prompt Routing Decision ===")
    # need to update the code
    # print("Using MATH prompt" if best_index == 1 else "Using PHYSICS prompt")
    return PromptTemplate.from_template(chosen_prompt)


# Step 4: Create the full chain
llm = OllamaLLM(model="llama3.2")

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

# Step 5: Ask something!
question = "What the black hole ?"
print("\n=== User Question ===")
print(question)

response = chain.invoke(question)

print("\n=== Final Answer ===")
print(response)
