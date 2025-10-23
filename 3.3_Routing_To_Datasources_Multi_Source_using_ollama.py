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
physics_template = """You are a very smart physics professor. 
You are great at answering questions about physics in a concise and easy-to-understand manner. 
If the question is not directly about physics, use your general knowledge to help answer it as best you can."""


math_template = """You are a very good mathematician. 
You are great at answering math questions. You are so good because you are able to break down hard problems into parts, 
solve the parts, and combine them for the final answer. 
If the question is not directly about math, use your reasoning and general knowledge to provide an answer."""


general_template = """You are a knowledgeable generalist who can provide information on a wide range of topics.

Break down the problem, solve the parts if needed, and then provide a complete answer.

Question:
{query}"""

prompt_templates = [general_template, physics_template, math_template]

# Step 2: Embed the prompts
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
prompt_embeddings = embedding_model.embed_documents(prompt_templates)


# Step 3: Create router logic for multiple sources
def prompt_router(input):
    query_embedding = embedding_model.embed_query(input["query"])
    similarity_scores = cosine_similarity([query_embedding], prompt_embeddings)[0]

    # Get all prompts with similarity above threshold
    threshold = 0.3
    selected_indices = np.where(similarity_scores >= threshold)[0]

    if len(selected_indices) == 0:
        # If no prompts meet the threshold, use the best one
        best_index = np.argmax(similarity_scores)
        selected_indices = [best_index]

    print("\n=== Prompt Routing Decision ===")
    selected_prompts = []
    for idx in selected_indices:
        if idx == 0:
            print(f"Using PHYSICS prompt (Confidence: {similarity_scores[idx]:.2f})")
        else:
            print(f"Using MATH prompt (Confidence: {similarity_scores[idx]:.2f})")
        selected_prompts.append(prompt_templates[idx])

    # Create a combined prompt
    combined_prompt = """You are an expert in both physics and mathematics.

    Consider each of the following expert perspectives, and answer the question using relevant knowledge from those fields. 
    If the question isn’t specific to physics or math, use your general knowledge to provide a complete and thoughtful response.

    Perspectives:
    {perspectives}

    Question:
    {query}
    """

    # Format the perspectives
    perspectives = "\n\n".join(
        [f"Perspective {i+1}:\n{prompt}" for i, prompt in enumerate(selected_prompts)]
    )

    return PromptTemplate.from_template(combined_prompt).format(
        perspectives=perspectives, query=input["query"]
    )


# Step 4: Create the full chain
llm = OllamaLLM(model="llama3.2")

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

# Step 5: Ask something!
questions = [
    "What is the speed of light?",
    "What is the derivative of x^2?",
    "How does quantum entanglement work?",
    "What is the Pythagorean theorem?",
    "What is the relationship between energy and mass?",  # This should use both perspectives
]

print("\n=== Testing Multi-Source Router with Different Questions ===")
for question in questions:
    print("\n=== User Question ===")
    print(question)

    response = chain.invoke(question)

    print("\n=== Final Answer ===")
    print(response)
    print("-" * 80)
