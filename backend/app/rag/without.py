from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="mistralai/mistral-7b-instruct-v0.3",
    temperature=0.7
)

def load_docs(path: str):
    reader = PdfReader(path)
    all_page_texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:  # Ensure the page isn't empty/scanned image
            fixed = text.split()
            all_page_texts.append(" ".join(fixed))

    # FIXED: Combine all pages into one continuous string instead of returning on page 1
    return " ".join(all_page_texts)

def chunks(document, chunk_size=800, overlap=200):
    chunks_list = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunks_list.append(document[start:end])
        start += chunk_size - overlap
    return chunks_list

def create_embeddings(docs: list):
    embeddings = embeddings_model.encode(docs)
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# FIXED: We now pass the original raw text chunks into the retriever alongside the embeddings
def retriever(embeddings, text_chunks, query):
    results = []
    qembed = embeddings_model.encode(query)

    for i, doc_embedding in enumerate(embeddings):
         # FIXED: Compare the single document embedding against the query embedding
         score = cosine_similarity(doc_embedding, qembed)

         # FIXED: Map the score to the matching textual chunk string
         results.append((score, text_chunks[i]))

    # Sort by the score (which is index 0 of the tuple) in descending order
    results.sort(key=lambda x: x[0], reverse=True)

    # Return the highest scoring tuple (score, text)
    return results[0]

# Execution Pipeline
text = load_docs("Projects-AI_Eng.pdf")
chunnks = chunks(text)
embeddings = create_embeddings(chunnks)
question =  "What projects are in?"
best_score, best_match_text = retriever(embeddings, chunnks, question)

print(f"Similarity Score: {best_score:.4f}")
print(f"Best Matching Chunk:\n{best_match_text}")

def ask_llm(question, context):
    system_prompt = f"You are a intellegent system your goal is to provide the information you know don't hallucinate the context is : {context}"
    messages = [
        ("system", system_prompt),
        ("user", question)
    ]
    response = client.invoke(messages)
    print(response.content)

ask_llm(question, best_match_text)
