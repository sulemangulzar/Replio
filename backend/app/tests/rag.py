from pathlib import Path
from app.rag.ingestion import load_documents, splitting_docs
from app.rag.vector_store import add_documents


DOC_PATH = Path("app/rag/openai.txt")
docs = load_documents(DOC_PATH)
chunks = splitting_docs(docs)

store = add_documents(chunks)
print(len(chunks))
