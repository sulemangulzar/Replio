from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from langchain_pinecone import PineconeVectorStore

from app.config import settings


_embeddings = None
_pinecone = None
_vectorstore = None


def create_embeddings() -> OpenAIEmbeddings:
    global _embeddings

    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.openai_api_key),
        )

    return _embeddings


def _client() -> Pinecone:
    global _pinecone

    if _pinecone is None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")

        _pinecone = Pinecone(api_key=settings.pinecone_api_key)

    return _pinecone


def ensure_index():
    client = _client()
    index_name = settings.pinecone_index_name

    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return client.Index(index_name)


def vector_store() -> PineconeVectorStore:
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
            embedding=create_embeddings(),
            namespace=settings.pinecone_namespace
        )

    return _vectorstore

def get_retriever():
    top_k = getattr(settings, "top_k", 4)
    return vector_store().as_retriever(search_kwargs={"k": top_k})


def add_documents(chunks):
    store = vector_store()
    return store.add_documents(chunks)
