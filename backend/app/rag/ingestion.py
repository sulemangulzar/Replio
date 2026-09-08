from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Iterable

def load_documents(file: str | Path):
    file_path = Path(file)

    ext = file_path.suffix.lower()

    str_path = str(file_path)

    if ext == ".pdf":
        loader = PyMuPDFLoader(str_path).load()
    elif ext in {".md", ".txt"}:
        loader = TextLoader(str_path).load()
    elif ext == ".docx":
        loader = Docx2txtLoader(str_path).load()
    else:
        return "Not Supported Documents..."

    return loader

def splitting_docs(documents : Iterable[Document]) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", ",", ""]
    )

    return splitter.split_documents(list(documents))
