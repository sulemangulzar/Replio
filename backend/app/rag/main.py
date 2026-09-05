from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIEmbeddings()

def load_documents(file: str | Path):
    file_path = Path(file)

    ext = file_path.suffix.lower()

    str_path = str(file_path)

    if ext == ".pdf":
        loader = PyMuPDFLoader(str_path)
    elif ext == ".txt":
        loader = TextLoader(str_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(str_path)
    else:
        return "Not Supported Documents..."

    return loader.lazy_load()

def splitting_docs(documents : list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", ",", ""]
    )

    return splitter.split_documents(documents)


docs = load_documents("openai.txt")

all_docs = list(docs)

splitting = splitting_docs(all_docs)

# if splitting:
#     for index ,chunk in enumerate(splitting, start=1):
#         print("=" * 40)
#         print(f"Chunk: {index}")
#         print(chunk.page_content)
#         print("=" * 40)
