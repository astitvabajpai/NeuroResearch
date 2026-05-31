from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.tools.hg_llm import get_embeddings
from pathlib import Path

def build_vector_store(data_dir: str = "data/sample_papers"):
    embeddings = get_embeddings()
    docs = []
    
    for file in Path(data_dir).glob("*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file))
        else:
            continue
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    return FAISS.from_documents(chunks, embeddings)

def get_retriever(data_dir: str = "data/sample_papers"):
    return build_vector_store(data_dir).as_retriever(search_kwargs={"k": 3})