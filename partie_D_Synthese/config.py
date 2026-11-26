import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent

ROOT_DIR = BASE_DIR.parent

load_dotenv(ROOT_DIR / ".env")


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

MCP_SAV_URL = os.getenv(
    "MCP_SAV_URL",
    "http://127.0.0.1:8001/mcp"
)

DOC_DIR = BASE_DIR / "documents"
IMG_DIR = BASE_DIR / "images"
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "sav_multimodal"


def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY absente.")

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0
    )


def get_embeddings():

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def get_vectorstore():

    return Chroma(

        collection_name=COLLECTION_NAME,

        embedding_function=get_embeddings(),

        persist_directory=str(DB_DIR),

        collection_metadata={
            "hnsw:space": "cosine"
        }
    )
