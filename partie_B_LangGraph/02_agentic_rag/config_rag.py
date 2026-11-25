
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Racine du projet TP5
ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env", override=True)

# Base vectorielle de la partie A
DB_PATH = (
    ROOT
    / "partie_A_LangChain"
    / "06_agentic_rag"
    / "chroma_db"
)

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Base vectorielle introuvable : {DB_PATH}"
    )

# Modèle Groq
llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    ),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Embeddings Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# Charger la base vectorielle existante
vectorstore = Chroma(
    collection_name="tp5_documents",
    embedding_function=embeddings,
    persist_directory=str(DB_PATH)
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

MAX_RETRIES = 2
