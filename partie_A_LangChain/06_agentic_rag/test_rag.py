
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
DOSSIER_DB = BASE_DIR / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    collection_name="tp5_documents",
    embedding_function=embeddings,
    persist_directory=str(DOSSIER_DB)
)

question = "Qu'est-ce qu'un agent LangChain ?"

resultats = vectorstore.similarity_search(
    question,
    k=3
)

print("Question :", question)
print("Nombre de résultats :", len(resultats))

for i, doc in enumerate(resultats, start=1):
    source = doc.metadata.get("source", "Inconnue")
    page = doc.metadata.get("page", 0) + 1

    print(f"\n=== Résultat {i} ===")
    print("Source :", source)
    print("Page :", page)
    print("Contenu :", doc.page_content[:600])
