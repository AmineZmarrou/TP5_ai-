
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
DOSSIER_PDF = BASE_DIR / "documents"
DOSSIER_DB = BASE_DIR / "chroma_db"
COLLECTION = "tp5_documents"

MODELE_EMBEDDING = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 1. Vérifier le dossier des PDF
DOSSIER_PDF.mkdir(exist_ok=True)

fichiers_pdf = sorted(DOSSIER_PDF.glob("*.pdf"))

if not fichiers_pdf:
    raise SystemExit(
        "Aucun PDF trouvé dans le dossier documents/"
    )

# 2. Charger les documents
documents = []

for fichier in fichiers_pdf:
    print(f"[PDF] Chargement : {fichier.name}")

    loader = PyPDFLoader(str(fichier))
    pages = loader.load()

    for page in pages:
        page.metadata["source"] = fichier.name
        page.metadata["page"] = page.metadata.get("page", 0)

    documents.extend(pages)

print(f"[INFO] {len(documents)} pages chargées")

# 3. Découper les documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

# Retirer les morceaux vides
chunks = [
    chunk for chunk in chunks
    if chunk.page_content.strip()
]

if not chunks:
    raise SystemExit(
        "Aucun texte exploitable trouvé dans les PDF."
    )

print(f"[INFO] {len(chunks)} chunks créés")

# 4. Initialiser les embeddings
print("[INFO] Chargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODELE_EMBEDDING,
    encode_kwargs={"normalize_embeddings": True}
)

# 5. Reconstruire la base pour éviter les doublons
if DOSSIER_DB.exists():
    shutil.rmtree(DOSSIER_DB)

# 6. Créer la base ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=str(DOSSIER_DB)
)

# 7. Indexer les chunks
vectorstore.add_documents(chunks)

print("\n=== INDEXATION TERMINÉE ===")
print("PDF :", len(fichiers_pdf))
print("Pages :", len(documents))
print("Chunks :", len(chunks))
print("Base vectorielle :", DOSSIER_DB.resolve())
