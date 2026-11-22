
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver

# ==============================
# 1. CONFIGURATION
# ==============================

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
DOSSIER_DB = BASE_DIR / "chroma_db"

if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY manquante dans .env")

if not DOSSIER_DB.exists():
    raise SystemExit(
        "Base ChromaDB introuvable. "
        "Exécute d'abord : python ingestion.py"
    )

# ==============================
# 2. INITIALISER GROQ
# ==============================

llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    ),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ==============================
# 3. INITIALISER LES EMBEDDINGS
# ==============================

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    encode_kwargs={"normalize_embeddings": True}
)

# ==============================
# 4. CHARGER CHROMADB
# ==============================

vectorstore = Chroma(
    collection_name="tp5_documents",
    embedding_function=embeddings,
    persist_directory=str(DOSSIER_DB)
)

# ==============================
# 5. CRÉER L'OUTIL RAG
# ==============================

@tool
def rechercher_documents(question: str) -> str:
    """
    Recherche des informations dans les PDF indexés.
    Utiliser cet outil pour toute question portant sur
    les cours, documents ou fichiers PDF de l'utilisateur.
    Retourne les passages avec leurs sources et pages.
    """

    print(f"\n[TOOL RAG] Recherche : {question}")

    resultats = vectorstore.similarity_search(
        question,
        k=4
    )

    if not resultats:
        return "Aucun passage trouvé dans les documents."

    passages = []

    for i, doc in enumerate(resultats, start=1):
        source = doc.metadata.get(
            "source", "Document inconnu"
        )

        page = doc.metadata.get("page", 0) + 1

        passages.append(
            f"[Passage {i}]\n"
            f"Source : {source}\n"
            f"Page : {page}\n"
            f"Contenu : {doc.page_content}"
        )

    return "\n\n".join(passages)


# ==============================
# 6. CRÉER LA MÉMOIRE
# ==============================

memoire = InMemorySaver()

# ==============================
# 7. CRÉER L'AGENT
# ==============================

agent = create_agent(
    model=llm,
    tools=[rechercher_documents],
    checkpointer=memoire,
    system_prompt=(
        "Tu es un assistant pédagogique spécialisé "
        "en informatique et intelligence artificielle. "

        "Tu disposes d'un outil rechercher_documents "
        "qui permet de consulter les PDF de l'utilisateur. "

        "Pour toute question concernant les documents, "
        "tu dois utiliser cet outil avant de répondre. "

        "Réponds uniquement à partir des passages "
        "retrouvés pour les questions documentaires. "

        "Cite les sources avec le nom du PDF "
        "et le numéro de page. "

        "Si les documents ne permettent pas de "
        "répondre, indique clairement que "
        "l'information n'a pas été trouvée. "

        "Pour les questions générales, tu peux "
        "répondre sans utiliser l'outil. "

        "Réponds toujours en français."
    )
)

# ==============================
# 8. CONFIGURATION MÉMOIRE
# ==============================

config = {
    "configurable": {
        "thread_id": "agentic_rag_tp5"
    }
}

# ==============================
# 9. CHATBOT INTERACTIF
# ==============================

print("=" * 55)
print(" CHATBOT AGENTIC RAG - LANGCHAIN + GROQ")
print("=" * 55)
print("Modèle :", os.getenv(
    "GROQ_MODEL", "openai/gpt-oss-120b"
))
print("Documents : ChromaDB")
print("Tape 'quit' pour quitter.\n")

while True:

    question = input("\nVous : ").strip()

    if question.lower() in ["quit", "exit"]:
        print("Fin de la conversation.")
        break

    if not question:
        continue

    try:
        resultat = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config=config
        )

        # Afficher les outils réellement utilisés
        for message in resultat["messages"]:
            if getattr(message, "tool_calls", None):
                for appel in message.tool_calls:
                    print(
                        "[AGENT] Outil utilisé :",
                        appel["name"]
                    )

        # Afficher la réponse finale
        reponse = resultat["messages"][-1].content

        print("\nAgent :", reponse)
        print("-" * 55)

    except Exception as e:
        print(
            "Erreur :",
            type(e).__name__,
            str(e)
        )
