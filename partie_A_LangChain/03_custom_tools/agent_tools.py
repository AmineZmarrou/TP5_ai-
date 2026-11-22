
import os
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

# 1. Charger les variables d'environnement
load_dotenv(find_dotenv())

# 2. Initialiser le modèle Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 3. Créer les tools personnalisés

@tool
def additionner(a: float, b: float) -> float:
    """Additionne deux nombres et retourne leur somme."""
    print(f"[TOOL] additionner({a}, {b})")
    return a + b


@tool
def multiplier(a: float, b: float) -> float:
    """Multiplie deux nombres et retourne leur produit."""
    print(f"[TOOL] multiplier({a}, {b})")
    return a * b


@tool
def rechercher_etudiant(nom: str) -> str:
    """Recherche un étudiant par son nom dans la base locale."""
    print(f"[TOOL] rechercher_etudiant({nom})")

    etudiants = {
        "amine": "Amine : spécialité Intelligence Artificielle",
        "idriss": "Idriss : spécialité Développement Web",
        "mouad": "Mouad : spécialité Cybersécurité"
    }

    return etudiants.get(
        nom.strip().lower(),
        "Étudiant introuvable dans la base."
    )


# 4. Regrouper les tools
tools = [
    additionner,
    multiplier,
    rechercher_etudiant
]

# 5. Créer la mémoire
memoire = InMemorySaver()

# 6. Créer l'agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "Tu es un assistant intelligent spécialisé "
        "en informatique et intelligence artificielle. "
        "Réponds clairement en français. "
        "Utilise les outils disponibles pour effectuer "
        "les calculs et rechercher des étudiants. "
        "N'invente pas de résultats."
    ),
    checkpointer=memoire
)

# 7. Configuration de la conversation
config = {
    "configurable": {
        "thread_id": "conversation_tools_1"
    }
}

# 8. Fonction de discussion
def discuter(question):
    resultat = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        config=config
    )

    print("\nAgent :", resultat["messages"][-1].content)
    print("-" * 50)


# 9. Interface interactive
print("=== Agent LangChain + Groq + Tools ===")
print("Tape 'quit' pour quitter.\n")

while True:
    question = input("Vous : ").strip()

    if question.lower() in ["quit", "exit"]:
        print("Fin de la conversation.")
        break

    if not question:
        continue

    try:
        discuter(question)
    except Exception as e:
        print("Erreur :", e)
