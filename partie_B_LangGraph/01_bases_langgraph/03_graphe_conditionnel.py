
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

load_dotenv(override=True)

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0
)

# 1. Définir le State
class State(TypedDict):
    question: str
    categorie: str
    reponse: str


# 2. Nœud de classification
def classifier(state: State):
    resultat = llm.invoke([
        ("system",
         "Classe la question dans une seule catégorie : "
         "python, reseaux ou general. "
         "Retourne uniquement le nom de la catégorie."),
        ("human", state["question"])
    ])

    categorie = str(resultat.content).strip().lower()

    if categorie not in ["python", "reseaux", "general"]:
        categorie = "general"

    print("[CLASSIFICATION]", categorie)

    return {"categorie": categorie}


# 3. Fonction de routage
def router(state: State) -> Literal[
    "python", "reseaux", "general"
]:
    return state["categorie"]


# 4. Nœud expert Python
def expert_python(state: State):
    print("[NODE] Expert Python")

    resultat = llm.invoke([
        ("system", "Tu es un expert Python. "
                   "Réponds en français avec du code."),
        ("human", state["question"])
    ])

    return {"reponse": resultat.content}


# 5. Nœud expert Réseaux
def expert_reseaux(state: State):
    print("[NODE] Expert Réseaux")

    resultat = llm.invoke([
        ("system", "Tu es un ingénieur réseaux. "
                   "Réponds clairement en français."),
        ("human", state["question"])
    ])

    return {"reponse": resultat.content}


# 6. Nœud généraliste
def assistant_general(state: State):
    print("[NODE] Assistant général")

    resultat = llm.invoke([
        ("system", "Tu es un assistant généraliste. "
                   "Réponds en français."),
        ("human", state["question"])
    ])

    return {"reponse": resultat.content}


# 7. Créer le graphe
graph = StateGraph(State)

# 8. Ajouter les nœuds
graph.add_node("classifier", classifier)
graph.add_node("expert_python", expert_python)
graph.add_node("expert_reseaux", expert_reseaux)
graph.add_node("assistant_general", assistant_general)

# 9. Point de départ
graph.add_edge(START, "classifier")

# 10. Transitions conditionnelles
graph.add_conditional_edges(
    "classifier",
    router,
    {
        "python": "expert_python",
        "reseaux": "expert_reseaux",
        "general": "assistant_general"
    }
)

# 11. Points de terminaison
graph.add_edge("expert_python", END)
graph.add_edge("expert_reseaux", END)
graph.add_edge("assistant_general", END)

# 12. Compiler
app = graph.compile()

# 13. Interface utilisateur
print("=== LANGGRAPH ROUTAGE CONDITIONNEL ===")

while True:
    question = input("\nVous : ").strip()

    if question.lower() in ["quit", "exit"]:
        break

    if not question:
        continue

    try:
        resultat = app.invoke({
            "question": question,
            "categorie": "",
            "reponse": ""
        })

        print("Catégorie :", resultat["categorie"])
        print("Agent :", resultat["reponse"])

    except Exception as e:
        print("Erreur :", e)
