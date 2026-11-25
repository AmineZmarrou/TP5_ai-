
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Définir l'état
class State(TypedDict):
    question: str
    reponse: str


# 2. Créer un nœud
def traiter_question(state: State):
    print("[NODE] Traitement de la question")

    return {
        "reponse": "Bonjour " + state["question"]
    }


# 3. Créer le graphe
graph = StateGraph(State)

# 4. Ajouter le nœud
graph.add_node("traitement", traiter_question)

# 5. Ajouter les transitions
graph.add_edge(START, "traitement")
graph.add_edge("traitement", END)

# 6. Compiler le graphe
app = graph.compile()

# 7. Exécuter le graphe
resultat = app.invoke({
    "question": "tout le monde",
    "reponse": ""
})

print(resultat)
