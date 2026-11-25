
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    reponse: str


def traitement(state: State):
    return {"reponse": "Question traitée"}


graph = StateGraph(State)

graph.add_node("traitement", traitement)

graph.add_edge(START, "traitement")
graph.add_edge("traitement", END)

app = graph.compile()

# Afficher le code Mermaid
print(app.get_graph().draw_mermaid())

# Générer une image PNG
try:
    image = app.get_graph().draw_mermaid_png()

    with open("graphe.png", "wb") as fichier:
        fichier.write(image)

    print("Image graphe.png générée.")

except Exception as e:
    print("Erreur de génération PNG :", e)
