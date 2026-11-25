
import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

load_dotenv(override=True)

# 1. Initialiser Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 2. Définir l'état
class State(TypedDict):
    question: str
    reponse: str


# 3. Créer le nœud chatbot
def chatbot(state: State):
    print("[NODE] Appel du modèle Groq")

    resultat = llm.invoke([
        ("system", "Tu es un assistant informatique. "
                   "Réponds clairement en français."),
        ("human", state["question"])
    ])

    return {
        "reponse": resultat.content
    }


# 4. Construire le graphe
graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 5. Compiler
app = graph.compile()

# 6. Interface interactive
print("=== CHATBOT LANGGRAPH + GROQ ===")

while True:
    question = input("\nVous : ").strip()

    if question.lower() in ["quit", "exit"]:
        break

    if not question:
        continue

    try:
        resultat = app.invoke({
            "question": question,
            "reponse": ""
        })

        print("Agent :", resultat["reponse"])

    except Exception as e:
        print("Erreur :", e)
