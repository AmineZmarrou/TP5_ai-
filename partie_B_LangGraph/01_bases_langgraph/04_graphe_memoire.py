
import os
from dotenv import load_dotenv

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)

from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(override=True)

# 1. Modèle Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0
)

# 2. Nœud chatbot
def chatbot(state: MessagesState):

    messages = [
        SystemMessage(
            content="Tu es un assistant intelligent. "
                    "Réponds en français."
        )
    ] + state["messages"]

    reponse = llm.invoke(messages)

    return {"messages": [reponse]}


# 3. Construire le graphe
graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 4. Ajouter la mémoire
memoire = InMemorySaver()

app = graph.compile(
    checkpointer=memoire
)

# 5. Identifiant de conversation
config = {
    "configurable": {
        "thread_id": "conversation_1"
    }
}

# 6. Interface interactive
print("=== LANGGRAPH AVEC MEMOIRE ===")

while True:
    question = input("\nVous : ").strip()

    if question.lower() in ["quit", "exit"]:
        break

    if not question:
        continue

    try:
        resultat = app.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config
        )

        print(
            "Agent :",
            resultat["messages"][-1].content
        )

    except Exception as e:
        print("Erreur :", e)
