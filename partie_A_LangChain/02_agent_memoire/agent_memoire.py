
import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

# 1. Charger les variables d'environnement
load_dotenv(find_dotenv())

# 2. Initialiser le modèle Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY")
)

# 3. Créer une mémoire conversationnelle
memoire = InMemorySaver()

# 4. Créer l'agent avec mémoire
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "Tu es un assistant intelligent spécialisé "
        "en informatique et intelligence artificielle. "
        "Réponds clairement en français. "
        "Utilise l'historique des conversations."
    ),
    checkpointer=memoire
)

# 5. Identifier la conversation
config = {
    "configurable": {
        "thread_id": "conversation_1"
    }
}

# 6. Fonction de conversation
def discuter(question):
    resultat = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        config=config
    )

    print("Agent :", resultat["messages"][-1].content)


# 7. Interface interactive
print("=== Agent Groq avec mémoire ===")
print("Tape 'quit' pour quitter.\n")

while True:
    question = input("Vous : ").strip()

    if question.lower() == "quit":
        print("Fin de la conversation.")
        break

    if not question:
        continue

    try:
        discuter(question)
    except Exception as e:
        print("Erreur :", e)
