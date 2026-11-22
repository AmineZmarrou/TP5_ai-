
import os
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())

# 1. Modèle Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 2. Outils de recherche
duckduckgo = DuckDuckGoSearchRun()

tavily = TavilySearch(
    max_results=3,
    search_depth="basic"
)

# 3. Liste des outils
tools = [duckduckgo, tavily]

# 4. Mémoire
memoire = InMemorySaver()

# 5. Création de l'agent
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memoire,
    system_prompt=(
        "Tu es un assistant intelligent spécialisé "
        "en informatique et intelligence artificielle. "
        "Réponds en français. "
        "Utilise DuckDuckGo pour les recherches web générales "
        "et Tavily pour les recherches nécessitant "
        "des sources structurées. "
        "Pour les informations récentes, effectue une recherche. "
        "Indique les sources lorsque disponibles."
    )
)

config = {
    "configurable": {
        "thread_id": "conversation_predefinis"
    }
}

print("=== Agent Groq avec recherche web ===")
print("Tape 'quit' pour quitter.\n")

while True:
    question = input("Vous : ").strip()

    if question.lower() in ["quit", "exit"]:
        break

    if not question:
        continue

    try:
        resultat = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config=config
        )

        # Afficher les outils réellement appelés
        for message in resultat["messages"]:
            if getattr(message, "tool_calls", None):
                for appel in message.tool_calls:
                    print("[TOOL UTILISÉ]", appel["name"])

        print("\nAgent :", resultat["messages"][-1].content)

    except Exception as e:
        print("Erreur :", e)
