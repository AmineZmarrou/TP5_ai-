
import os
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(find_dotenv())

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

@tool
def envoyer_email(destinataire: str, message: str) -> str:
    """Simule l'envoi d'un e-mail à un destinataire."""
    print("[SIMULATION] E-mail envoyé")
    print("Destinataire :", destinataire)
    print("Message :", message)
    return "Envoi simulé avec succès."


agent = create_agent(
    model=llm,
    tools=[envoyer_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "envoyer_email": {
                    "allowed_decisions": [
                        "approve",
                        "reject"
                    ]
                }
            }
        )
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "Tu es un assistant professionnel. "
        "Pour envoyer un e-mail, utilise envoyer_email. "
        "L'outil simule uniquement l'envoi."
    )
)

config = {
    "configurable": {
        "thread_id": "test_hitl_1"
    }
}

question = (
    "Envoie un e-mail à test@example.com "
    "pour dire Bonjour, ceci est un test."
)

resultat = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": question}
        ]
    },
    config=config
)

# Vérifier si l'agent attend une décision
interruptions = resultat.get("__interrupt__", ())

if interruptions:
    print("\n[HITL] Validation humaine nécessaire")

    for interruption in interruptions:
        print(interruption.value)

    choix = input(
        "\nApprouver l'action ? (oui/non) : "
    ).strip().lower()

    decision = (
        "approve" if choix == "oui" else "reject"
    )

    # Reprendre l'exécution après décision
    resultat = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": decision}
                ]
            }
        ),
        config=config
    )

print("\nRéponse finale :")
print(resultat["messages"][-1].content)
