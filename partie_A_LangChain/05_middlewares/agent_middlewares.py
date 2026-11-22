
import os
import re
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    dynamic_prompt,
    wrap_tool_call,
    before_agent,
    HumanInTheLoopMiddleware,
)
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(find_dotenv())

# =========================
# 1. MODÈLES GROQ
# =========================

rapide = ChatGroq(
    model=os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b"),
    temperature=0
)

puissant = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0
)

# =========================
# 2. TOOLS
# =========================

@tool
def diviser(a: float, b: float) -> float:
    """Divise deux nombres."""
    print("[TOOL] Division")
    return a / b


@tool
def envoyer_email(destinataire: str, message: str) -> str:
    """Simule l'envoi d'un e-mail."""
    print("[TOOL] Envoi simulé :", destinataire)
    return "E-mail simulé avec succès."


# =========================
# 3. DYNAMIC MODEL
# =========================

@wrap_model_call
def dynamic_model(request, handler):
    question = next(
        (str(m.content).lower()
         for m in reversed(request.state["messages"])
         if m.type == "human"),
        ""
    )

    if any(x in question for x in [
        "architecture", "complexe", "approfondie"
    ]):
        print("[MIDDLEWARE] Modèle puissant")
        modele = puissant
    else:
        print("[MIDDLEWARE] Modèle rapide")
        modele = rapide

    return handler(request.override(model=modele))


# =========================
# 4. DYNAMIC PROMPT
# =========================

@dynamic_prompt
def dynamic_system_prompt(request):
    question = next(
        (str(m.content).lower()
         for m in reversed(request.state["messages"])
         if m.type == "human"),
        ""
    )

    if "python" in question:
        print("[MIDDLEWARE] Prompt Python")
        return "Tu es expert Python. Réponds en français."

    print("[MIDDLEWARE] Prompt général")
    return (
        "Tu es un assistant informatique. "
        "Réponds en français. "
        "Utilise les outils pour les calculs et les e-mails. "
        "L'envoi d'e-mail est uniquement simulé."
    )


# =========================
# 5. TOOL ERROR HANDLING
# =========================

@wrap_tool_call
def tool_error_handling(request, handler):
    try:
        return handler(request)
    except ZeroDivisionError:
        print("[MIDDLEWARE] Division par zéro")
        return ToolMessage(
            content="Division par zéro impossible.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        print("[MIDDLEWARE] Erreur :", type(e).__name__)
        return ToolMessage(
            content="Erreur interne pendant l'exécution de l'outil.",
            tool_call_id=request.tool_call["id"]
        )


# =========================
# 6. GUARDRAILS
# =========================

@before_agent(can_jump_to=["end"])
def guardrails(state, runtime):
    dernier = next(
        (m for m in reversed(state["messages"])
         if m.type == "human"),
        None
    )

    if dernier and re.search(
        r"(?:sk-proj-|sk-|gsk_)[A-Za-z0-9_-]{15,}",
        str(dernier.content)
    ):
        print("[MIDDLEWARE] Secret détecté")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Message bloqué : clé API détectée."
                    )
                ]
            },
            goto="end"
        )

    return None


# =========================
# 7. HUMAN IN THE LOOP
# =========================

hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "envoyer_email": {
            "allowed_decisions": ["approve", "reject"]
        }
    }
)

# =========================
# 8. AGENT
# =========================

agent = create_agent(
    model=rapide,
    tools=[diviser, envoyer_email],
    middleware=[
        guardrails,
        dynamic_model,
        dynamic_system_prompt,
        hitl,
        tool_error_handling,
    ],
    checkpointer=InMemorySaver()
)

config = {
    "configurable": {
        "thread_id": "tp5_middlewares"
    }
}

# =========================
# 9. CHATBOT
# =========================

print("=== AGENT GROQ AVEC 5 MIDDLEWARES ===")
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

        # Validation humaine
        while resultat.get("__interrupt__"):
            for interruption in resultat["__interrupt__"]:
                print("\n[VALIDATION HUMAINE]")
                print(interruption.value)

            choix = input(
                "Autoriser ? (oui/non) : "
            ).strip().lower()

            decision = (
                "approve" if choix == "oui" else "reject"
            )

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

        print("\nAgent :", resultat["messages"][-1].content)
        print("-" * 50)

    except Exception as e:
        print("Erreur :", type(e).__name__, str(e))
