
import os
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage

load_dotenv(find_dotenv())

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

@tool
def diviser(a: float, b: float) -> float:
    """Divise a par b. Utiliser pour les divisions."""
    print(f"[TOOL] diviser({a}, {b})")
    return a / b


@wrap_tool_call
def gerer_erreurs(request, handler):
    try:
        return handler(request)
    except ZeroDivisionError:
        print("[ERREUR] Division par zéro")
        return ToolMessage(
            content="Erreur : division par zéro impossible.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        print("[ERREUR TOOL]", type(e).__name__)
        return ToolMessage(
            content=(
                "L'outil a rencontré une erreur interne. "
                "Explique que le calcul n'a pas abouti."
            ),
            tool_call_id=request.tool_call["id"]
        )


agent = create_agent(
    model=llm,
    tools=[diviser],
    middleware=[gerer_erreurs],
    system_prompt=(
        "Tu es un assistant de calcul. "
        "Utilise l'outil diviser pour les divisions. "
        "N'invente jamais un résultat après une erreur."
    )
)

while True:
    question = input("\nVous : ").strip()
    if question.lower() == "quit":
        break

    resultat = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("Agent :", resultat["messages"][-1].content)
