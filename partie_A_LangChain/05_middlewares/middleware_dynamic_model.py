
import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

modele_rapide = ChatGroq(
    model=os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b"),
    temperature=0
)

modele_puissant = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0
)

@wrap_model_call
def dynamic_model(request, handler):
    messages = request.state["messages"]

    # Récupérer la dernière question de l'utilisateur
    question = next(
        (str(m.content).lower()
         for m in reversed(messages)
         if m.type == "human"),
        ""
    )

    mots_complexes = [
        "architecture",
        "analyse approfondie",
        "complexe",
        "raisonnement",
        "comparaison détaillée"
    ]

    if any(mot in question for mot in mots_complexes):
        print("[DYNAMIC MODEL] GPT-OSS-120B")
        modele = modele_puissant
    else:
        print("[DYNAMIC MODEL] GPT-OSS-20B")
        modele = modele_rapide

    return handler(request.override(model=modele))


agent = create_agent(
    model=modele_rapide,
    tools=[],
    middleware=[dynamic_model]
)

while True:
    question = input("\nVous : ").strip()
    if question.lower() == "quit":
        break

    resultat = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("Agent :", resultat["messages"][-1].content)
