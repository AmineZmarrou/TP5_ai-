
import os
import re
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import before_agent
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langgraph.types import Command

load_dotenv(find_dotenv())

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

# Détecter certaines clés API
PATTERN_CLE = re.compile(
    r"(?:sk-proj-|sk-|gsk_)[A-Za-z0-9_-]{15,}"
)

@before_agent(can_jump_to=["end"])
def guardrails(state, runtime):
    dernier_message = next(
        (m for m in reversed(state["messages"])
         if m.type == "human"),
        None
    )

    if dernier_message is None:
        return None

    texte = str(dernier_message.content)

    if PATTERN_CLE.search(texte):
        print("[GUARDRAIL] Clé API détectée")

        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=(
                            "Une clé API potentielle a été "
                            "détectée. Ne partage pas de "
                            "secrets dans cette conversation."
                        )
                    )
                ]
            },
            goto="end"
        )

    print("[GUARDRAIL] Message accepté")
    return None


agent = create_agent(
    model=llm,
    tools=[],
    middleware=[guardrails],
    system_prompt="Tu es un assistant informatique."
)

while True:
    question = input("\nVous : ").strip()

    if question.lower() == "quit":
        break

    resultat = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("Agent :", resultat["messages"][-1].content)
