
import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

@dynamic_prompt
def adapter_prompt(request):
    question = next(
        (str(m.content).lower()
         for m in reversed(request.state["messages"])
         if m.type == "human"),
        ""
    )

    if "python" in question:
        print("[PROMPT] Expert Python")
        return (
            "Tu es un expert Python. "
            "Explique avec des exemples de code. "
            "Réponds en français."
        )

    elif "réseau" in question:
        print("[PROMPT] Expert Réseaux")
        return (
            "Tu es un ingénieur réseau. "
            "Explique les protocoles et architectures "
            "avec des exemples pratiques."
        )

    print("[PROMPT] Assistant général")
    return (
        "Tu es un assistant informatique. "
        "Réponds clairement et en français."
    )


agent = create_agent(
    model=llm,
    tools=[],
    middleware=[adapter_prompt]
)

while True:
    question = input("\nVous : ").strip()

    if question.lower() == "quit":
        break

    resultat = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("Agent :", resultat["messages"][-1].content)
