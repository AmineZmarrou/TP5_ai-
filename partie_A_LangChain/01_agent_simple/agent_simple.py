import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# 1. Charger les variables d'environnement
load_dotenv(find_dotenv())

# 2. Initialiser le modèle LLM avec Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY")
)

# 3. Créer un agent simple
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "Tu es un assistant intelligent spécialisé "
        "en informatique et intelligence artificielle. "
        "Réponds clairement et en français."
    )
)

# 4. Poser une question à l'agent
question = "Quelle est la différence entre IA et Machine Learning ?"

resultat = agent.invoke({
    "messages": [
        {"role": "user", "content": question}
    ]
})

# 5. Afficher la réponse
print("Question :", question)
print("Réponse :", resultat["messages"][-1].content)
