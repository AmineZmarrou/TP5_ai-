
from langgraph_sdk import get_sync_client

# Connexion au serveur LangGraph
client = get_sync_client(
    url="http://127.0.0.1:2024"
)

# Créer une conversation
thread = client.threads.create()

thread_id = thread["thread_id"]

print("Thread :", thread_id)

# Question
question = (
    "D'après mon PDF, combien de couches "
    "possède l'encodeur du Transformer ?"
)

# Exécuter le graphe distant
resultat = client.runs.wait(
    thread_id,
    "agentic_rag",
    input={
        "messages": [
            {
                "role": "human",
                "content": question
            }
        ]
    }
)

print("\nQuestion :", question)
print("\nRéponse :", resultat.get("answer"))
print("\nRoute :", resultat.get("route"))
print("Reformulations :", resultat.get("retries"))
