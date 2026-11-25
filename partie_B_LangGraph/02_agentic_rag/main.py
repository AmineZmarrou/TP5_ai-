
from langchain_core.messages import HumanMessage

from graph import app
from config_rag import llm


def main():
    # Configuration de la mémoire
    config = {
        "configurable": {
            "thread_id": "agentic_rag_langgraph"
        },
        "recursion_limit": 25
    }


    print("=" * 60)
    print(" AGENTIC RAG - LANGGRAPH + GROQ")
    print("=" * 60)

    print("Document : Attention Is All You Need")
    print("Modèle :", llm.model_name)
    print("Tape 'quit' pour quitter.")


    while True:

        question = input("\nVous : ").strip()

        if question.lower() in ["quit", "exit"]:
            print("Fin de la conversation.")
            break

        if not question:
            continue

        try:

            resultat = app.invoke(
                {
                    "messages": [
                        HumanMessage(content=question)
                    ]
                },
                config=config
            )

            print("\n" + "=" * 60)

            print("Agent :", resultat["answer"])

            print("\n--- Informations d'exécution ---")

            print("Route :", resultat["route"])

            print(
                "Documents :",
                len(resultat["documents"])
            )

            print(
                "Reformulations :",
                resultat["retries"]
            )

            print("=" * 60)

        except Exception as e:

            print(
                "Erreur :",
                type(e).__name__,
                str(e)
            )


if __name__ == "__main__":
    main()
