
from graph import app, route_documents
from config_rag import MAX_RETRIES
from uuid import uuid4
from langchain_core.messages import HumanMessage


def main():
    assert route_documents({"relevant": True, "retries": 0}) == "generate"
    assert route_documents({"relevant": False, "retries": 0}) == "rewrite"
    assert route_documents({"relevant": False, "retries": MAX_RETRIES}) == "stop"
    questions = [
        (
            "D'après mon PDF, combien de têtes "
            "d'attention utilise le Transformer ?",
            "retrieve"
        ),
        (
            "Combien font 10 + 20 ?",
            "direct"
        ),
        (
            "D'après mon PDF, quel est le prix "
            "d'un abonnement Netflix ?",
            "retrieve"
        )
    ]


    for i, (question, route_attendue) in enumerate(
        questions, start=1
    ):

        print("\n" + "=" * 60)
        print("TEST", i)
        print("Question :", question)

        config = {
            "configurable": {
                "thread_id": f"test_{i}_{uuid4()}"
            },
            "recursion_limit": 25
        }

        resultat = app.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config
        )

        route = resultat["route"]

        print("Route attendue :", route_attendue)
        print("Route obtenue :", route)

        if route == route_attendue:
            print("[OK] Routage correct")
        else:
            print("[ECHEC] Routage incorrect")

        assert route == route_attendue, f"Route incorrecte : {route}"
        assert 0 <= resultat["retries"] <= MAX_RETRIES
        assert resultat["answer"].strip()
        if route == "direct":
            assert not resultat["documents"]
            assert resultat["retries"] == 0

        print("Réponse :", resultat["answer"])


if __name__ == "__main__":
    main()
