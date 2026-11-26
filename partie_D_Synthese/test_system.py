import asyncio

from graph import app


TESTS = [

    {
        "question": (
            "Que signifie le voyant STATUS "
            "orange clignotant du PX-200 ?"
        ),

        "category": "documentation",

        "tools": ["rag_multimodal"],

        "image": "img_px200_01"
    },

    {
        "question": (
            "Comment nettoyer le PX-200 ?"
        ),

        "category": "documentation",

        "tools": ["rag_multimodal"]
    },

    {
        "question": (
            "Consulte la garantie du produit "
            "PX200-0042."
        ),

        "category": "garantie",

        "tools": ["consulter_garantie"],

        "trouve": True
    },

    {
        "question": (
            "Consulte la garantie du produit "
            "TV500-0125."
        ),

        "category": "garantie",

        "tools": ["consulter_garantie"],

        "trouve": True
    },

    {
        "question": (
            "Consulte la garantie du numero "
            "de serie XYZ-9999."
        ),

        "category": "garantie",

        "tools": ["consulter_garantie"],

        "trouve": False
    },

    {
        "question": (
            "Mon produit est-il sous garantie ?"
        ),

        "category": "garantie",

        "tools": []
    }
]


async def main():

    success = 0

    for index, test in enumerate(
        TESTS,
        start=1
    ):

        print("\n" + "=" * 50)

        print("TEST", index)

        print("Question :", test["question"])

        try:

            executed_nodes = []

            final_state = {}

            async for event in app.astream(

                {
                    "question": test["question"]
                },

                stream_mode="updates"

            ):

                for node, update in event.items():

                    executed_nodes.append(node)

                    if isinstance(update, dict):

                        final_state.update(update)

            # Verifier la categorie
            assert (
                final_state["category"]
                == test["category"]
            ), "Categorie incorrecte."

            # Verifier les outils
            assert (
                final_state.get("tool_calls", [])
                == test["tools"]
            ), "Outil incorrect."

            # Verifier qu'un seul agent est execute
            expected_node = (

                "agent_documentaire"

                if test["category"] == "documentation"

                else "agent_garantie"
            )

            assert executed_nodes == [

                "orchestrateur",

                expected_node

            ], "Routage incorrect."

            # Verifier la figure attendue
            if "image" in test:

                image_ids = [

                    img["image_id"]

                    for img in final_state.get(
                        "images", []
                    )
                ]

                assert test["image"] in image_ids, (
                    "Figure attendue introuvable."
                )

            # Verifier les sources documentaires
            if test["category"] == "documentation":

                assert final_state.get("sources"), (
                    "Aucune source documentaire."
                )

            # Verifier la garantie
            if "trouve" in test:

                warranty = final_state["warranty"]

                assert (
                    warranty["trouve"]
                    == test["trouve"]
                )

            assert final_state.get("answer")

            print("Noeuds :", executed_nodes)

            print(
                "Outils :",
                final_state.get("tool_calls", [])
            )

            print(
                "Reponse :",
                final_state["answer"]
            )

            print("[OK] Test valide")

            success += 1

        except Exception as error:

            print("[ECHEC]", str(error))

    print("\n" + "=" * 50)

    print(
        f"SCORE GLOBAL : {success}/{len(TESTS)}"
    )

    assert success == len(TESTS), (
        "Certains tests ont echoue."
    )


if __name__ == "__main__":

    asyncio.run(main())
