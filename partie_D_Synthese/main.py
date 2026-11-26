import asyncio

from graph import app

from config import IMG_DIR


async def main():

    print("=" * 55)

    print("ASSISTANT INTELLIGENT SAV")

    print("LangGraph + Groq + MCP")

    print("Tapez exit pour quitter.")

    print("=" * 55)

    while True:

        question = input("\nVous : ").strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:
            break

        if not question:
            continue

        try:

            result = await app.ainvoke({

                "question": question,

                "tool_calls": [],

                "sources": [],

                "images": [],

                "warranty": {}
            })

            print("\nCategorie :",
                  result["category"])

            print("Agent selectionne :",
                  result["selected_agent"])

            print("Outils :",
                  result.get("tool_calls", []))

            print("\nAssistant :")

            print(result["answer"])

            # Affichage des sources
            if result.get("sources"):

                print("\nSOURCES :")

                seen = set()

                for source in result["sources"]:

                    ref = (
                        source["source"],
                        source["page"]
                    )

                    if ref in seen:
                        continue

                    seen.add(ref)

                    print(
                        "-",
                        source["source"],
                        "Page",
                        source["page"]
                    )

            # Affichage des references images
            if result.get("images"):

                print("\nFIGURES :")

                for image in result["images"]:

                    print(
                        "-",
                        image["image_id"]
                    )

                    print(
                        "  Fichier :",
                        IMG_DIR / image["file"]
                    )

                    print(
                        "  Page :",
                        image["page"]
                    )

        except Exception as error:

            print(
                "[ERREUR]",
                type(error).__name__,
                str(error)
            )


if __name__ == "__main__":

    asyncio.run(main())
