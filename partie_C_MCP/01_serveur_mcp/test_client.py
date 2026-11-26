import asyncio

from mcp import ClientSession

from mcp.client.streamable_http import (
    streamablehttp_client
)


SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main():

    # ======================================
    # 1. CONNEXION AU SERVEUR MCP
    # ======================================

    async with streamablehttp_client(
        SERVER_URL
    ) as (read_stream, write_stream, _):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            # ==============================
            # 2. INITIALISER LA SESSION
            # ==============================

            await session.initialize()

            print("[OK] Connexion MCP réussie")

            # ==============================
            # 3. LISTER LES OUTILS
            # ==============================

            resultat = await session.list_tools()

            print("\nOutils disponibles :")

            for outil in resultat.tools:

                print("-", outil.name)

                print(
                    "  Description :",
                    outil.description
                )

            noms = {
                outil.name
                for outil in resultat.tools
            }

            assert noms == {
                "additionner",
                "consulter_garantie"
            }

            # ==============================
            # 4. TESTER ADDITIONNER
            # ==============================

            print("\nTEST 1 : Additionner")

            resultat = await session.call_tool(

                "additionner",

                arguments={
                    "a": 25,
                    "b": 35
                }
            )

            assert not resultat.isError

            for contenu in resultat.content:

                if contenu.type == "text":
                    print(contenu.text)

            # ==============================
            # 5. TESTER CONSULTER GARANTIE
            # ==============================

            print("\nTEST 2 : Garantie active")

            resultat = await session.call_tool(

                "consulter_garantie",

                arguments={
                    "numero_serie": "PX200-0042"
                }
            )

            assert not resultat.isError

            for contenu in resultat.content:

                if contenu.type == "text":
                    print(contenu.text)

            # ==============================
            # 6. TESTER UN PRODUIT INCONNU
            # ==============================

            print("\nTEST 3 : Produit inconnu")

            resultat = await session.call_tool(

                "consulter_garantie",

                arguments={
                    "numero_serie": "INCONNU"
                }
            )

            assert not resultat.isError

            for contenu in resultat.content:

                if contenu.type == "text":
                    print(contenu.text)

            print("\nTests de communication terminés.")


if __name__ == "__main__":

    asyncio.run(main())
