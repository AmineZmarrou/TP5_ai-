import asyncio

from config_mcp import get_mcp_client


async def main():

    # 1. Connexion MCP
    client = get_mcp_client()

    # 2. Recuperation des outils
    tools = await client.get_tools()

    print("\nOutils MCP disponibles :\n")

    for tool in tools:

        print("Nom :", tool.name)
        print("Description :", tool.description)
        print("Arguments :", tool.args)
        print("-" * 50)

    # 3. Dictionnaire des outils
    tools_dict = {
        tool.name: tool
        for tool in tools
    }

    assert "additionner" in tools_dict

    assert "consulter_garantie" in tools_dict

    # 4. Tester additionner
    print("\nTEST 1 : Addition")

    result = await tools_dict[
        "additionner"
    ].ainvoke({

        "a": 25,
        "b": 35
    })

    print(result)

    # 5. Tester consulter_garantie
    print("\nTEST 2 : Garantie")

    result = await tools_dict[
        "consulter_garantie"
    ].ainvoke({

        "numero_serie": "PX200-0042"
    })

    print(result)

    # 6. Produit inconnu
    print("\nTEST 3 : Produit inconnu")

    result = await tools_dict[
        "consulter_garantie"
    ].ainvoke({

        "numero_serie": "INCONNU"
    })

    print(result)

    print("\nTests MCP termines.")


if __name__ == "__main__":

    asyncio.run(main())
