import asyncio

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

from agent_langchain import build_agent

from agent_langgraph import build_graph


TESTS = [

    {
        "question": "Combien font 25 + 35 ?",
        "expected_tool": "additionner"
    },

    {
        "question": (
            "Consulte la garantie du produit "
            "PX200-0042."
        ),
        "expected_tool": "consulter_garantie"
    },

    {
        "question": (
            "Consulte la garantie du produit "
            "TV500-0125."
        ),
        "expected_tool": "consulter_garantie"
    },

    {
    "question": (
        "Consulte la garantie du produit "
        "dont le numero de serie est XYZ-9999."
    ),
    "expected_tool": "consulter_garantie"
},

    {
        "question": "Bonjour, comment vas-tu ?",
        "expected_tool": None
    }
]


async def test_agent(agent, name):

    print("\n" + "=" * 50)
    print("TESTS :", name)
    print("=" * 50)

    success_count = 0

    for i, test in enumerate(TESTS, start=1):

        print("\nTEST", i)
        print("Question :", test["question"])

        config = {

            "configurable": {
                "thread_id": f"{name}_test_{i}"
            },

            "recursion_limit": 15
        }

        try:

            result = await agent.ainvoke(

                {
                    "messages": [

                        HumanMessage(
                            content=test["question"]
                        )
                    ]
                },

                config=config
            )

            called_tools = []

            tool_results = []

            for message in result["messages"]:

                if isinstance(message, AIMessage):

                    for call in message.tool_calls:

                        called_tools.append(
                            call["name"]
                        )

                elif isinstance(message, ToolMessage):

                    tool_results.append(message)

            expected = test["expected_tool"]

            print("Outils appeles :", called_tools)

            if expected is None:

                assert not called_tools, (
                    "Aucun outil ne devait etre appele"
                )

            else:

                assert expected in called_tools, (
                    f"Outil attendu : {expected}"
                )

                assert any(

                    msg.name == expected

                    and msg.status != "error"

                    for msg in tool_results

                ), "Aucun resultat valide de l'outil"

            final_message = result["messages"][-1]

            assert isinstance(
                final_message,
                AIMessage
            )

            assert final_message.content

            print(
                "Reponse :",
                final_message.content
            )

            print("[OK] Test valide")

            success_count += 1

        except Exception as e:

            print("[ECHEC]", str(e))

    print(
        f"\nScore : {success_count}/{len(TESTS)}"
    )

    return success_count


async def main():

    langchain_agent = await build_agent()

    langgraph_agent = await build_graph()

    score_1 = await test_agent(
        langchain_agent,
        "LangChain"
    )

    score_2 = await test_agent(
        langgraph_agent,
        "LangGraph"
    )

    total = score_1 + score_2

    maximum = len(TESTS) * 2

    print(f"\nSCORE GLOBAL : {total}/{maximum}")

    assert total == maximum, (
        "Certains tests ont echoue."
    )


if __name__ == "__main__":

    asyncio.run(main())
