import asyncio

from langchain.agents import create_agent

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.checkpoint.memory import (
    InMemorySaver
)

from config_mcp import (
    get_llm,
    get_mcp_client,
    SYSTEM_PROMPT
)


# =====================================
# 1. CREATION DE L'AGENT
# =====================================

async def build_agent():

    llm = get_llm()

    client = get_mcp_client()

    tools = await client.get_tools()

    print("\nOutils MCP charges :")

    for tool in tools:
        print("-", tool.name)

    agent = create_agent(

        model=llm,

        tools=tools,

        system_prompt=SYSTEM_PROMPT,

        checkpointer=InMemorySaver()
    )

    return agent


# =====================================
# 2. CHATBOT INTERACTIF
# =====================================

async def main():

    agent = await build_agent()

    config = {

        "configurable": {
            "thread_id": "langchain_mcp_001"
        },

        "recursion_limit": 15
    }

    print("\n" + "=" * 50)
    print("AGENT LANGCHAIN + MCP + GROQ")
    print("Tapez exit pour quitter.")
    print("=" * 50)

    while True:

        question = input("\nVous : ").strip()

        if question.lower() in [
            "exit", "quit"
        ]:
            break

        if not question:
            continue

        try:

            # Recuperer la taille de l'ancien historique
            snapshot = await agent.aget_state(config)

            previous_count = len(
                snapshot.values.get("messages", [])
            )

            # Executer l'agent
            result = await agent.ainvoke(

                {
                    "messages": [
                        HumanMessage(content=question)
                    ]
                },

                config=config
            )

            # Afficher uniquement les nouveaux messages
            new_messages = result[
                "messages"
            ][previous_count:]

            for message in new_messages:

                if isinstance(message, AIMessage):

                    for call in message.tool_calls:

                        print(
                            "\n[MCP TOOL]",
                            call["name"]
                        )

                        print(
                            "[ARGUMENTS]",
                            call["args"]
                        )

                elif isinstance(message, ToolMessage):

                    print(
                        "[MCP RESULT]",
                        message.content
                    )

            print(
                "\nAssistant :",
                result["messages"][-1].content
            )

        except Exception as e:

            print(
                "\nErreur :",
                type(e).__name__,
                str(e)
            )


if __name__ == "__main__":

    asyncio.run(main())
