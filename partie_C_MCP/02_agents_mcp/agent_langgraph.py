import asyncio

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage
)

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
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
# 1. CONSTRUCTION DU GRAPHE
# =====================================

async def build_graph():

    # Charger le LLM
    llm = get_llm()

    # Recuperer les outils MCP
    client = get_mcp_client()

    tools = await client.get_tools()

    print("\nOutils disponibles :")

    for tool in tools:
        print("-", tool.name)

    # Associer les outils au LLM
    llm_with_tools = llm.bind_tools(tools)

    # =================================
    # 2. NOEUD AGENT
    # =================================

    async def agent_node(state: MessagesState):

        print("\n[NODE] agent")

        messages = [

            SystemMessage(
                content=SYSTEM_PROMPT
            ),

            *state["messages"]
        ]

        response = await llm_with_tools.ainvoke(
            messages
        )

        for call in response.tool_calls:

            print(
                "[TOOL CALL]",
                call["name"],
                call["args"]
            )

        return {
            "messages": [response]
        }

    # =================================
    # 3. NOEUD TOOLS
    # =================================

    tool_node = ToolNode(tools)

    # =================================
    # 4. CREATION DU STATEGRAPH
    # =================================

    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "agent",
        agent_node
    )

    workflow.add_node(
        "tools",
        tool_node
    )

    # =================================
    # 5. TRANSITIONS
    # =================================

    workflow.add_edge(
        START,
        "agent"
    )

    workflow.add_conditional_edges(

        "agent",

        tools_condition,

        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge(
        "tools",
        "agent"
    )

    # =================================
    # 6. COMPILATION AVEC MEMOIRE
    # =================================

    graph = workflow.compile(

        checkpointer=InMemorySaver()
    )

    return graph


# =====================================
# 7. CHATBOT INTERACTIF
# =====================================

async def main():

    graph = await build_graph()

    config = {

        "configurable": {
            "thread_id": "langgraph_mcp_001"
        },

        "recursion_limit": 15
    }

    print("\n" + "=" * 50)
    print("AGENT LANGGRAPH + MCP + GROQ")
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

            async for event in graph.astream(

                {
                    "messages": [
                        HumanMessage(
                            content=question
                        )
                    ]
                },

                config=config,

                stream_mode="updates"

            ):

                for node_name, update in event.items():

                    if node_name.startswith("__"):
                        continue

                    print(
                        "[EXECUTION]",
                        node_name
                    )

                    if not isinstance(update, dict):
                        continue

                    for message in update.get(
                        "messages", []
                    ):

                        if isinstance(
                            message,
                            ToolMessage
                        ):

                            print(
                                "[MCP RESULT]",
                                message.content
                            )

            snapshot = await graph.aget_state(
                config
            )

            final_message = snapshot.values[
                "messages"
            ][-1]

            if isinstance(final_message, AIMessage):

                print(
                    "\nAssistant :",
                    final_message.content
                )

        except Exception as e:

            print(
                "\nErreur :",
                type(e).__name__,
                str(e)
            )


if __name__ == "__main__":

    asyncio.run(main())
