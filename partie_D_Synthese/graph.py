from langgraph.graph import (
    StateGraph,
    START,
    END
)

from state import SAVState

from nodes import (
    orchestrator_node,
    documentation_node,
    garantie_node
)


# =====================================
# 1. ROUTAGE CONDITIONNEL
# =====================================

def route_agent(state: SAVState):

    return state["category"]


# =====================================
# 2. CREATION DU GRAPHE
# =====================================

workflow = StateGraph(SAVState)


# =====================================
# 3. AJOUT DES NOEUDS
# =====================================

workflow.add_node(

    "orchestrateur",

    orchestrator_node
)

workflow.add_node(

    "agent_documentaire",

    documentation_node
)

workflow.add_node(

    "agent_garantie",

    garantie_node
)


# =====================================
# 4. TRANSITIONS
# =====================================

workflow.add_edge(

    START,

    "orchestrateur"
)

workflow.add_conditional_edges(

    "orchestrateur",

    route_agent,

    {

        "documentation":
            "agent_documentaire",

        "garantie":
            "agent_garantie"
    }
)

workflow.add_edge(

    "agent_documentaire",

    END
)

workflow.add_edge(

    "agent_garantie",

    END
)


# =====================================
# 5. COMPILATION
# =====================================

app = workflow.compile()
