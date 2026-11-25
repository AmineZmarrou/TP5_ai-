
import sys
from pathlib import Path

# Ajouter le dossier courant aux chemins Python
CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

from state import RAGState
from config_rag import MAX_RETRIES

from nodes import (
    agent_node,
    retrieve_node,
    grade_documents_node,
    rewrite_question_node,
    generate_answer_node,
    direct_answer_node,
    no_answer_node
)


# ====================================
# 1. ROUTAGE APRÈS L'AGENT
# ====================================

def route_agent(state: RAGState) -> Literal[
    "retrieve", "direct"
]:
    return state["route"]


# ====================================
# 2. ROUTAGE APRÈS ÉVALUATION
# ====================================

def route_documents(state: RAGState) -> Literal[
    "generate", "rewrite", "stop"
]:

    if state.get("relevant", False):
        return "generate"

    if state.get("retries", 0) >= MAX_RETRIES:
        return "stop"

    return "rewrite"


# ====================================
# 3. CRÉATION DU GRAPHE
# ====================================

workflow = StateGraph(RAGState, input_schema=MessagesState)


# ====================================
# 4. AJOUT DES NODES
# ====================================

workflow.add_node("agent", agent_node)

workflow.add_node("retrieve", retrieve_node)

workflow.add_node(
    "grade_documents",
    grade_documents_node
)

workflow.add_node(
    "rewrite_question",
    rewrite_question_node
)

workflow.add_node(
    "generate_answer",
    generate_answer_node
)

workflow.add_node(
    "direct_answer",
    direct_answer_node
)

workflow.add_node(
    "no_answer",
    no_answer_node
)


# ====================================
# 5. POINT DE DÉPART
# ====================================

workflow.add_edge(START, "agent")


# ====================================
# 6. TRANSITIONS CONDITIONNELLES
# ====================================

workflow.add_conditional_edges(
    "agent",
    route_agent,
    {
        "retrieve": "retrieve",
        "direct": "direct_answer"
    }
)

workflow.add_edge(
    "retrieve",
    "grade_documents"
)

workflow.add_conditional_edges(
    "grade_documents",
    route_documents,
    {
        "generate": "generate_answer",
        "rewrite": "rewrite_question",
        "stop": "no_answer"
    }
)


# ====================================
# 7. BOUCLE DE RECHERCHE
# ====================================

workflow.add_edge(
    "rewrite_question",
    "retrieve"
)


# ====================================
# 8. TERMINAISON
# ====================================

workflow.add_edge("generate_answer", END)
workflow.add_edge("direct_answer", END)
workflow.add_edge("no_answer", END)


# ====================================
# 9. COMPILATION POUR LE TERMINAL
# ====================================

memory = InMemorySaver()

app = workflow.compile(
    checkpointer=memory
)


# ====================================
# 10. COMPILATION POUR STUDIO
# ====================================

studio_graph = workflow.compile()
