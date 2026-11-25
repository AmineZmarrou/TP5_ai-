
from typing import Literal
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class RAGState(MessagesState, total=False):

    # Question originale de l'utilisateur
    question: str

    # Requête utilisée pour la recherche
    query: str

    # Documents récupérés
    documents: list[Document]

    # Décision de l'agent
    route: Literal["retrieve", "direct"]

    # Pertinence des documents
    relevant: bool

    # Nombre de reformulations
    retries: int

    # Réponse finale
    answer: str
