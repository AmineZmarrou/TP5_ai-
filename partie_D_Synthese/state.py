from typing import TypedDict, Literal


class SAVState(TypedDict, total=False):

    # Question originale
    question: str

    # Categorie selectionnee
    category: Literal[
        "documentation",
        "garantie"
    ]

    # Justification du routage
    reason: str

    # Agent execute
    selected_agent: str

    # Outils executes
    tool_calls: list[str]

    # Sources documentaires
    sources: list[dict]

    # Figures retrouvees
    images: list[dict]

    # Resultat de garantie
    warranty: dict

    # Reponse finale
    answer: str
