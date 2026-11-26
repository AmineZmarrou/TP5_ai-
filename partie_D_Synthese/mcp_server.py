from mcp.server.fastmcp import FastMCP

from config import (
    get_vectorstore,
    IMG_DIR
)


mcp = FastMCP(

    "SAV-RAG-Multimodal",

    host="127.0.0.1",

    port=8001
)

store = None


def load_store():

    global store

    if store is None:

        store = get_vectorstore()

    return store


@mcp.tool()
def rag_multimodal(question: str) -> dict:

    """
    Recherche des passages techniques et des
    figures dans les manuels des produits.

    Retourne les textes, les sources, les pages
    et les references des images pertinentes.
    """

    if not question.strip():

        return {
            "success": False,
            "message": "Question vide."
        }

    db = load_store()

    # =================================
    # 1. RECHERCHE TEXTUELLE
    # =================================

    text_docs = db.similarity_search_with_score(

        question,

        k=3,

        filter={
            "type": "text"
        }
    )

    # =================================
    # 2. RECHERCHE DES FIGURES
    # =================================

    image_docs = db.similarity_search_with_score(

        question,

        k=2,

        filter={
            "type": "image"
        }
    )

    texts = []

    images = []

    # Seuil pedagogique de distance cosinus
    MAX_DISTANCE = 0.65

    for doc, distance in text_docs:

        if distance > MAX_DISTANCE:
            continue

        texts.append({

            "content": doc.page_content,

            "source": doc.metadata["source"],

            "page": doc.metadata["page"]
        })

    for doc, distance in image_docs:

        if distance > MAX_DISTANCE:
            continue

        file_path = (
            IMG_DIR / doc.metadata["file"]
        )

        if not file_path.is_file():
            continue

        images.append({

            "image_id": doc.metadata["image_id"],

            "description": doc.page_content,

            "source": doc.metadata["source"],

            "page": doc.metadata["page"],

            "file": doc.metadata["file"]
        })

    return {

        "success": True,

        "question": question,

        "texts": texts,

        "images": images
    }


if __name__ == "__main__":

    print("=" * 50)

    print("SERVEUR MCP DOCUMENTAIRE")

    print("http://127.0.0.1:8001/mcp")

    print("=" * 50)

    mcp.run(
        transport="streamable-http"
    )
