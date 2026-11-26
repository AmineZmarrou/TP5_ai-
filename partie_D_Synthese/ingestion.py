import json

import pymupdf

from langchain_core.documents import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    DOC_DIR,
    DATA_DIR,
    get_vectorstore
)


def main():

    documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    # =================================
    # 1. EXTRACTION DU TEXTE
    # =================================

    for pdf_path in DOC_DIR.glob("*.pdf"):

        print("[PDF]", pdf_path.name)

        with pymupdf.open(pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf,
                start=1
            ):

                text = page.get_text().strip()

                if not text:
                    continue

                doc = Document(

                    page_content=text,

                    metadata={
                        "type": "text",
                        "source": pdf_path.name,
                        "page": page_number
                    }
                )

                documents.extend(
                    splitter.split_documents([doc])
                )

    # =================================
    # 2. INDEXATION DES FIGURES
    # =================================

    with open(
        DATA_DIR / "figures.json",
        "r",
        encoding="utf-8"
    ) as f:

        figures = json.load(f)

    for figure in figures:

        doc = Document(

            page_content=figure["description"],

            metadata={
                "type": "image",
                "source": figure["source"],
                "page": figure["page"],
                "image_id": figure["id"],
                "file": figure["file"]
            }
        )

        documents.append(doc)

    if not documents:
        raise ValueError("Aucun document a indexer.")

    # =================================
    # 3. STOCKAGE VECTORIEL
    # =================================

    store = get_vectorstore()

    # Reconstruire la collection de demonstration
    existing_ids = store.get()["ids"]

    if existing_ids:
        store.delete(ids=existing_ids)

    ids = [
        f"sav_doc_{i}"
        for i in range(len(documents))
    ]

    store.add_documents(
        documents=documents,
        ids=ids
    )

    print("\n[OK] Indexation terminee.")

    print(
        "Documents indexes :",
        len(documents)
    )

    print(
        "Textes :",
        sum(
            d.metadata["type"] == "text"
            for d in documents
        )
    )

    print(
        "Figures :",
        sum(
            d.metadata["type"] == "image"
            for d in documents
        )
    )


if __name__ == "__main__":

    main()
