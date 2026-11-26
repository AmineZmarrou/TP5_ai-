import json

from datetime import date

from langchain_core.tools import tool

from config import DATA_DIR


@tool
def consulter_garantie(numero_serie: str) -> dict:

    """
    Consulte localement la garantie d'un produit
    a partir de son numero de serie.

    Cet outil est en lecture seule.
    """

    numero_serie = numero_serie.strip().upper()

    with open(

        DATA_DIR / "garanties.json",

        "r",

        encoding="utf-8"

    ) as f:

        garanties = json.load(f)

    if numero_serie not in garanties:

        return {

            "trouve": False,

            "numero_serie": numero_serie,

            "message": "Produit introuvable."
        }

    produit = garanties[numero_serie]

    expiration = date.fromisoformat(
        produit["date_expiration"]
    )

    today = date.today()

    active = today <= expiration

    return {

        "trouve": True,

        "numero_serie": numero_serie,

        "produit": produit["produit"],

        "date_achat": produit["date_achat"],

        "date_expiration": produit["date_expiration"],

        "garantie_active": active,

        "statut": (
            "Active" if active else "Expiree"
        ),

        "jours_restants": max(
            (expiration - today).days,
            0
        )
    }


if __name__ == "__main__":

    result = consulter_garantie.invoke({

        "numero_serie": "PX200-0042"
    })

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )
