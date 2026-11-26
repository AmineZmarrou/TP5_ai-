import json

from pathlib import Path
from datetime import date

from mcp.server.fastmcp import FastMCP


# ==========================================
# 1. CONFIGURATION DU SERVEUR MCP
# ==========================================

mcp = FastMCP(
    "TP5-MCP-Server",
    host="127.0.0.1",
    port=8000
)


# ==========================================
# 2. CHEMIN DE LA BASE DE DONNÉES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "garanties.json"
)


# ==========================================
# 3. OUTIL MCP : ADDITIONNER
# ==========================================

@mcp.tool()
def additionner(a: float, b: float) -> float:
    """
    Additionne deux nombres et retourne
    le résultat de leur somme.

    Args:
        a: Premier nombre.
        b: Deuxième nombre.
    """

    return a + b


# ==========================================
# 4. OUTIL MCP : CONSULTER GARANTIE
# ==========================================

@mcp.tool()
def consulter_garantie(
    numero_serie: str
) -> dict:
    """
    Consulte la garantie d'un produit
    électronique à partir de son numéro
    de série.

    Args:
        numero_serie: Numéro de série du produit.
    """

    numero_serie = numero_serie.strip().upper()

    if not numero_serie:

        return {
            "trouve": False,
            "message": "Numéro de série vide."
        }

    # Charger les données
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as fichier:

        garanties = json.load(fichier)

    # Vérifier l'existence du produit
    if numero_serie not in garanties:

        return {
            "trouve": False,
            "numero_serie": numero_serie,
            "message": "Produit introuvable."
        }

    produit = garanties[numero_serie]

    # Vérifier la date d'expiration
    expiration = date.fromisoformat(
        produit["date_expiration"]
    )

    aujourd_hui = date.today()

    active = aujourd_hui <= expiration

    jours_restants = max(
        (expiration - aujourd_hui).days,
        0
    )

    return {
        "trouve": True,
        "numero_serie": numero_serie,
        "produit": produit["produit"],
        "date_achat": produit["date_achat"],
        "date_expiration": produit["date_expiration"],
        "garantie_active": active,
        "statut": (
            "Active" if active else "Expirée"
        ),
        "jours_restants": jours_restants
    }


# ==========================================
# 5. DÉMARRAGE DU SERVEUR
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("TP5 - SERVEUR MCP")
    print("Transport : Streamable HTTP")
    print("Adresse : http://127.0.0.1:8000/mcp")
    print("=" * 50)

    mcp.run(
        transport="streamable-http"
    )
