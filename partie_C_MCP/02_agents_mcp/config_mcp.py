import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_mcp_adapters.client import (
    MultiServerMCPClient
)


# =====================================
# 1. CHARGEMENT DU FICHIER .env
# =====================================

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")


# =====================================
# 2. CONFIGURATION
# =====================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://127.0.0.1:8000/mcp"
)


# =====================================
# 3. CREATION DU LLM
# =====================================

def get_llm():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY absente du fichier .env"
        )

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )


# =====================================
# 4. CLIENT MCP
# =====================================

def get_mcp_client():

    return MultiServerMCPClient({

        "tp5_server": {

            "url": MCP_SERVER_URL,

            "transport": "streamable_http"
        }
    })


# =====================================
# 5. PROMPT COMMUN
# =====================================

SYSTEM_PROMPT = """
Tu es un assistant intelligent francophone.

Tu disposes de deux outils MCP distants :

- additionner : effectuer une addition.
- consulter_garantie : rechercher la garantie
  d'un produit par son numero de serie.

REGLES OBLIGATOIRES :

1. Pour toute addition, appelle l'outil additionner.

2. Pour toute demande de consultation de garantie
   contenant un numero de serie ou un identifiant
   de produit, appelle consulter_garantie.

3. Appelle consulter_garantie meme si le numero
   de serie semble incorrect, fictif ou inexistant.

4. Ne decide jamais toi-meme si un produit existe.
   Seul le serveur MCP peut verifier son existence.

5. Les valeurs INCONNU, INVALID, TEST et XYZ-9999
   doivent etre traitees comme des identifiants
   lorsqu'elles sont fournies explicitement.

6. Demande le numero de serie uniquement si aucun
   identifiant n'a ete fourni par l'utilisateur
   ou dans l'historique de conversation.

7. Si l'outil retourne trouve=False, indique
   que le produit est introuvable.

8. Ne fabrique jamais les informations de garantie.

9. Pour les questions generales sans rapport
   avec les outils, reponds directement.

10. Apres avoir obtenu le resultat d'un outil,
    utilise ce resultat pour formuler la reponse
    finale sans repeter inutilement l'appel.

11. Reponds toujours en francais.

EXEMPLES :

Utilisateur : Combien font 25 + 35 ?
Action : appeler additionner(a=25, b=35).

Utilisateur : Consulte la garantie de PX200-0042.
Action : appeler consulter_garantie(
    numero_serie="PX200-0042"
).

Utilisateur : Consulte la garantie du produit INCONNU.
Action : appeler consulter_garantie(
    numero_serie="INCONNU"
).

Utilisateur : Consulte la garantie de XYZ-9999.
Action : appeler consulter_garantie(
    numero_serie="XYZ-9999"
).

Utilisateur : Bonjour.
Action : repondre directement sans outil.
"""

