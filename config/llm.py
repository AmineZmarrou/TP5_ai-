import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

# Charger le fichier .env depuis la racine du projet
load_dotenv(find_dotenv())

def get_groq_llm(model: str = None, temperature: float = 0):
    """
    Retourne une instance configurée de ChatGroq.
    Par défaut, utilise le modèle puissant défini dans .env (ex: openai/gpt-oss-120b).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY introuvable dans le fichier .env")

    model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature
    )

def get_fast_groq_llm(model: str = None, temperature: float = 0):
    """
    Retourne une instance configurée du modèle rapide Groq (ex: openai/gpt-oss-20b).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY introuvable dans le fichier .env")

    model_name = model or os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature
    )

# Instances prêtes à l'emploi
llm = get_groq_llm()
modele_rapide = get_fast_groq_llm()
modele_puissant = llm
