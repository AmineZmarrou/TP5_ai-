import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st

from langchain_core.messages import HumanMessage


# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================

st.set_page_config(
    page_title="Agentic RAG | TP5",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. IMPORTATION DU GRAPHE LANGGRAPH
# ==========================================

RAG_DIR = (
    Path(__file__).resolve().parent.parent
    / "02_agentic_rag"
)

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


@st.cache_resource
def load_agent():

    from graph import app
    from config_rag import llm

    return app, llm.model_name


try:
    app, model_name = load_agent()

except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()


# ==========================================
# 3. PERSONNALISATION CSS
# ==========================================

st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-header {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #312E81,
        #4338CA
    );
    margin-bottom: 25px;
}

.main-header h1 {
    color: white;
    margin: 0;
    font-size: 32px;
}

.main-header p {
    color: #E0E7FF;
    margin-bottom: 0;
}

[data-testid="stChatMessage"] {
    border: 1px solid #303746;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 12px;
}

[data-testid="stMetric"] {
    background: #1B2130;
    border: 1px solid #303746;
    padding: 15px;
    border-radius: 12px;
}

.stButton button {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. GESTION DES CONVERSATIONS
# ==========================================

def create_conversation():

    thread_id = str(uuid4())

    st.session_state.conversations[thread_id] = {
        "title": "Nouvelle conversation",
        "messages": []
    }

    st.session_state.active_thread = thread_id


if "conversations" not in st.session_state:

    st.session_state.conversations = {}

    create_conversation()


# ==========================================
# 5. BARRE LATÉRALE
# ==========================================

with st.sidebar:

    st.title("🤖 Agentic RAG")

    st.caption("TP5 — LangGraph & Streamlit")

    st.divider()

    if st.button(
        "＋ Nouvelle conversation",
        use_container_width=True,
        type="primary"
    ):
        create_conversation()
        st.rerun()

    st.subheader("💬 Historique")

    conversations = list(
        st.session_state.conversations.items()
    )

    for thread_id, conversation in reversed(
        conversations
    ):

        if st.button(
            conversation["title"],
            key=f"thread_{thread_id}",
            use_container_width=True,
            type=(
                "primary"
                if thread_id == st.session_state.active_thread
                else "secondary"
            )
        ):
            st.session_state.active_thread = thread_id
            st.rerun()

    st.divider()

    st.subheader("⚙️ Configuration")

    st.success("Agent chargé")

    st.caption("Modèle LLM")
    st.code(model_name)

    st.caption("Orchestration")
    st.write("LangGraph")

    st.caption("Base vectorielle")
    st.write("ChromaDB")

    st.caption("Document")
    st.info("📄 Attention Is All You Need.pdf")

    show_sources = st.toggle(
        "Afficher les sources",
        value=True
    )

    show_execution = st.toggle(
        "Afficher les détails d'exécution",
        value=True
    )

    st.divider()

    st.caption("TP5 — Agentic AI & MCP")


# ==========================================
# 6. CONVERSATION ACTIVE
# ==========================================

active_thread = st.session_state.active_thread

conversation = (
    st.session_state.conversations[active_thread]
)

config = {
    "configurable": {
        "thread_id": active_thread
    },
    "recursion_limit": 25
}


# ==========================================
# 7. NOMS DES NŒUDS
# ==========================================

NODE_LABELS = {

    "agent": "Analyse de la question",

    "retrieve": "Recherche dans ChromaDB",

    "grade_documents": "Évaluation des passages",

    "rewrite_question": "Reformulation de la requête",

    "generate_answer": "Génération de la réponse RAG",

    "direct_answer": "Génération directe",

    "no_answer": "Information non trouvée"
}


# ==========================================
# 8. EXTRACTION DES SOURCES
# ==========================================

def extract_sources(documents):

    sources = []

    for doc in documents:

        metadata = doc.metadata

        source = Path(
            str(metadata.get("source", "Inconnue"))
        ).name

        page = metadata.get("page")

        if isinstance(page, int):
            page = page + 1
        else:
            page = "Inconnue"

        sources.append({
            "source": source,
            "page": page,
            "content": doc.page_content
        })

    return sources


# ==========================================
# 9. AFFICHAGE DES DÉTAILS
# ==========================================

def display_details(details):

    if show_execution:

        with st.expander(
            "📊 Détails de l'exécution"
        ):

            route = details.get("route")

            if route == "retrieve":
                st.info("Mode : recherche documentaire")
            else:
                st.info("Mode : réponse directe")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Passages",
                details.get("document_count", 0)
            )

            col2.metric(
                "Reformulations",
                details.get("retries", 0)
            )

            col3.metric(
                "Durée",
                f"{details.get('duration', 0):.2f} s"
            )

            if route == "retrieve":

                if details.get("relevant"):
                    st.success("Passages pertinents")
                else:
                    st.warning(
                        "Passages jugés insuffisants"
                    )

            st.markdown("**Chemin exécuté**")

            for node in details.get("nodes", []):

                label = NODE_LABELS.get(node, node)

                st.write(f"✓ {label}")

            st.markdown("**Requête finale**")

            st.code(
                details.get("query", ""),
                language=None
            )

    if show_sources and details.get("sources"):

        with st.expander(
            "📚 Passages documentaires récupérés"
        ):

            if not details.get("relevant"):
                st.warning(
                    "Ces passages ont été récupérés "
                    "mais jugés insuffisants."
                )

            for i, source in enumerate(
                details["sources"],
                start=1
            ):

                st.markdown(
                    f"**Passage {i}**"
                )

                st.write(
                    f"📄 {source['source']}"
                )

                st.caption(
                    f"Page : {source['page']}"
                )

                st.text(source["content"])

                st.divider()


# ==========================================
# 10. EN-TÊTE PRINCIPAL
# ==========================================

st.markdown("""
<div class="main-header">

    <h1>🤖 Agentic RAG Assistant</h1>

    <p>
        Interrogez vos documents avec
        LangGraph, Groq et ChromaDB.
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# 11. AFFICHAGE DE L'HISTORIQUE
# ==========================================

if not conversation["messages"]:

    st.subheader("Bienvenue 👋")

    st.write(
        "Posez une question sur le Transformer "
        "ou sur le document Attention Is All You Need."
    )

    st.markdown("**Exemples de questions :**")

    st.write(
        "• Combien de têtes d'attention utilise "
        "le Transformer ?"
    )

    st.write(
        "• Explique le Multi-Head Attention."
    )

    st.write(
        "• Combien de couches possède l'encodeur ?"
    )


for message in conversation["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "details" in message
        ):
            display_details(message["details"])


# ==========================================
# 12. SAISIE UTILISATEUR
# ==========================================

prompt = st.chat_input(
    "Posez votre question..."
)


# ==========================================
# 13. EXÉCUTION DE L'AGENT LANGGRAPH
# ==========================================

if prompt and prompt.strip():

    prompt = prompt.strip()

    if not conversation["messages"]:

        conversation["title"] = (
            prompt[:35] + "..."
            if len(prompt) > 35
            else prompt
        )

    conversation["messages"].append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        start_time = time.perf_counter()

        executed_nodes = []

        try:

            with st.status(
                "Exécution de l'Agentic RAG...",
                expanded=True
            ) as status:

                for event in app.stream(

                    {
                        "messages": [
                            HumanMessage(
                                content=prompt
                            )
                        ]
                    },

                    config=config,

                    stream_mode="updates"

                ):

                    for node_name in event:

                        if node_name.startswith("__"):
                            continue

                        executed_nodes.append(
                            node_name
                        )

                        label = NODE_LABELS.get(
                            node_name,
                            node_name
                        )

                        st.write(f"✓ {label}")

                # Récupérer le State final
                snapshot = app.get_state(config)

                result = snapshot.values

                answer = result.get("answer")

                if not answer:
                    raise RuntimeError(
                        "Aucune réponse finale générée."
                    )

                status.update(
                    label="Exécution terminée",
                    state="complete",
                    expanded=False
                )

            duration = (
                time.perf_counter() - start_time
            )

            documents = result.get(
                "documents", []
            ) or []

            details = {

                "route": result.get("route"),

                "query": result.get("query", ""),

                "retries": result.get("retries", 0),

                "relevant": result.get(
                    "relevant", False
                ),

                "document_count": len(documents),

                "duration": duration,

                "nodes": executed_nodes,

                "sources": extract_sources(documents)
            }

            st.markdown(answer)

            display_details(details)

            # Enregistrer la réponse et ses sources
            conversation["messages"].append({

                "role": "assistant",

                "content": answer,

                "details": details
            })

            st.rerun()

        except Exception as e:

            error_message = (
                f"Erreur : {type(e).__name__}. "
                "Consultez le terminal pour les détails."
            )

            print(
                "[STREAMLIT ERROR]",
                type(e).__name__,
                str(e)
            )

            st.error(error_message)

            conversation["messages"].append({

                "role": "assistant",

                "content": error_message
            })
