
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.rag_pipeline import (
    initialize_rag_system,
    answer_question
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MedEvidence-AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
EVALUATION_DIR = BASE_DIR / "evaluation"


# ==========================================
# LOAD CUSTOM CSS
# ==========================================

def load_css():

    css_path = ASSETS_DIR / "style.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ==========================================
# INITIALIZE RAG SYSTEM
# ==========================================

@st.cache_resource(show_spinner=False)
def load_rag_system():

    return initialize_rag_system()


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================
# REUSABLE UI FUNCTIONS
# ==========================================

def section_header(title, subtitle):

    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value, description, icon):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <span>{label}</span>
                <span class="metric-icon">{icon}</span>
            </div>
            <h2>{value}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_sources(sources):

    if not sources:
        st.info("No cited sources were returned for this answer.")
        return

    st.markdown("### 📚 Supporting medical sources")

    for source in sources:

        number = source.get("evidence_number", "")
        name = source.get("source_name", "Unknown source")
        chunk_id = source.get("chunk_id", "Unknown")
        url = source.get("source_url", "")

        with st.expander(
            f"Evidence [{number}] — {name}"
        ):

            st.write(f"**Chunk ID:** {chunk_id}")

            if isinstance(url, str) and url.startswith(("https://", "http://")):
                st.markdown(f"[Open original source]({url})")
            else:
                st.caption("No source URL available.")


def show_retrieved_chunks(chunks):

    if chunks is None or chunks.empty:
        st.info("No evidence chunks available.")
        return

    st.markdown("### Retrieved evidence")

    for _, chunk in chunks.iterrows():

        score = float(chunk.get("retrieval_score", 0))
        question = chunk.get("question", "Not available")
        answer = chunk.get("answer_chunk", "")
        source = chunk.get("source_name", "Unknown")
        chunk_id = chunk.get("chunk_id", "Unknown")

        with st.expander(
            f"{source}  |  Similarity: {score:.4f}"
        ):

            st.write(f"**Related question:** {question}")
            st.write(f"**Chunk ID:** {chunk_id}")
            st.write("**Retrieved medical information:**")
            st.write(answer)

            st.progress(
                min(max(score, 0.0), 1.0)
            )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✚</div>
            <div>
                <h2>MedEvidence-AI</h2>
                <p>Evidence-based medical intelligence</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Medical AI Chat",
            "Evidence Explorer",
            "Model Evaluation",
            "About"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-status">
            <span class="status-dot"></span>
            RAG application
            <br>
            <small>Dense retrieval + Groq LLM</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("MedEvidence-AI | Academic Project")


# ==========================================
# LOAD BACKEND
# ==========================================

try:

    with st.spinner(
        "Loading medical knowledge base and AI models..."
    ):

        rag_system = load_rag_system()

    system_ready = True

except Exception as error:

    system_ready = False
    rag_system = None

    st.error(
        "Unable to initialize the RAG system. "
        "Please check your dataset, FAISS index, and model files."
    )

    with st.expander("Technical error"):
        st.exception(error)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">AI-POWERED MEDICAL KNOWLEDGE</div>
            <h1>Medical answers.<br>Grounded in evidence.</h1>
            <p>
                Explore medical information through semantic search,
                retrieved evidence, and AI-generated explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if system_ready:

        total_chunks = len(rag_system["chunks_df"])
        index_size = rag_system["index"].ntotal

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "Medical Knowledge",
                f"{total_chunks:,}",
                "Available evidence chunks",
                "📚"
            )

        with col2:
            metric_card(
                "Vector Index",
                f"{index_size:,}",
                "FAISS indexed records",
                "🔎"
            )

        with col3:
            metric_card(
                "Embedding Model",
                "MiniLM",
                "all-MiniLM-L6-v2",
                "🧠"
            )

        with col4:
            metric_card(
                "Answer Generation",
                "Groq",
                "GPT-OSS-20B",
                "⚡"
            )

    else:
        st.warning("The RAG backend is not ready.")

    st.write("")

    section_header(
        "Explore the platform",
        "A complete workflow from medical question to supporting evidence."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <h3>Medical AI Chat</h3>
                <p>
                    Ask medical questions and receive structured answers
                    supported by retrieved evidence and citations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Evidence Explorer</h3>
                <p>
                    Inspect retrieved medical passages, similarity
                    scores, and original source information.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Model Evaluation</h3>
                <p>
                    Compare retrieval approaches and review
                    preliminary answer evaluation results.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <h3>Evidence-first approach</h3>
                <p>
                    Responses are restricted to retrieved knowledge
                    and include citations to supporting sources.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.info(
        "Important: MedEvidence-AI is an academic medical "
        "information project. It does not diagnose conditions "
        "or replace professional medical advice."
    )


# ==========================================
# PAGE 2: MEDICAL AI CHAT
# ==========================================

elif page == "Medical AI Chat":

    section_header(
        "Medical AI Assistant",
        "Ask a question and explore an evidence-supported response."
    )

    if not system_ready:
        st.warning("The RAG system could not be loaded.")

    else:

        col1, col2 = st.columns([5, 1])

        with col1:
            st.caption("Your conversation")

        with col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.rerun()

        # Display previous messages

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message["role"] == "assistant":

                    show_sources(message.get("sources", []))

        # Suggested questions

        if not st.session_state.messages:

            st.markdown("#### Try asking")

            suggestions = [
                "What are the symptoms of asthma?",
                "What causes high blood pressure?",
                "What are the symptoms of anemia?"
            ]

            cols = st.columns(3)

            for index, suggestion in enumerate(suggestions):

                with cols[index]:

                    if st.button(
                        suggestion,
                        key=f"suggestion_{index}",
                        use_container_width=True
                    ):

                        st.session_state.pending_question = suggestion
                        st.rerun()

        question = st.chat_input(
            "Ask a medical question..."
        )

        if "pending_question" in st.session_state:

            question = st.session_state.pop("pending_question")

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching medical evidence and generating an answer..."
                ):

                    result = answer_question(
                        question=question,
                        rag_system=rag_system,
                        top_k=10
                    )

                st.markdown(result["answer"])

                show_sources(result["sources"])

                st.session_state.last_result = result

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    }
                )

            st.rerun()


# ==========================================
# PAGE 3: EVIDENCE EXPLORER
# ==========================================

elif page == "Evidence Explorer":

    section_header(
        "Evidence Explorer",
        "Inspect the medical evidence retrieved by the dense retrieval model."
    )

    if not system_ready:
        st.warning("The RAG system could not be loaded.")

    elif st.session_state.last_result is None:

        st.info(
            "No evidence to display yet. Ask a question in Medical AI Chat first."
        )

    else:

        result = st.session_state.last_result

        st.markdown("### Last searched question")

        st.write(result.get("answer", ""))

        st.divider()

        chunks = result.get("retrieved_chunks")

        if chunks is not None and not chunks.empty:

            col1, col2 = st.columns(2)

            with col1:
                metric_card(
                    "Retrieved Chunks",
                    len(chunks),
                    "Evidence selected for generation",
                    "🔍"
                )

            with col2:
                best_score = float(
                    chunks["retrieval_score"].max()
                )

                metric_card(
                    "Best Similarity Score",
                    f"{best_score:.4f}",
                    "Highest dense retrieval score",
                    "🎯"
                )

            show_retrieved_chunks(chunks)

        show_sources(result.get("sources", []))


# ==========================================
# PAGE 4: MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    section_header(
        "Model Evaluation",
        "Review retrieval benchmark results and human answer evaluation."
    )

    st.markdown("### Retrieval model comparison")

    st.caption(
        "Preliminary benchmark using a five-question manual test set. "
        "These results are not a large-scale evaluation."
    )

    retrieval_data = pd.DataFrame(
        {
            "Method": [
                "BM25",
                "Dense",
                "Hybrid",
                "TF-IDF"
            ],
            "Precision@5": [
                0.44,
                0.56,
                0.52,
                0.44
            ],
            "Recall@5": [
                0.6043,
                0.7329,
                0.6829,
                0.6157
            ],
            "MRR": [
                0.64,
                1.00,
                1.00,
                0.90
            ],
            "Hit Rate@5": [
                1.00,
                1.00,
                1.00,
                1.00
            ]
        }
    )

    st.dataframe(
        retrieval_data,
        use_container_width=True,
        hide_index=True
    )

    selected_metric = st.selectbox(
        "Choose a metric to compare",
        [
            "Precision@5",
            "Recall@5",
            "MRR",
            "Hit Rate@5"
        ]
    )

    fig = px.bar(
        retrieval_data,
        x="Method",
        y=selected_metric,
        color="Method",
        text_auto=".2f",
        title=f"{selected_metric} Comparison"
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis_range=[0, 1.1],
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Dense retrieval performed best on this preliminary benchmark. "
        "Dense + MedCPT reranking did not improve the measured metrics "
        "on the same small test set."
    )

    st.divider()

    st.markdown("### Human answer evaluation")

    evaluation_file = (
        EVALUATION_DIR / "answer_evaluation_results.csv"
    )

    if evaluation_file.exists():

        try:

            evaluation_df = pd.read_csv(evaluation_file)

            st.caption(
                "Results loaded from your local answer evaluation file."
            )

            st.dataframe(
                evaluation_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error("Unable to read the evaluation results file.")
            st.caption(str(error))

    else:

        st.warning(
            "No answer evaluation results file was found. "
            "Run your answer evaluation script first."
        )

    st.caption(
        "Human evaluation scores depend on the questions and ratings "
        "entered in your evaluation CSV."
    )


# ==========================================
# PAGE 5: ABOUT
# ==========================================

elif page == "About":

    section_header(
        "About MedEvidence-AI",
        "An evidence-based medical question-answering project using RAG."
    )

    st.markdown(
        """
        <div class="about-card">
            <h3>What is MedEvidence-AI?</h3>
            <p>
                MedEvidence-AI is a Retrieval-Augmented Generation
                application designed to answer medical information
                questions using a medical knowledge base.
            </p>
            <p>
                It retrieves relevant medical passages and provides
                them as context to a large language model, which
                generates a structured answer with evidence citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Technology stack")

    tech = pd.DataFrame(
        {
            "Component": [
                "Programming Language",
                "Dataset",
                "Embeddings",
                "Vector Search",
                "Language Model",
                "LLM API",
                "Frontend",
                "Evaluation"
            ],
            "Technology": [
                "Python",
                "MedQuAD",
                "all-MiniLM-L6-v2",
                "FAISS",
                "GPT-OSS-20B",
                "Groq",
                "Streamlit",
                "Precision, Recall, MRR, Human Evaluation"
            ]
        }
    )

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### How the system works")

    st.markdown(
        """
        <div class="workflow">
            <div class="workflow-step">
                <strong>01</strong>
                <h4>User Question</h4>
                <p>Medical question entered by the user</p>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <strong>02</strong>
                <h4>Dense Retrieval</h4>
                <p>FAISS retrieves relevant medical chunks</p>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <strong>03</strong>
                <h4>Relevance Filtering</h4>
                <p>Low-relevance evidence is filtered out</p>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <strong>04</strong>
                <h4>Answer Generation</h4>
                <p>Groq generates an evidence-grounded response</p>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <strong>05</strong>
                <h4>Answer + Citations</h4>
                <p>Answer and supporting source references</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Medical safety")

    st.warning(
        "This application provides educational medical information "
        "from its available knowledge base. It is not a medical "
        "diagnostic system. Always consult a qualified healthcare "
        "professional for diagnosis and treatment."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        <p>MedEvidence-AI</p>
        <span>Evidence-based medical information • Academic project</span>
    </div>
    """,
    unsafe_allow_html=True
)
