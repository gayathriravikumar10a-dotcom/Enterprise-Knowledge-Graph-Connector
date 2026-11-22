
import streamlit as st
import streamlit.components.v1 as components
import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.visualizer import render_knowledge_graph_html

st.set_page_config(
    page_title="Enterprise Knowledge Graph Connector",
    page_icon="🤖",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

st.title("Enterprise Knowledge Graph Connector")
st.caption(
    "Ask questions across enterprise documents using "
    "Hybrid Graph RAG and grounded AI answers."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("System Information")
    st.write("Architecture: Hybrid Graph + Vector RAG")
    st.write("Knowledge Graph: Neo4j")
    st.write("Vector Database: ChromaDB")
    st.write("LLM: Gemini")

    st.divider()

    API_URL = st.text_input(
        "Backend API URL",
        value=API_URL
    ).rstrip("/")

    st.info("Make sure the FastAPI backend is running.")

# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.header("1. Upload Enterprise Documents")

st.write(
    "Upload a document to extract entities, identify relationships, "
    "create knowledge graph connections, and index its content."
)

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX and TXT. Maximum size: 10 MB."
)

if uploaded_file is not None:

    st.write(f"Selected file: **{uploaded_file.name}**")
    st.write(
        f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    if st.button("Upload and Process Document", type="primary"):

        if uploaded_file.size == 0:
            st.warning("The selected file is empty.")

        elif uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size exceeds the 10 MB limit.")

        else:
            with st.spinner(
                "Processing document, extracting knowledge and indexing..."
            ):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        },
                        timeout=300
                    )

                    response.raise_for_status()
                    result = response.json()

                    st.success(result["message"])

                    st.subheader("Document Processing Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Chunks Created",
                            result["chunks_created"]
                        )

                    with col2:
                        st.metric(
                            "Vectors Indexed",
                            result["vectors_indexed"]
                        )

                    with col3:
                        st.metric(
                            "Entities Extracted",
                            result["entities_extracted"]
                        )

                    col4, col5 = st.columns(2)

                    with col4:
                        st.metric(
                            "Relationships Extracted",
                            result["relationships_extracted"]
                        )

                    with col5:
                        st.metric(
                            "Graph Entities Stored",
                            result["graph_storage"]["entities_stored"]
                        )

                    st.metric(
                        "Graph Relationships Stored",
                        result["graph_storage"]["relationships_stored"]
                    )

                    with st.expander("Document processing details"):
                        st.json(result)

                    st.info(
                        "Your document has been processed. "
                        "You can now ask questions about its content below."
                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to FastAPI. "
                        "Start the backend and try again."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "Document processing timed out. "
                        "Check the backend terminal before retrying."
                    )

                except requests.exceptions.HTTPError:
                    try:
                        error_detail = response.json().get(
                            "detail",
                            response.text
                        )
                    except Exception:
                        error_detail = response.text

                    st.error(f"Upload failed: {error_detail}")

                except Exception as error:
                    st.error(f"Unexpected error: {error}")

st.divider()

# --------------------------------------------------
# ASK QUESTIONS
# --------------------------------------------------

st.header("2. Ask Your Enterprise Knowledge Base")

with st.form("question_form"):
    question = st.text_area(
        "Enter your question",
        placeholder="Example: Who manages Project Atlas?",
        height=100
    )

    submitted = st.form_submit_button(
        "Generate Answer",
        type="primary"
    )

if submitted:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner(
            "Searching documents and knowledge graph..."
        ):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question.strip()},
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

                st.divider()
                st.subheader("Answer")

                if result["answer_status"] == "grounded":
                    st.success(result["answer"])
                    st.success(
                        "Answer verified against retrieved evidence."
                    )
                else:
                    st.warning(result["answer"])
                    st.warning("Insufficient grounding")

                st.subheader("Grounding Evaluation")

                evaluation = result["grounding_evaluation"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Evidence Support Ratio",
                        f'{evaluation["support_ratio"] * 100:.2f}%'
                    )

                with col2:
                    st.metric(
                        "Grounding Status",
                        "Grounded" if evaluation["grounded"]
                        else "Insufficient"
                    )

                with st.expander("Grounding details"):
                    st.write("Supported terms")
                    st.write(evaluation["supported_terms"])

                    st.write("Unsupported terms")
                    st.write(evaluation["unsupported_terms"])

                st.subheader("Source Information")

                sources = result["sources"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Document Chunks",
                        sources["document_chunks"]
                    )

                with col2:
                    st.metric(
                        "Graph Relationships",
                        sources["graph_relationships"]
                    )

                st.caption(f'Model: {result["model"]}')

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Start the backend on port 8000 and try again."
                )

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")

            except requests.exceptions.HTTPError as error:
                st.error(f"Backend error: {error}")

            except Exception as error:
                st.error(f"Unexpected error: {error}")

st.divider()

st.divider()

# --------------------------------------------------
# KNOWLEDGE GRAPH VISUALIZATION
# --------------------------------------------------

st.header("3. Explore Knowledge Graph")

st.write(
    "Inspect resolved enterprise entities, aliases, and multi-hop relationships."
)

if st.button("Refresh Knowledge Graph"):
    try:
        graph_response = requests.get(
            f"{API_URL}/graph",
            timeout=30
        )
        graph_response.raise_for_status()

        graph_data = graph_response.json()

        if not isinstance(graph_data, dict):
            raise ValueError("Invalid graph response format.")

        if "nodes" not in graph_data or "edges" not in graph_data:
            raise ValueError("Graph response is missing nodes or edges.")

        st.session_state["graph_snapshot"] = graph_data
        st.session_state["graph_error"] = None

    except Exception as graph_err:
        st.session_state["graph_error"] = str(graph_err)

snapshot = st.session_state.get(
    "graph_snapshot",
    {"nodes": [], "edges": []}
)

if st.session_state.get("graph_error"):
    st.error(
        f"Unable to load live Neo4j graph: "
        f"{st.session_state['graph_error']}"
    )

elif not snapshot.get("nodes"):
    st.info(
        "No graph data is currently available. "
        "Upload and index a document, then refresh."
    )

else:
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Graph Entities", len(snapshot["nodes"]))

    with col2:
        st.metric("Graph Relationships", len(snapshot["edges"]))

    st.success("Live Neo4j graph data loaded successfully.")

    components.html(
        render_knowledge_graph_html(
            snapshot["nodes"],
            snapshot["edges"],
            height=520
        ),
        height=530,
    )