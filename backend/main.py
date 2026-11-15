
import sys
import hashlib
import traceback
from pathlib import Path
from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_pipeline import EnterpriseRAGPipeline
from knowledge_graph.workflow import build_knowledge_graph_from_document
from knowledge_graph.neo4j_store import Neo4jStore
from vector_store.chroma_store import ChromaVectorStore


app = FastAPI(
    title="Enterprise Knowledge Graph Connector",
    description="Hybrid Graph RAG with enterprise document ingestion.",
    version="1.4.0"
)

pipeline = None

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


def get_pipeline():
    global pipeline

    if pipeline is None:
        pipeline = EnterpriseRAGPipeline()

    return pipeline


def generate_document_id(filename, content):
    """
    Generate a stable document ID.

    The same filename and file content always produce
    the same ID.
    """

    normalized_filename = Path(filename).name.lower()

    hasher = hashlib.sha256()
    hasher.update(normalized_filename.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(content)

    return hasher.hexdigest()


@app.get("/")
def home():
    return {
        "message": "Enterprise Knowledge Graph Connector API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Enterprise Knowledge Graph Connector"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        result = get_pipeline().ask(request.question)

        return {
            "question": result["query"],
            "answer": result["answer"],
            "answer_status": result["answer_status"],
            "grounding_evaluation": result["grounding_evaluation"],
            "sources": result["sources"],
            "model": result["model"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("\n========== COMPLETE ASK ERROR TRACEBACK ==========")
        traceback.print_exc()
        print("==================================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {error}"
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    original_filename = Path(
        file.filename or "unknown"
    ).name

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files are supported."
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB limit."
            )

        # Stable ID prevents duplicate indexing of the
        # same file uploaded repeatedly.
        document_id = generate_document_id(
            original_filename,
            content
        )

        file_path = RAW_DIR / f"{document_id}{extension}"

        file_path.write_bytes(content)

        workflow_result = build_knowledge_graph_from_document(
            str(file_path)
        )

        chunks = workflow_result["chunks"]
        knowledge_graph = workflow_result["knowledge_graph"]

        indexed_chunks = []

        for index, chunk in enumerate(chunks):

            if isinstance(chunk, str):
                chunk_text = chunk
                metadata = {}

            elif isinstance(chunk, Mapping):
                chunk_text = chunk.get("text", "")
                raw_metadata = chunk.get("metadata", {})

                metadata = (
                    dict(raw_metadata)
                    if isinstance(raw_metadata, Mapping)
                    else {}
                )

            else:
                raise TypeError(
                    f"Unsupported chunk format at index {index}: "
                    f"{type(chunk).__name__}"
                )

            if (
                not isinstance(chunk_text, str)
                or not chunk_text.strip()
            ):
                continue

            metadata.update({
                "document_id": document_id,
                "original_filename": original_filename,
                "source": original_filename
            })

            indexed_chunks.append({
                "chunk_id": f"{document_id}_chunk_{index}",
                "text": chunk_text.strip(),
                "metadata": metadata
            })

        if not indexed_chunks:
            raise ValueError(
                "No readable text chunks were extracted."
            )

        vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )

        try:
            vectors_added = vector_store.add_documents(
                indexed_chunks
            )
        finally:
            if hasattr(vector_store, "close"):
                vector_store.close()

        graph_store = Neo4jStore()

        try:
            graph_store.verify_connection()
            graph_store.create_constraints()

            graph_stats = graph_store.store_graph(
                knowledge_graph
            )

        finally:
            graph_store.close()

        return {
            "message": "Document processed and indexed successfully.",
            "document_id": document_id,
            "filename": original_filename,
            "chunks_created": len(indexed_chunks),
            "vectors_indexed": vectors_added,
            "graph_storage": graph_stats,
            "entities_extracted": len(workflow_result["entities"]),
            "relationships_extracted": len(
                workflow_result["relationships"]
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        print("\n========== COMPLETE UPLOAD ERROR TRACEBACK ==========")
        traceback.print_exc()
        print("======================================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}"
        )


@app.on_event("shutdown")
def shutdown():
    global pipeline

    if pipeline is not None:
        if hasattr(pipeline, "close"):
            pipeline.close()

        pipeline = None
