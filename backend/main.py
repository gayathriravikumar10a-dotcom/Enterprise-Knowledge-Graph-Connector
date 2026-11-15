
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
    version="1.5.0"
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
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {error}"
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    original_filename = Path(file.filename or "unknown").name
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
                    f"Unsupported chunk type: {type(chunk).__name__}"
                )

            if not isinstance(chunk_text, str) or not chunk_text.strip():
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
            raise ValueError("No readable text chunks were extracted.")

        vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )

        try:
            vectors_added = vector_store.add_documents(indexed_chunks)
        finally:
            vector_store.close()

        graph_store = Neo4jStore()

        try:
            graph_store.verify_connection()
            graph_store.create_constraints()

            graph_stats = graph_store.store_graph(
                knowledge_graph,
                document_id=document_id
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
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}"
        )


@app.get("/documents")
def list_documents():
    vector_store = None

    try:
        vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )

        documents = vector_store.list_documents()

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not list documents: {error}"
        )

    finally:
        if vector_store is not None:
            vector_store.close()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    vector_store = None
    graph_store = None

    try:
        vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )

        documents = vector_store.list_documents()

        document = next(
            (
                item for item in documents
                if item["document_id"] == document_id
            ),
            None
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found in the vector index."
            )

        # Remove graph provenance first. Neo4j preserves graph
        # records that are still supported by other documents.
        graph_store = Neo4jStore()
        graph_store.verify_connection()

        graph_result = graph_store.delete_document_data(document_id)

        # Delete only this document's vector chunks.
        chunks_deleted = vector_store.delete_document(document_id)

        # Delete the stored original file after database cleanup.
        deleted_files = []

        for extension in ALLOWED_EXTENSIONS:
            stored_file = RAW_DIR / f"{document_id}{extension}"

            if stored_file.is_file():
                stored_file.unlink()
                deleted_files.append(stored_file.name)

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
            "filename": document["filename"],
            "chunks_deleted": chunks_deleted,
            "graph_cleanup": graph_result,
            "files_deleted": deleted_files
        }

    except HTTPException:
        raise

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {error}"
        )

    finally:
        if vector_store is not None:
            vector_store.close()

        if graph_store is not None:
            graph_store.close()


@app.put("/documents/{document_id}")
async def update_document(
    document_id: str,
    file: UploadFile = File(...)
):
    vector_store = None

    try:
        # Find the existing document.
        vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )

        documents = vector_store.list_documents()

        existing_document = next(
            (
                item for item in documents
                if item["document_id"] == document_id
            ),
            None
        )

        if existing_document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

    finally:
        if vector_store is not None:
            vector_store.close()

    # Only allow replacement using the same filename.
    new_filename = Path(file.filename or "unknown").name

    if new_filename.casefold() != existing_document["filename"].casefold():
        raise HTTPException(
            status_code=400,
            detail="Replacement filename must match the existing filename."
        )

    # Index the new version first.
    new_document = await upload_document(file)

    new_document_id = new_document["document_id"]

    # Same content means no replacement is necessary.
    if new_document_id == document_id:
        return {
            "message": "Document is already up to date.",
            "document_id": document_id,
            "filename": new_filename
        }

    # Remove the old version only after successful indexing.
    try:
        old_document_cleanup = delete_document(document_id)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "New version was indexed, but old version cleanup failed.",
                "new_document_id": new_document_id,
                "old_document_id": document_id,
                "cleanup_error": str(error)
            }
        )

    return {
        "message": "Document updated and re-indexed successfully.",
        "old_document_id": document_id,
        "new_document_id": new_document_id,
        "filename": new_filename,
        "new_chunks_indexed": new_document["chunks_created"],
        "old_chunks_deleted": old_document_cleanup["chunks_deleted"],
        "old_graph_cleanup": old_document_cleanup["graph_cleanup"]
    }

@app.on_event("shutdown")
def shutdown():
    global pipeline

    if pipeline is not None:
        if hasattr(pipeline, "close"):
            pipeline.close()

        pipeline = None
