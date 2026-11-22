
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.workflow import build_knowledge_graph_from_document
from knowledge_graph.neo4j_store import Neo4jStore
from vector_store.chroma_store import ChromaVectorStore
from ingestion.pipeline import process_document
from ingestion.document_manager import DocumentManager
from ingestion.document_loader import SUPPORTED_EXTENSIONS, DEFAULT_MAX_FILE_SIZE


app = FastAPI(
    title="Enterprise Knowledge Graph Connector",
    description="Hybrid Graph RAG with enterprise document ingestion.",
    version="1.5.0"
)

pipeline = None

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS
MAX_FILE_SIZE = DEFAULT_MAX_FILE_SIZE
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "documents.sqlite3"


def get_document_manager():
    return DocumentManager(str(CATALOG_PATH))


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


def get_pipeline():
    global pipeline

    if pipeline is None:
        from rag.rag_pipeline import EnterpriseRAGPipeline
        pipeline = EnterpriseRAGPipeline()

    return pipeline


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
    original_filename = Path((file.filename or "unknown").replace("\\", "/")).name
    extension = Path(original_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")

    # Limit the read itself rather than buffering an arbitrarily large upload.
    content = await file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the 10 MB limit.")

    try:
        with TemporaryDirectory(dir=RAW_DIR) as temporary_dir:
            temporary_path = Path(temporary_dir) / original_filename
            temporary_path.write_bytes(content)
            document = process_document(str(temporary_path), max_file_size=MAX_FILE_SIZE)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    manager = get_document_manager()
    document_id = document["metadata"]["document_id"]
    try:
        existing = manager.get_document(document_id)
    except KeyError:
        existing = None
    if existing is not None and "indexing_result" in existing:
        return {**existing["indexing_result"], "duplicate": True,
                "message": "Document is already indexed."}

    file_path = RAW_DIR / f"{document_id}{extension}"
    vector_store = graph_store = None
    vector_attempted = graph_attempted = False
    try:
        file_path.write_bytes(content)
        document["metadata"]["file_path"] = str(file_path.resolve())
        for index, chunk in enumerate(document["chunks"]):
            chunk["chunk_id"] = f"{document_id}_chunk_{index}"
            chunk["metadata"].update(document["metadata"])
            chunk["metadata"]["original_filename"] = original_filename

        workflow_result = build_knowledge_graph_from_document(str(file_path))
        vector_store = ChromaVectorStore(collection_name="enterprise_documents")
        vector_attempted = True
        vectors_added = vector_store.add_documents(document["chunks"])
        graph_store = Neo4jStore()
        graph_store.verify_connection()
        graph_store.create_constraints()
        graph_attempted = True
        graph_stats = graph_store.store_graph(workflow_result["knowledge_graph"], document_id=document_id)
        response = {
            "message": "Document processed and indexed successfully.",
            "document_id": document_id,
            "filename": original_filename,
            "chunks_created": document["total_chunks"],
            "vectors_indexed": vectors_added,
            "graph_storage": graph_stats,
            "entities_extracted": len(workflow_result["entities"]),
            "relationships_extracted": len(workflow_result["relationships"]),
            "duplicate": False,
        }
        document["indexing_result"] = response
        manager.register(document, replace=existing is not None)
        return response
    except Exception as error:
        # Best-effort compensation: never publish failed indexing in the catalog.
        cleanup_errors = []
        for store, attempted, operation in [
            (graph_store, graph_attempted, "delete_document_data"),
            (vector_store, vector_attempted, "delete_document"),
        ]:
            if attempted:
                try:
                    getattr(store, operation)(document_id)
                except Exception as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
        if not cleanup_errors:
            file_path.unlink(missing_ok=True)
        detail = f"Document processing failed: {error}"
        if cleanup_errors:
            detail += f"; cleanup incomplete for {document_id}: {cleanup_errors}"
        raise HTTPException(status_code=500, detail=detail) from error
    finally:
        if vector_store is not None:
            vector_store.close()
        if graph_store is not None:
            graph_store.close()


@app.get("/documents")
def list_documents():
    vector_store = None

    try:
        vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )

        documents = vector_store.list_documents()
        manager = get_document_manager()
        for document in documents:
            try:
                record = manager.get_document(document["document_id"])
            except KeyError:
                continue  # Older uploads have no catalog entry.
            document["metadata"] = record["metadata"]

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
            # Allow cleanup retries if vector removal succeeded previously.
            try:
                record = get_document_manager().get_document(document_id)
                document = {"filename": record["metadata"]["source"]}
            except KeyError:
                raise HTTPException(status_code=404, detail="Document not found.")

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

        get_document_manager().delete_document(document_id)

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
    new_filename = Path((file.filename or "unknown").replace("\\", "/")).name

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

    if new_document["duplicate"] and new_document["filename"].casefold() != new_filename.casefold():
        raise HTTPException(status_code=409, detail="Replacement content belongs to another document.")

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

@app.get("/graph")
def get_knowledge_graph(limit: int = 200):
    try:
        active_pipeline = get_pipeline()

        if active_pipeline is None:
            return {"nodes": [], "edges": []}

        graph_store = active_pipeline.retriever.graph_store
        graph_store.verify_connection()

        return graph_store.get_graph_snapshot(limit=limit)

    except Exception as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail=f"Unable to retrieve knowledge graph: {exc}"
        ) from exc

@app.on_event("shutdown")
def shutdown():
    global pipeline

    if pipeline is not None:
        if hasattr(pipeline, "close"):
            pipeline.close()

        pipeline = None
