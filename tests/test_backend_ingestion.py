"""HTTP integration with real ingestion/SQLite and isolated external indexes."""
from io import BytesIO
from types import SimpleNamespace

from docx import Document
from fastapi.testclient import TestClient
import pymupdf
import pytest

from backend import main


@pytest.fixture
def api(tmp_path, monkeypatch):
    state = SimpleNamespace(vectors={}, graphs={}, failure=None, closed=0)

    class VectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, chunks):
            document_id = chunks[0]["metadata"]["document_id"]
            state.vectors[document_id] = chunks
            if state.failure == "vector":
                raise RuntimeError("vector write failed")
            return len(chunks)

        def list_documents(self):
            return [{"document_id": key,
                     "filename": chunks[0]["metadata"]["original_filename"],
                     "chunks_indexed": len(chunks)} for key, chunks in state.vectors.items()]

        def delete_document(self, document_id):
            return len(state.vectors.pop(document_id, []))

        def close(self):
            state.closed += 1

    class GraphStore:
        def verify_connection(self):
            pass

        def create_constraints(self):
            pass

        def store_graph(self, graph, document_id):
            state.graphs[document_id] = graph
            if state.failure == "graph":
                raise RuntimeError("graph write failed")
            return {"nodes": 1}

        def delete_document_data(self, document_id):
            state.graphs.pop(document_id, None)
            return {"deleted": True}

        def close(self):
            state.closed += 1

    monkeypatch.setattr(main, "RAW_DIR", tmp_path / "raw")
    main.RAW_DIR.mkdir()
    monkeypatch.setattr(main, "CATALOG_PATH", tmp_path / "catalog.sqlite3")
    monkeypatch.setattr(main, "ChromaVectorStore", VectorStore)
    monkeypatch.setattr(main, "Neo4jStore", GraphStore)
    monkeypatch.setattr(main, "build_knowledge_graph_from_document", lambda path: {
        "knowledge_graph": "graph", "entities": ["Asha"], "relationships": []})
    with TestClient(main.app) as client:
        yield client, state


def upload(client, filename="sample.txt", content=b"Asha manages Project Atlas."):
    return client.post("/upload", files={"file": (filename, content)})


@pytest.mark.parametrize("extension", ["txt", "pdf", "docx"])
def test_upload_listing_and_delete(api, extension):
    client, state = api
    content = b"Asha manages Project Atlas."
    if extension == "pdf":
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 50), content.decode())
            content = pdf.tobytes()
    elif extension == "docx":
        stream = BytesIO()
        doc = Document()
        doc.add_paragraph(content.decode())
        doc.save(stream)
        content = stream.getvalue()
    response = upload(client, f"sample.{extension}", content)
    assert response.status_code == 200, response.text
    result = response.json()
    document_id = result["document_id"]
    assert not result["duplicate"]
    chunks = state.vectors[document_id]
    assert chunks[0]["metadata"]["source"] == f"sample.{extension}"
    assert chunks[0]["metadata"]["word_start"] == 0
    listing = client.get("/documents").json()
    assert listing["total_documents"] == 1
    assert listing["documents"][0]["metadata"]["file_size_bytes"] == len(content)
    assert main.get_document_manager().get_document(document_id)["total_chunks"] == len(chunks)
    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert not state.vectors and not state.graphs
    assert main.get_document_manager().list_documents() == []
    assert list(main.RAW_DIR.iterdir()) == []
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_duplicates_use_content_not_filename(api):
    client, state = api
    first = upload(client).json()
    second = upload(client, "renamed.txt", b"Asha  manages\nProject Atlas.").json()
    assert second["duplicate"]
    assert second["document_id"] == first["document_id"]
    assert second["filename"] == "sample.txt"
    assert len(state.vectors) == 1
    assert len(list(main.RAW_DIR.iterdir())) == 1


@pytest.mark.parametrize("filename,content,status", [
    ("empty.txt", b"", 400), ("blank.txt", b"  \n", 400),
    ("bad.pdf", b"not a pdf", 400), ("bad.docx", b"not a docx", 400),
    ("bad.txt", b"binary\x00data", 400), ("bad.csv", b"data", 400),
])
def test_invalid_uploads_leave_no_records(api, filename, content, status):
    client, state = api
    assert upload(client, filename, content).status_code == status
    assert not state.vectors and not state.graphs
    assert main.get_document_manager().list_documents() == []
    assert list(main.RAW_DIR.iterdir()) == []


def test_upload_size_limit(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 5)
    assert upload(client, content=b"123456").status_code == 413
    assert upload(client, content=b"12345").status_code == 200


@pytest.mark.parametrize("failure", ["vector", "graph", "catalog"])
def test_failed_indexing_is_compensated_and_retryable(api, monkeypatch, failure):
    client, state = api
    if failure == "catalog":
        manager = main.get_document_manager()
        original_register = manager.register
        monkeypatch.setattr(main, "get_document_manager", lambda: manager)
        def failed_register(document, **kwargs):
            raise RuntimeError("catalog unavailable")
        monkeypatch.setattr(manager, "register", failed_register)
    else:
        state.failure = failure
    response = upload(client)
    assert response.status_code == 500
    assert not state.vectors and not state.graphs
    assert main.get_document_manager().list_documents() == []
    assert list(main.RAW_DIR.iterdir()) == []
    assert state.closed >= 1
    state.failure = None
    if failure == "catalog":
        monkeypatch.setattr(manager, "register", original_register)
    assert upload(client).status_code == 200


def test_update_indexes_new_version_then_removes_old(api):
    client, state = api
    old_id = upload(client).json()["document_id"]
    unchanged = client.put(f"/documents/{old_id}", files={"file": ("sample.txt", b"Asha manages Project Atlas.")})
    assert unchanged.status_code == 200
    assert "up to date" in unchanged.json()["message"]
    mismatch = client.put(f"/documents/{old_id}", files={"file": ("other.txt", b"New text")})
    assert mismatch.status_code == 400
    updated = client.put(f"/documents/{old_id}", files={"file": ("sample.txt", b"New document evidence.")})
    assert updated.status_code == 200, updated.text
    assert old_id not in state.vectors
    assert len(state.vectors) == 1
    assert len(main.get_document_manager().list_documents()) == 1


def test_failed_update_preserves_old_document(api):
    client, state = api
    old_id = upload(client).json()["document_id"]
    state.failure = "graph"
    result = client.put(f"/documents/{old_id}", files={"file": ("sample.txt", b"New document evidence.")})
    assert result.status_code == 500
    assert list(state.vectors) == [old_id]
    assert list(state.graphs) == [old_id]
    assert len(main.get_document_manager().list_documents()) == 1


def test_legacy_vector_documents_remain_manageable(api):
    client, state = api
    state.vectors["legacy"] = [{"metadata": {"original_filename": "legacy.txt"}}]
    assert client.get("/documents").json()["documents"][0]["filename"] == "legacy.txt"
    assert client.delete("/documents/legacy").status_code == 200


def test_preprocessed_catalog_document_can_be_indexed(api):
    client, state = api
    source = main.RAW_DIR.parent / "local.txt"
    source.write_text("Asha manages Project Atlas.")
    main.get_document_manager().ingest(str(source))
    first = upload(client)
    assert first.status_code == 200, first.text
    assert not first.json()["duplicate"]
    assert upload(client).json()["duplicate"]
    assert len(state.vectors) == 1


def test_deletion_retry_after_vectors_already_removed(api):
    client, state = api
    document_id = upload(client).json()["document_id"]
    state.vectors.clear()
    assert client.delete(f"/documents/{document_id}").status_code == 200
    assert not state.graphs
    assert main.get_document_manager().list_documents() == []


def test_replacement_cannot_take_over_another_documents_content(api):
    client, state = api
    first_id = upload(client).json()["document_id"]
    second_id = upload(client, "second.txt", b"Other document evidence.").json()["document_id"]
    response = client.put(f"/documents/{first_id}", files={"file": ("sample.txt", b"Other document evidence.")})
    assert response.status_code == 409
    assert set(state.vectors) == {first_id, second_id}
