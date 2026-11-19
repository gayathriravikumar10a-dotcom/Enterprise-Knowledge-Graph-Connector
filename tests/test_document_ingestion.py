"""Member 2 reliability checks using locally generated upload fixtures."""
from concurrent.futures import ThreadPoolExecutor
import hashlib

from docx import Document
import pymupdf
import pytest

from ingestion.document_loader import clean_text, load_document
from ingestion.chunker import chunk_spans, chunk_text
from ingestion.document_manager import DocumentManager
from ingestion.pipeline import process_document


def write_document(path, text):
    if path.suffix.lower() == ".pdf":
        with pymupdf.open() as pdf:
            # Bound page content so generated large PDFs do not silently clip.
            words = text.split()
            for start in range(0, max(len(words), 1), 100):
                page = pdf.new_page()
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 780),
                                    " ".join(words[start:start + 100]), fontsize=10)
            pdf.save(path)
    elif path.suffix.lower() == ".docx":
        doc = Document()
        doc.add_paragraph(text)
        doc.save(path)
    else:
        path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("extension", ["txt", "docx", "pdf"])
@pytest.mark.parametrize("word_count", [1, 200, 3000])
def test_formats_and_sizes(tmp_path, extension, word_count):
    path = write_document(tmp_path / f"sample.{extension}", " ".join(["evidence"] * word_count))
    result = process_document(str(path), chunk_size=80, overlap=10)
    metadata = result["metadata"]
    assert metadata["word_count"] == word_count
    assert metadata["file_size_bytes"] == path.stat().st_size
    assert metadata["file_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["source"] == path.name
    assert metadata["file_type"] == extension
    assert metadata["modified_at"].endswith("+00:00")
    words = load_document(str(path))["text"].split()
    for chunk in result["chunks"]:
        meta = chunk["metadata"]
        assert chunk["text"].split() == words[meta["word_start"]:meta["word_end"]]
        assert len(chunk["text"].split()) <= 80
    assert result["chunks"][-1]["metadata"]["word_end"] == word_count


@pytest.mark.parametrize("extension", ["pdf", "docx", "txt"])
def test_empty_and_corrupt_documents(tmp_path, extension):
    path = tmp_path / f"bad.{extension}"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_document(str(path))
    path.write_bytes(b"\x00\xffinvalid document")
    with pytest.raises(ValueError, match="Unable to read"):
        load_document(str(path))
    write_document(path, "   ")
    with pytest.raises(ValueError, match="No readable text"):
        load_document(str(path))


def test_validation_and_encoding(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="regular file"):
        load_document(str(tmp_path))
    path = tmp_path / "file.csv"
    path.write_text("data")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(str(path))
    path = tmp_path / "file.TXT"
    for encoding in ["utf-8-sig", "utf-16"]:
        path.write_bytes("Café evidence".encode(encoding))
        assert load_document(str(path))["text"] == "Café evidence"
    with pytest.raises(ValueError, match="size limit"):
        load_document(str(path), max_file_size=1)
    path.write_bytes(b"binary\x00text")
    with pytest.raises(ValueError, match="control"):
        load_document(str(path))
    assert clean_text("Cafe\u0301\u200b\ufeff\x00\t  evidence\n") == "Café evidence"


def test_docx_table_order(tmp_path):
    path = tmp_path / "table.docx"
    doc = Document()
    doc.add_paragraph("before")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "inside"
    doc.add_paragraph("after")
    doc.save(path)
    assert load_document(str(path))["text"] == "before inside after"


def test_encrypted_pdf(tmp_path):
    path = tmp_path / "locked.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), "secret")
        pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                 owner_pw="owner", user_pw="reader")
    with pytest.raises(ValueError, match="Password-protected"):
        load_document(str(path))


@pytest.mark.parametrize("overlap", [0, 1, 19])
def test_chunk_coverage_and_progress(overlap):
    words = ("First sentence ends. More evidence follows! " * 1000).split()
    spans = chunk_spans(" ".join(words), 20, overlap)
    assert spans[0]["word_start"] == 0
    assert spans[-1]["word_end"] == len(words)
    for previous, current in zip(spans, spans[1:]):
        assert current["word_start"] == previous["word_end"] - overlap
        assert current["word_start"] > previous["word_start"]
    assert all(s["text"].split() == words[s["word_start"]:s["word_end"]] for s in spans)


@pytest.mark.parametrize("options", [{"chunk_size": 0}, {"chunk_size": 1.5},
    {"overlap": -1}, {"overlap": 500}, {"overlap": True}])
def test_invalid_chunk_options(options):
    with pytest.raises(ValueError):
        chunk_text("some words", **options)


def test_catalog_duplicates_persistence_and_management(tmp_path):
    manager = DocumentManager(str(tmp_path / "catalog.sqlite3"))
    first = write_document(tmp_path / "first.txt", "Shared   evidence")
    renamed = write_document(tmp_path / "renamed.docx", "Shared evidence")
    result = manager.ingest(str(first))
    document_id = result["document"]["metadata"]["document_id"]
    assert not result["duplicate"]
    assert manager.ingest(str(renamed))["duplicate"]
    assert manager.list_documents()[0]["source"] == "first.txt"
    second = write_document(tmp_path / "second.docx", "Different evidence")
    manager.ingest(str(second))
    reopened = DocumentManager(str(tmp_path / "catalog.sqlite3"))
    assert reopened.get_document(document_id) == result["document"]
    assert len(reopened.list_documents()) == 2
    assert len(reopened.list_documents(file_type=".DOCX")) == 1
    assert len(reopened.list_documents(limit=1, offset=1)) == 1
    assert reopened.delete_document(document_id)
    assert not reopened.delete_document(document_id)
    assert first.exists()
    with pytest.raises(KeyError):
        reopened.get_document(document_id)
    assert len(reopened.list_documents()) == 1


def test_failed_ingestion_and_concurrent_duplicates(tmp_path):
    manager = DocumentManager(str(tmp_path / "catalog.sqlite3"))
    path = tmp_path / "empty.txt"
    path.touch()
    with pytest.raises(ValueError):
        manager.ingest(str(path))
    assert manager.list_documents() == []
    path.write_text("Concurrent document evidence")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: manager.ingest(str(path)), range(4)))
    assert sum(not result["duplicate"] for result in results) == 1
    assert len(manager.list_documents()) == 1


@pytest.mark.parametrize("options", [{"limit": 0}, {"offset": -1}, {"limit": True}])
def test_catalog_invalid_pagination(tmp_path, options):
    manager = DocumentManager(str(tmp_path / "catalog.sqlite3"))
    with pytest.raises(ValueError):
        manager.list_documents(**options)


def test_large_txt_and_exact_size_limit(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("evidence " * 100_000)
    size = path.stat().st_size
    result = process_document(str(path), max_file_size=size)
    assert result["metadata"]["word_count"] == 100_000
    assert result["chunks"][-1]["metadata"]["word_end"] == 100_000
    with pytest.raises(ValueError, match="size limit"):
        load_document(str(path), max_file_size=size - 1)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_size_configuration(tmp_path, limit):
    path = write_document(tmp_path / "sample.txt", "evidence")
    with pytest.raises(ValueError, match="positive integer"):
        load_document(str(path), max_file_size=limit)


def test_catalog_connections_close_after_success_and_failure(tmp_path, monkeypatch):
    import sqlite3
    from ingestion import document_manager

    connections = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(document_manager.sqlite3, "connect", tracked_connect)
    manager = DocumentManager(str(tmp_path / "catalog.sqlite3"))
    manager.list_documents()
    with pytest.raises(sqlite3.OperationalError):
        with manager._connect() as connection:
            connection.execute("SELECT * FROM missing_table")
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
