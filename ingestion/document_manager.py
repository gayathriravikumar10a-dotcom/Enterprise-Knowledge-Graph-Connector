"""Persistent processed-document catalog, independent of graph/vector services.

Records are deduplicated by SHA-256 of normalized text, including across file
names and formats. The first ingestion owns canonical metadata and chunking.
This catalog stores extracted text/chunks, not source files; deletion removes
only its own record. Backend index cleanup remains the backend's responsibility.
"""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from ingestion.pipeline import process_document


class DocumentManager:
    """A local SQLite catalog with atomic duplicate detection.

    Example::

        manager = DocumentManager('data/processed/documents.sqlite3')
        result = manager.ingest('data/sample_documents/company_info.txt')
        documents = manager.list_documents(limit=20)
        document = manager.get_document(result['document']['metadata']['document_id'])

    Each operation closes its connection. Separate instances may safely share
    the same database. Failed extraction never creates a catalog record.
    """

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                file_type TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )''')

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def ingest(self, file_path: str, **processing_options) -> dict:
        return self.register(process_document(file_path, **processing_options))

    def register(self, processed_document: dict, *, replace: bool = False) -> dict:
        """Persist processed data; replace=True explicitly updates an existing record."""
        document = deepcopy(processed_document)
        metadata = document["metadata"]
        metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
        for chunk in document["chunks"]:
            chunk["metadata"]["ingested_at"] = metadata["ingested_at"]
        conflict = (
            "DO UPDATE SET source=excluded.source, file_type=excluded.file_type, "
            "ingested_at=excluded.ingested_at, payload=excluded.payload"
            if replace else "DO NOTHING"
        )
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?) ON CONFLICT(document_id) " + conflict,
                (metadata["document_id"], metadata["source"], metadata["file_type"],
                 metadata["ingested_at"], json.dumps(document, ensure_ascii=False)),
            )
            duplicate = cursor.rowcount == 0
            if duplicate:
                document = json.loads(connection.execute(
                    "SELECT payload FROM documents WHERE document_id = ?",
                    (metadata["document_id"],),
                ).fetchone()[0])
        return {"duplicate": duplicate, "document": document}

    def get_document(self, document_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM documents WHERE document_id = ?",
                                     (document_id,)).fetchone()
        if row is None:
            raise KeyError(f"Document not found: {document_id}")
        return json.loads(row[0])

    def list_documents(self, *, file_type: str | None = None,
                       limit: int = 100, offset: int = 0) -> list:
        """List metadata in ingestion order with optional type filter and paging."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        query = "SELECT payload FROM documents"
        params = []
        if file_type is not None:
            query += " WHERE file_type = ?"
            params.append(file_type.lower().lstrip("."))
        query += " ORDER BY ingested_at, document_id LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [{**record["metadata"], "total_chunks": record["total_chunks"]}
                for (payload,) in rows for record in [json.loads(payload)]]

    def delete_document(self, document_id: str) -> bool:
        """Remove this catalog record; leave original files and other indexes intact."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0
