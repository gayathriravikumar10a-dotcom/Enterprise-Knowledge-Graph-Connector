# Document ingestion and management — Member 2

Member 2's contribution is a reliable document ingestion and management module:
validated PDF, DOCX, and TXT extraction, text cleaning, metadata, predictable
chunking, duplicate detection, and persistent document management. The module
is connected to the backend's upload, listing, replacement, and deletion flows.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `document_loader.py` | Validate files, extract and clean text, calculate hashes and metadata |
| `chunker.py` | Split text into bounded, overlapping word chunks and record offsets |
| `pipeline.py` | Combine loading and chunking into a structured processing result |
| `document_manager.py` | Persist processed documents in SQLite; deduplicate, list, retrieve, and delete records |
| `../backend/main.py` | Connect ingestion and the catalog to graph/vector indexing and HTTP endpoints |
| `../tests/test_document_ingestion.py` | Format, metadata, chunking, catalog, and reliability tests |
| `../tests/test_backend_ingestion.py` | HTTP integration tests with isolated external indexes |

Existing loader and processing return fields are preserved. `chunk_text()` still
returns a list of strings, and local processing retains source-based chunk IDs.
The optional sentence-boundary heuristic was removed; word-based chunking is the
single supported strategy.

## Environment and setup

Run commands from the repository root. The current local environment uses Python
3.12; `hashlib.file_digest()` in the loader requires Python 3.11 or newer. Install
the repository dependencies and the spaCy model used by the graph workflow:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m spacy download en_core_web_sm
```

The dependency list includes `python-multipart`, required for FastAPI file uploads.
SQLite is provided by Python's standard library. Local loading, chunking, and
catalog operations do not require Neo4j, Gemini, embeddings, or vector storage.

For this checkout, the `AI:DS project` parent directory contains `:`, which Python
rejects when creating a virtual environment there. The repository's ignored
`venv` link points to `/private/tmp/enterprise-knowledge-graph-connector-venv`.
If that temporary environment is removed, recreate it outside the colon-containing
path and restore the link before reinstalling dependencies. This is a local setup
workaround rather than a module requirement.

## Loading, cleaning, and validation

```python
from ingestion.document_loader import load_document

loaded = load_document("data/sample_documents/company_info.txt")
print(loaded["text"])
print(loaded["metadata"])
```

| Format | Supported extraction |
| --- | --- |
| PDF | Selectable text from all pages; PDFs requiring a password are rejected |
| DOCX | Body paragraphs and table cells, preserving paragraph/table order |
| TXT | UTF-8 with or without a BOM, or BOM-marked UTF-16 |

Extensions are case-insensitive. Cleaning normalizes Unicode to NFC, removes
non-whitespace control and format characters, and collapses whitespace. TXT input
with non-whitespace control characters is rejected before cleaning to avoid
silently accepting binary content. Case, punctuation, and accents are preserved.

The loader rejects missing files, directories, unsupported extensions, zero-byte
files, corrupt or undecodable content, documents with no readable text, and files
above the size limit. Missing paths raise `FileNotFoundError`; validation and
extraction failures raise `ValueError` with a descriptive message.

The default limit is **10 MiB (10 × 1024 × 1024 bytes)**. A file exactly at the
limit is accepted if otherwise valid. Local callers can override the positive
integer byte limit:

```python
loaded = load_document("data/sample_documents/company_info.txt", max_file_size=1024 * 1024)
```

## Metadata

| Field | Meaning |
| --- | --- |
| `source` | Source filename, without parent directories |
| `file_type` | Lowercase extension without the dot |
| `file_path` | Absolute source path; backend uploads use their final raw-storage path |
| `file_size_bytes` | Original file size in bytes |
| `modified_at` | File modification time as a UTC ISO timestamp |
| `file_hash` | SHA-256 of the original file bytes |
| `content_hash` | SHA-256 of cleaned text encoded as UTF-8 |
| `document_id` | The content hash, used as the catalog primary key |
| `word_count` | Number of words in cleaned text using `split()` |
| `character_count` | Number of characters in cleaned text |
| `ingested_at` | UTC ISO timestamp added when the catalog registers the document |

For uploaded files, `modified_at` reflects the server-side file creation process;
it is not the original client file's modification time.

Each processed chunk inherits document metadata and adds `chunk_index`,
`word_start`, `word_end`, `chunk_strategy` (`word`), and `total_chunks`.
Backend chunks also include `original_filename`.

## Processing and chunking

```python
from ingestion.pipeline import process_document

processed = process_document(
    "data/sample_documents/company_info.txt",
    chunk_size=500,
    overlap=100,
    max_file_size=10 * 1024 * 1024,
)

print(processed["source"], processed["total_chunks"])
for chunk in processed["chunks"]:
    print(chunk["chunk_id"], chunk["text"], chunk["metadata"])
```

The processing result contains `source`, document-level `metadata`, `total_chunks`,
and `chunks`. Each chunk contains `chunk_id`, `text`, and `metadata`. Local chunk
IDs follow `<source>_chunk_<index>`; backend chunk IDs use
`<document_id>_chunk_<index>` to avoid collisions between documents.

Chunks contain at most `chunk_size` words, with exactly `overlap` words repeated
between adjacent chunks. The final chunk can be shorter. `word_start` is inclusive
and `word_end` is exclusive, both referring to the cleaned text's `split()` word
list. Chunking covers all words and makes forward progress even with overlap
close to the chunk size.

`chunk_size` must be a positive integer; `overlap` must be an integer satisfying
`0 <= overlap < chunk_size`. Booleans and non-integers are rejected. Direct
chunking of empty text returns an empty list; document processing rejects empty
documents before chunking.

For callers that only need strings or offsets:

```python
from ingestion.chunker import chunk_text, chunk_spans

strings = chunk_text("one two three four five", chunk_size=3, overlap=1)
# ["one two three", "three four five"]
spans = chunk_spans("one two three four five", chunk_size=3, overlap=1)
# First span: {"text": "one two three", "word_start": 0, "word_end": 3}
```

## Persistent catalog and duplicate detection

```python
from ingestion.document_manager import DocumentManager

manager = DocumentManager("data/processed/documents.sqlite3")
result = manager.ingest("data/sample_documents/company_info.txt", chunk_size=500, overlap=100)
document_id = result["document"]["metadata"]["document_id"]

print(result["duplicate"])
metadata_list = manager.list_documents(file_type="txt", limit=20, offset=0)
processed_document = manager.get_document(document_id)
removed = manager.delete_document(document_id)
```

| Method | Behavior |
| --- | --- |
| `ingest(file_path, **processing_options)` | Process and register; return `duplicate` and the stored `document` |
| `register(processed_document, replace=False)` | Register an already processed result, without modifying the caller's object |
| `get_document(document_id)` | Return the complete stored result; raise `KeyError` if missing |
| `list_documents(file_type=None, limit=100, offset=0)` | Return metadata and chunk counts, ordered by ingestion time and ID |
| `delete_document(document_id)` | Delete the catalog record; return whether one existed |

Content hashes detect identical cleaned text across filenames and formats.
Whitespace and Unicode-normalization differences do not create new IDs. Case,
punctuation, and extraction differences can create different IDs; this is exact
normalized-text matching, not semantic similarity detection.

By default, a duplicate returns the first stored document, preserving its source,
ingestion time, and chunk settings. `register(..., replace=True)` explicitly
replaces the matching record and refreshes its ingestion timestamp. It does not
replace graph/vector data. The backend uses this option when indexing a record
previously processed only through the local catalog.

SQLite transactions and the document-ID uniqueness constraint prevent duplicate
catalog rows under concurrent identical ingestion. Only document-ID conflicts
are ignored; unrelated database errors are not silently suppressed. Every
operation commits or rolls back and closes its connection, including on failure.
Failed extraction creates no record. Listing validates positive integer limits
and non-negative integer offsets; file-type filters accept forms such as `txt`
and `.TXT`.

The catalog stores processed chunks and metadata, not copies of original files.
Its deletion method alone does not delete source files or external indexes.

## Backend integration

Start the API from the repository root:

```bash
venv/bin/python -m uvicorn backend.main:app --reload
```

Actual upload/indexing requires the spaCy English model, the configured
Sentence Transformers embedding model, ChromaDB, and a reachable Neo4j instance.
Set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` using the repository's
`.env.example`. The embedding model may need a download on first use.

Question-answering imports are deferred until `/ask`, so missing Gemini setup
does not block ingestion-route startup. `/ask` still needs its own SDK and
credentials. The existing dependency list installs `google-generativeai`, while
the answer generator imports `google.genai`; this unresolved SDK mismatch is
outside the ingestion implementation.

| Endpoint | Ingestion and management behavior |
| --- | --- |
| `POST /upload` | Validate and chunk, check the catalog, index vectors and graph, then register successful ingestion |
| `GET /documents` | List vector-indexed documents and enrich matching entries with catalog metadata |
| `PUT /documents/{document_id}` | Index a replacement before deleting the previous version |
| `DELETE /documents/{document_id}` | Remove graph provenance, vector chunks, raw files, then the catalog record |

Uploads read at most the configured limit plus one byte from `UploadFile` for
validation. Client filenames are reduced to a basename. Temporary validation files
are removed automatically. New raw files are stored as
`data/raw/<document_id>.<extension>`; the catalog is stored at
`data/processed/documents.sqlite3`. Existing vector/graph storage remains managed
by its respective modules.

Successful upload responses retain the existing fields: `message`, `document_id`,
`filename`, `chunks_created`, `vectors_indexed`, `graph_storage`,
`entities_extracted`, and `relationships_extracted`. The added `duplicate` field
is `false` for new indexing and `true` for already indexed content. Duplicate
uploads skip indexing and return the original stored indexing result and canonical
filename, even when the incoming filename differs.

Document lists retain `total_documents` and `documents`; entries retain
`document_id`, `filename`, and `chunks_indexed`, with `metadata` added where a
catalog record exists. Local catalog-only documents are not included in the HTTP
list until indexed. Older vector-only documents remain listed and deletable;
they are not automatically migrated or deduplicated against the new content IDs.

Replacement filenames must match the existing filename, ignoring case. Identical
content returns an already-up-to-date response. Changed content receives a new
ID. Replacement content belonging to another document under a different filename
is rejected. If new indexing fails, the old document remains; if old-version
cleanup fails after new indexing succeeds, the error reports both IDs.

Deletion also accepts an existing catalog record when vectors are already absent,
allowing cleanup retries after partial deletion. If neither a vector entry nor a
catalog record exists, deletion returns HTTP 404.

### HTTP error behavior

| Status | Examples |
| --- | --- |
| 400 | Unsupported format, empty/unreadable document, or mismatched replacement filename |
| 404 | Document ID not found for replacement or deletion |
| 409 | Replacement duplicates another document under a different filename |
| 413 | Upload exceeds the byte limit |
| 500 | Indexing, storage, or cleanup failure |

On indexing failure, the backend attempts to remove partial graph and vector
writes. If compensation succeeds, it removes the raw file; if compensation fails,
it reports the affected ID and cleanup errors and keeps the raw file for recovery.
It registers a new catalog record only after graph/vector indexing succeeds.

## Reliability boundaries

- PDF extraction does not perform OCR; image-only documents require a separate OCR stage.
- DOCX extraction covers body paragraphs and tables; it does not promise complete extraction of headers, footers, embedded objects, or every layout feature.
- Cleaning flattens paragraph/line spacing and removes format characters; the processed text is not a layout-preserving copy.
- File byte limits do not bound decompressed content or extraction memory. Processing and chunking operate in memory rather than streaming document text.
- SQLite, ChromaDB, Neo4j, and the filesystem do not share an atomic transaction. Process crashes or external-service outages can require manual reconciliation.
- Concurrent standalone catalog calls are tested, but cross-store concurrency is not coordinated. Do not assume safe concurrent management of the same document across backend workers and direct catalog callers.
- Catalog duplicate responses assume external indexes still match the recorded successful indexing result; they do not revalidate external stores.

## Tests and verification

Run the ingestion and graph-workflow compatibility tests:

```bash
venv/bin/python -m pytest tests/test_document_loader.py tests/test_chunker.py tests/test_pipeline.py tests/test_document_ingestion.py tests/test_workflow.py -q
```

Run HTTP integration tests with real ingestion and SQLite and test doubles for
Neo4j and ChromaDB:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python -m pytest tests/test_backend_ingestion.py -q
```

Coverage includes generated PDF/DOCX/TXT documents of 1, 200, and 3,000 words;
a 100,000-word TXT; supported encodings; empty/corrupt/password-protected files;
exact byte-limit boundaries; metadata hashes; DOCX table order; chunk coverage,
overlap, and offsets; invalid options; persistence; concurrent duplicate ingestion;
connection closure on success/failure; upload/list/delete/replace behavior;
legacy entries; previously processed catalog records; failure compensation;
and preservation of originals after failed replacements.

The latest combined verification completed with **95 tests passed**:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python -m pytest tests --ignore=tests/test_rag_pipeline.py -q
```

The RAG test file is excluded because of the existing `google.genai` SDK mismatch.
These results verify local ingestion and isolated API behavior, not live Neo4j,
ChromaDB, embedding downloads, or Gemini integration.
