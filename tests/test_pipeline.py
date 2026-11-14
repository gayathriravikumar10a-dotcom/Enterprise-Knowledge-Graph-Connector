from ingestion.pipeline import process_document


def test_process_document():
    result = process_document(
        "data/sample_documents/company_info.txt",
        chunk_size=30,
        overlap=5
    )

    assert result["source"] == "company_info.txt"
    assert result["total_chunks"] > 0

    first_chunk = result["chunks"][0]

    assert "Asha" in first_chunk["text"]
    assert first_chunk["chunk_id"] == "company_info.txt_chunk_0"
    assert first_chunk["metadata"]["file_type"] == "txt"
    assert first_chunk["metadata"]["chunk_index"] == 0
