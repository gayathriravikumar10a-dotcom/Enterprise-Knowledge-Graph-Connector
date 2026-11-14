from ingestion.document_loader import load_document


def test_load_txt_document():
    result = load_document("data/sample_documents/company_info.txt")

    assert "Asha" in result["text"]
    assert "Ravi" in result["text"]
    assert result["metadata"]["file_type"] == "txt"
    assert result["metadata"]["source"] == "company_info.txt"
