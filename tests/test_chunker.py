from ingestion.chunker import chunk_text


def test_chunk_text_splits_long_text():
    text = " ".join([f"word{i}" for i in range(1200)])

    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 500
    assert len(chunks[1].split()) == 500
    assert len(chunks[2].split()) == 400


def test_chunk_text_preserves_overlap():
    text = " ".join([f"word{i}" for i in range(600)])

    chunks = chunk_text(text, chunk_size=500, overlap=100)

    first_words = chunks[0].split()
    second_words = chunks[1].split()

    assert first_words[-100:] == second_words[:100]


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_chunk_text_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=100, overlap=100)
