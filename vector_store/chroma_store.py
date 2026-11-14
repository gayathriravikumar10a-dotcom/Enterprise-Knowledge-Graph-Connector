from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"


class ChromaVectorStore:
    def __init__(
        self,
        collection_name="enterprise_documents",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.embedding_model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Enterprise document embeddings"}
        )

    def _clean_metadata(self, metadata, index):
        cleaned = {}

        for key, value in (metadata or {}).items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = str(value)

        # ChromaDB does not accept empty metadata dictionaries.
        if not cleaned:
            cleaned = {
                "chunk_index": index,
                "source": "enterprise_document"
            }

        return cleaned

    def add_documents(self, chunks):
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            if isinstance(chunk, str):
                content = chunk.strip()
                chunk_id = f"chunk_{index}"
                metadata = {}

            elif isinstance(chunk, dict):
                content = chunk.get("text", "").strip()
                chunk_id = chunk.get("chunk_id", f"chunk_{index}")
                metadata = chunk.get("metadata", {})

            else:
                raise TypeError(
                    f"Unsupported chunk type: {type(chunk).__name__}"
                )

            if not content:
                continue

            ids.append(str(chunk_id))
            documents.append(content)
            metadatas.append(
                self._clean_metadata(metadata, index)
            )

        if not documents:
            return 0

        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(documents)

    def search(self, query, top_k=5):
        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        total_count = self.collection.count()

        if total_count == 0:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, total_count),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []

        for index, chunk_id in enumerate(results["ids"][0]):
            retrieved.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            })

        return retrieved

    def count(self):
        return self.collection.count()

    def delete_collection(self):
        self.client.delete_collection(
            name=self.collection.name
        )
