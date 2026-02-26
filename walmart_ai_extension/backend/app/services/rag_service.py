from typing import Any

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None


class RAGService:
    def __init__(self, host: str, port: int):
        self.client = None
        self.collection = None
        if chromadb is not None:
            self.client = chromadb.HttpClient(host=host, port=port)
            self.collection = self.client.get_or_create_collection("retail_knowledge")

    def seed(self) -> None:
        if self.collection is None:
            return
        docs = [
            "Inflation can reduce discretionary demand, especially for price-sensitive categories.",
            "Holiday periods usually create non-linear demand spikes.",
            "Inventory constraints can cause apparent demand drops despite healthy underlying demand.",
        ]
        ids = [f"doc_{i}" for i in range(len(docs))]
        self.collection.upsert(ids=ids, documents=docs)

    def search(self, query: str, k: int = 3) -> list[str]:
        if self.collection is None:
            return []
        out: Any = self.collection.query(query_texts=[query], n_results=k)
        return out.get("documents", [[]])[0]
