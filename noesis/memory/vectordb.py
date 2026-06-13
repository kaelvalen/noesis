"""Vector DB tier using Qdrant in embedded mode.

No external Qdrant server is required: storage is disk-backed and lives in
the project directory.
"""

import os
import warnings

import torch


class NoesisVectorDB:
    def __init__(self, path="noesis_vectordb", dim=1024):
        self.path = path
        self.dim = dim
        self.collection = "noesis_memory"
        self._client = None

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(path=path)

            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception as exc:
            warnings.warn(f"Qdrant embedded unavailable: {exc}. VectorDB disabled.")
            self.client = None

    def ingest(self, chunks, embeddings):
        """Ingest text chunks with precomputed embeddings."""
        if self.client is None:
            return False

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={"text": chunk},
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return True

    def search(self, query_embedding, k=10):
        """Return list of (text, score) tuples."""
        if self.client is None:
            return []

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu()

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding.tolist(),
            limit=k,
        )
        return [(r.payload["text"], r.score) for r in results]
