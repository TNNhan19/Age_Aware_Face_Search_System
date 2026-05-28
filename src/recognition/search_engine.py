import pickle
import time
from pathlib import Path

import numpy as np

from src.recognition.face_embedder import FaceEmbedder


class FaceSearchEngine:
    def __init__(self, gallery_path="models/gallery_embeddings.pkl"):
        """
        Load gallery embeddings and initialize FaceEmbedder.
        """
        self.gallery_path = Path(gallery_path)
        self.embedder = FaceEmbedder()
        self.gallery_data = self._load_gallery()

    def _load_gallery(self):
        """
        Load gallery embedding database from pickle file.
        """
        if not self.gallery_path.exists():
            raise FileNotFoundError(
                f"Gallery embedding file not found: {self.gallery_path}. "
                f"Please run: python -m src.recognition.build_gallery"
            )

        with open(self.gallery_path, "rb") as f:
            gallery_data = pickle.load(f)

        if len(gallery_data) == 0:
            raise ValueError("Gallery embedding database is empty.")

        return gallery_data

    def _cosine_similarity(self, query_embedding, gallery_embedding):
        """
        Compute cosine similarity between two normalized embeddings.

        Since embeddings are already normalized, dot product is enough.
        """
        return float(np.dot(query_embedding, gallery_embedding))

    def search(self, query_image_path, top_k=5):
        """
        Search top-k most similar faces from gallery.

        Args:
            query_image_path: Path to query image.
            top_k: Number of results to return.

        Returns:
            results: List of top-k search results.
            query_info: Face detection information of query image.
            search_time: Search time in seconds.
        """
        start_time = time.time()

        query_embedding, query_info = self.embedder.extract_embedding(query_image_path)

        results = []

        for item in self.gallery_data:
            score = self._cosine_similarity(
                query_embedding,
                item["embedding"]
            )

            results.append({
                "person_id": item["person_id"],
                "image_path": item["image_path"],
                "score": score,
                "face_info": item["face_info"]
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        search_time = time.time() - start_time

        return results, query_info, search_time


if __name__ == "__main__":
    engine = FaceSearchEngine()

    query_path = "data/raw/query/cuong.jpg"
    top_k = 5

    results, query_info, search_time = engine.search(query_path, top_k=top_k)

    print("\nQuery image:", query_path)
    print("Query face info:", query_info)
    print(f"Search time: {search_time:.4f} seconds")

    print(f"\nTop-{top_k} results:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"person_id={result['person_id']} | "
            f"score={result['score']:.4f} | "
            f"path={result['image_path']}"
        )