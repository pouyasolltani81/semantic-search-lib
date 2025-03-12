import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarityBackend:
    def __init__(self):
        self.embeddings = []

    def add(self, vectors):
        if not vectors:
            raise ValueError("No vectors provided to add to the index.")
        try:
            self.embeddings.extend(vectors)
        except Exception as e:
            raise RuntimeError("Error adding vectors to the cosine backend.") from e

    def search(self, query_vector, top_k=5):
        if not self.embeddings:
            raise ValueError("Index is empty. Please add vectors first.")
        try:
            sims = cosine_similarity([query_vector], self.embeddings)[0]
            indices = np.argsort(sims)[::-1][:top_k]
            return sims[indices], indices
        except Exception as e:
            raise RuntimeError("Error during cosine similarity search.") from e
