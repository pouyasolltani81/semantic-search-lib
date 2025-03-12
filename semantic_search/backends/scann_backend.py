try:
    import scann
except ImportError:
    scann = None

import numpy as np

class ScaNNBackend:
    def __init__(self, dim):
        self.dim = dim
        self.index = None
        if scann is None:
            raise ImportError("ScaNN is not installed. To use the ScaNN backend, install it from source or follow the instructions at https://github.com/google-research/scann")

    def add(self, vectors):
        if not vectors:
            raise ValueError("No vectors provided to add to ScaNN index.")
        try:
            vectors_np = np.array(vectors)
            self.index = scann.scann_ops.builder(vectors_np, 10, "dot_product").build()
        except Exception as e:
            raise RuntimeError("Error building ScaNN index.") from e

    def search(self, query_vector, top_k=5):
        if self.index is None:
            raise ValueError("ScaNN index has not been built yet.")
        try:
            indices, distances = self.index.search(np.array(query_vector), top_k)
            return distances, indices
        except Exception as e:
            raise RuntimeError("Error during ScaNN search.") from e
