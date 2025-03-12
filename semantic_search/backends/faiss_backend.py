import faiss
import numpy as np

class FAISSBackend:
    def __init__(self, dim):
        try:
            self.index = faiss.IndexFlatL2(dim)
        except Exception as e:
            raise RuntimeError("Error initializing FAISS index.") from e

    def add(self, vectors):
        # print(vectors)
        
        # if not vectors:
          
        #     raise ValueError("No vectors provided to add to FAISS index.")
        try:
            print(np.array(vectors))
            
            self.index.add(np.array(vectors))
        except Exception as e:
            raise RuntimeError("Error adding vectors to FAISS index.") from e

    def search(self, query_vector, top_k=5):
        try:
            distances, indices = self.index.search(np.array([query_vector]), top_k)
            return distances[0], indices[0]
        except Exception as e:
            raise RuntimeError("Error during FAISS search.") from e
