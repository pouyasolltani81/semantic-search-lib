try:
    import chromadb
except ImportError:
    chromadb = None

class ChromaDBBackend:
    def __init__(self):
        if chromadb is None:
            raise ImportError("The 'chromadb' package is required for the ChromaDB backend. Install it via `pip install chromadb`.")
        try:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection("semantic_search")
        except Exception as e:
            raise RuntimeError("Error initializing Chroma DB client.") from e

    def add(self, texts, vectors):
        if not vectors or not texts:
            raise ValueError("No texts or vectors provided for Chroma DB backend.")
        try:
            ids = [str(i) for i in range(len(vectors))]
            self.collection.add(
                embeddings=vectors,
                metadatas=[{"text": t} for t in texts],
                ids=ids
            )
        except Exception as e:
            raise RuntimeError("Error adding data to Chroma DB.") from e

    def search(self, query_vector, top_k=5):
        try:
            results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
            distances = results.get('distances', [[]])
            ids = results.get('ids', [[]])
            return distances[0], ids[0]
        except Exception as e:
            raise RuntimeError("Error during Chroma DB search.") from e
