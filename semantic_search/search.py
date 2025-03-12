from .embeddings import Embedder
from .data_processor import DataProcessor
from .backends.cosine import CosineSimilarityBackend
from .backends.faiss_backend import FAISSBackend
from .backends.scann_backend import ScaNNBackend
from .backends.chromadb_backend import ChromaDBBackend

class SemanticSearch:
    def __init__(self, backend="cosine", model_path=None, model_name="all-MiniLM-L6-v2"):
        self.embedder = Embedder(model_path=model_path, model_name=model_name)
        self.backend_name = backend
        self.backend = self._initialize_backend(backend)
        self.index_built = False

    def _initialize_backend(self, backend):
        if backend == "cosine":
            return CosineSimilarityBackend()
        elif backend == "faiss":
            return FAISSBackend(dim=384)  # Adjust dimension as per your model
        elif backend == "scann":
            return ScaNNBackend(dim=384)
        elif backend == "chromadb":
            return ChromaDBBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def build_index(self, sentences):
        if not sentences:
            raise ValueError("No sentences provided to build the index.")
        try:
            embeddings = self.embedder.encode(sentences)
            if self.backend_name == "chromadb":
                self.backend.add(sentences, embeddings)
            else:
                self.backend.add(embeddings)
            self.index_built = True
        except Exception as e:
            raise RuntimeError("Error building index.") from e

    def build_index_from_csv(self, csv_path, template):
        processor = DataProcessor(csv_path)
        print()
        sentences = processor.format_data(template)
        self.build_index(sentences)

    def query(self, text, top_k=5):
        if not self.index_built:
            raise ValueError("Index has not been built. Please build the index first.")
        if not text:
            raise ValueError("No query text provided.")
        try:
            query_vector = self.embedder.encode([text])[0]
            return self.backend.search(query_vector, top_k)
        except Exception as e:
            raise RuntimeError("Error during query processing.") from e
