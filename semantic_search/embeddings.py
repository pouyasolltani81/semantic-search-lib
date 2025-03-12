try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("The 'sentence-transformers' package is required. Install it via `pip install sentence-transformers`.") from e

class Embedder:
    def __init__(self, model_path=None, model_name="all-MiniLM-L6-v2"):
        try:
            if model_path:
                self.model = SentenceTransformer(model_path)
            else:
                self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError("Failed to load model. Check the model path or model name.") from e

    def encode(self, texts):
        if not texts:
            raise ValueError("No texts provided for encoding.")
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy() if hasattr(embeddings, "cpu") else embeddings
        except Exception as e:
            raise RuntimeError("Error during encoding texts.") from e
